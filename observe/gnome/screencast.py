"""
Portal-based multi-monitor screencast recording.

Uses xdg-desktop-portal ScreenCast API with PipeWire + GStreamer to record
each monitor as a separate file. This replaces the old GNOME Shell D-Bus approach.

Runtime deps:
  - xdg-desktop-portal with org.freedesktop.portal.ScreenCast
  - Portal backend: xdg-desktop-portal-gnome (or -kde, -wlr, etc.)
  - PipeWire running
  - GStreamer with PipeWire plugin: gst-launch-1.0 pipewiresrc
"""

import asyncio
import logging
import os
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from dbus_next import Variant, introspection
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.dbus import get_monitor_geometries

# Workaround for dbus-next issue #122: portal has properties with hyphens
# (e.g., "power-saver-enabled") which violate strict D-Bus naming validation.
introspection.assert_member_name_valid = lambda name: None

logger = logging.getLogger(__name__)

# Portal D-Bus constants
PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SC_IFACE = "org.freedesktop.portal.ScreenCast"
REQ_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"


@dataclass
class StreamInfo:
    """Information about a single monitor's recording stream."""

    node_id: int
    position: str
    connector: str
    x: int
    y: int
    width: int
    height: int
    temp_path: str

    def final_name(self, time_part: str, duration: int) -> str:
        """Generate the final filename for this stream."""
        return f"{time_part}_{duration}_{self.position}_{self.connector}_screen.webm"


def _get_restore_token_path() -> Path:
    """Get path for restore token storage."""
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        return Path(journal) / "health" / "screencast_restore_token"
    # Fallback to XDG state
    state_home = os.environ.get("XDG_STATE_HOME")
    if not state_home:
        state_home = os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(state_home) / "sunstone" / "screencast_restore_token"


def _load_restore_token() -> str | None:
    """Load restore token from disk."""
    path = _get_restore_token_path()
    try:
        data = path.read_text(encoding="utf-8").strip()
        return data or None
    except (FileNotFoundError, OSError):
        return None


def _save_restore_token(token: str) -> None:
    """Save restore token to disk."""
    path = _get_restore_token_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token.strip() + "\n", encoding="utf-8")
        logger.debug(f"Saved restore token to {path}")
    except OSError as e:
        logger.warning(f"Failed to save restore token: {e}")


def _make_request_handle(bus: MessageBus, token: str) -> str:
    """Compute expected Request object path for a handle_token."""
    sender = bus.unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def _prepare_request_handler(bus: MessageBus, handle: str) -> asyncio.Future:
    """Set up signal handler for Request::Response before calling portal method."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def _message_handler(msg):
        if (
            msg.message_type.name == "SIGNAL"
            and msg.path == handle
            and msg.interface == REQ_IFACE
            and msg.member == "Response"
        ):
            response = msg.body[0]
            results = msg.body[1] if len(msg.body) > 1 else {}
            if not fut.done():
                fut.set_result((int(response), results))
            bus.remove_message_handler(_message_handler)

    bus.add_message_handler(_message_handler)
    return fut


def _variant_or_value(val):
    """Extract value from Variant if needed."""
    if isinstance(val, Variant):
        return val.value
    return val


def _match_streams_to_monitors(streams: list[dict], monitors: list[dict]) -> list[dict]:
    """
    Match portal stream geometries to GDK monitor info.

    Portal streams have position (x, y) and size (width, height).
    GDK monitors have connector IDs and box coordinates.

    Returns streams augmented with connector and position labels.
    """
    matched = []

    for stream in streams:
        props = stream.get("props", {})

        # Extract stream geometry from portal properties
        stream_pos = _variant_or_value(props.get("position", (0, 0)))
        stream_size = _variant_or_value(props.get("size", (0, 0)))

        if isinstance(stream_pos, tuple) and len(stream_pos) >= 2:
            sx, sy = int(stream_pos[0]), int(stream_pos[1])
        else:
            sx, sy = 0, 0

        if isinstance(stream_size, tuple) and len(stream_size) >= 2:
            sw, sh = int(stream_size[0]), int(stream_size[1])
        else:
            sw, sh = 0, 0

        # Find matching monitor by geometry
        best_match = None
        best_overlap = 0

        for monitor in monitors:
            mx1, my1, mx2, my2 = monitor["box"]
            mw, mh = mx2 - mx1, my2 - my1

            # Check if geometries match (within tolerance for scaling)
            if abs(sx - mx1) < 10 and abs(sy - my1) < 10:
                overlap = min(sw, mw) * min(sh, mh)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = monitor

        if best_match:
            stream["connector"] = best_match["id"]
            stream["position_label"] = best_match.get("position", "unknown")
            stream["x"] = best_match["box"][0]
            stream["y"] = best_match["box"][1]
            stream["width"] = best_match["box"][2] - best_match["box"][0]
            stream["height"] = best_match["box"][3] - best_match["box"][1]
        else:
            # Fallback: use stream index as identifier
            stream["connector"] = f"monitor-{stream['idx']}"
            stream["position_label"] = "unknown"
            stream["x"] = sx
            stream["y"] = sy
            stream["width"] = sw
            stream["height"] = sh

        matched.append(stream)

    return matched


class Screencaster:
    """Portal-based multi-monitor screencast manager."""

    def __init__(self):
        self.bus: MessageBus | None = None
        self.session_handle: str | None = None
        self.pw_fd: int | None = None
        self.gst_process: subprocess.Popen | None = None
        self.streams: list[StreamInfo] = []
        self._started = False

    async def connect(self) -> bool:
        """
        Establish D-Bus connection and verify portal availability.

        Returns:
            True if portal is available, False otherwise.
        """
        if self.bus is not None:
            return True

        try:
            self.bus = await MessageBus(
                bus_type=BusType.SESSION,
                negotiate_unix_fd=True,
            ).connect()

            # Verify portal interface exists
            root_intro = await self.bus.introspect(PORTAL_BUS, PORTAL_PATH)
            root_obj = self.bus.get_proxy_object(PORTAL_BUS, PORTAL_PATH, root_intro)
            root_obj.get_interface(SC_IFACE)
            return True

        except Exception as e:
            logger.error(f"Portal not available: {e}")
            self.bus = None
            return False

    async def start(
        self,
        base_path: str,
        timestamp: str,
        framerate: int = 1,
        draw_cursor: bool = True,
    ) -> list[StreamInfo]:
        """
        Start screencast recording for all monitors.

        Args:
            base_path: Directory for output files
            timestamp: Timestamp prefix for temp files (HHMMSS format)
            framerate: Frames per second (default: 1)
            draw_cursor: Whether to draw mouse cursor (default: True)

        Returns:
            List of StreamInfo for each monitor being recorded.

        Raises:
            RuntimeError: If recording fails to start.
        """
        if not await self.connect():
            raise RuntimeError("Portal not available")

        # Get monitor info from GDK for connector IDs
        try:
            monitors = get_monitor_geometries()
        except Exception as e:
            logger.warning(f"Failed to get monitor geometries: {e}")
            monitors = []

        # Get portal interface
        root_intro = await self.bus.introspect(PORTAL_BUS, PORTAL_PATH)
        root_obj = self.bus.get_proxy_object(PORTAL_BUS, PORTAL_PATH, root_intro)
        screencast = root_obj.get_interface(SC_IFACE)

        # 1) CreateSession
        create_token = "h_" + uuid.uuid4().hex
        create_handle = _make_request_handle(self.bus, create_token)
        create_fut = _prepare_request_handler(self.bus, create_handle)

        create_opts = {
            "handle_token": Variant("s", create_token),
            "session_handle_token": Variant("s", "s_" + uuid.uuid4().hex),
        }

        await screencast.call_create_session(create_opts)
        resp, results = await create_fut
        if resp != 0:
            raise RuntimeError(f"CreateSession failed with code {resp}")

        self.session_handle = str(_variant_or_value(results.get("session_handle")))
        if not self.session_handle:
            raise RuntimeError("CreateSession returned no session_handle")

        logger.debug(f"Portal session: {self.session_handle}")

        # 2) SelectSources
        restore_token = _load_restore_token()
        if restore_token:
            logger.debug("Using saved restore token")

        cursor_mode = 1 if draw_cursor else 0

        select_token = "h_" + uuid.uuid4().hex
        select_handle = _make_request_handle(self.bus, select_token)
        select_fut = _prepare_request_handler(self.bus, select_handle)

        select_opts = {
            "handle_token": Variant("s", select_token),
            "types": Variant("u", 1),  # 1 = MONITOR
            "multiple": Variant("b", True),
            "cursor_mode": Variant("u", cursor_mode),
            "persist_mode": Variant("u", 2),  # Persist until revoked
        }
        if restore_token:
            select_opts["restore_token"] = Variant("s", restore_token)

        await screencast.call_select_sources(self.session_handle, select_opts)
        resp, _ = await select_fut
        if resp != 0:
            await self._close_session()
            raise RuntimeError(f"SelectSources failed with code {resp}")

        # 3) Start
        start_token = "h_" + uuid.uuid4().hex
        start_handle = _make_request_handle(self.bus, start_token)
        start_fut = _prepare_request_handler(self.bus, start_handle)

        start_opts = {"handle_token": Variant("s", start_token)}
        await screencast.call_start(self.session_handle, "", start_opts)
        resp, results = await start_fut
        if resp != 0:
            await self._close_session()
            raise RuntimeError(f"Start failed with code {resp}")

        portal_streams = _variant_or_value(results.get("streams")) or []
        if not portal_streams:
            await self._close_session()
            raise RuntimeError("Start returned no streams")

        # Save new restore token if provided
        new_token = _variant_or_value(results.get("restore_token"))
        if isinstance(new_token, str) and new_token.strip():
            _save_restore_token(new_token)

        # Parse streams
        stream_info = []
        for idx, stream in enumerate(portal_streams):
            try:
                node_id = int(stream[0])
                props = stream[1] if len(stream) > 1 else {}
                stream_info.append({"idx": idx, "node_id": node_id, "props": props})
            except Exception as e:
                logger.warning(f"Could not parse stream {idx}: {e}")

        if not stream_info:
            await self._close_session()
            raise RuntimeError("No valid streams found")

        # Match streams to monitors
        stream_info = _match_streams_to_monitors(stream_info, monitors)

        logger.info(f"Portal returned {len(stream_info)} stream(s)")

        # 4) OpenPipeWireRemote
        fd_obj = await screencast.call_open_pipe_wire_remote(self.session_handle, {})
        if hasattr(fd_obj, "take"):
            self.pw_fd = fd_obj.take()
        else:
            self.pw_fd = int(fd_obj)

        # 5) Build GStreamer pipeline
        self.streams = []
        pipeline_parts = []

        for info in stream_info:
            node_id = info["node_id"]
            position = info["position_label"]
            connector = info["connector"]

            # Temp file: .HHMMSS_position_connector.webm
            temp_path = os.path.join(
                base_path, f".{timestamp}_{position}_{connector}.webm"
            )

            stream_obj = StreamInfo(
                node_id=node_id,
                position=position,
                connector=connector,
                x=info["x"],
                y=info["y"],
                width=info["width"],
                height=info["height"],
                temp_path=temp_path,
            )
            self.streams.append(stream_obj)

            # GStreamer branch for this stream
            # VP8 encoding optimized for screen content
            branch = (
                f"pipewiresrc fd={self.pw_fd} path={node_id} ! "
                f"videorate ! video/x-raw,framerate={framerate}/1 ! "
                f"videoconvert ! vp8enc end-usage=cq cq-level=4 max-quantizer=15 "
                f"keyframe-max-dist=30 static-threshold=100 ! webmmux ! "
                f"filesink location={temp_path}"
            )
            pipeline_parts.append(branch)

            logger.info(f"  Stream {node_id}: {position} ({connector}) -> {temp_path}")

        pipeline_str = " ".join(pipeline_parts)
        cmd = ["gst-launch-1.0", "-e"] + pipeline_str.split()

        try:
            self.gst_process = subprocess.Popen(
                cmd,
                pass_fds=(self.pw_fd,),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            await self._close_session()
            raise RuntimeError("gst-launch-1.0 not found")
        except Exception as e:
            await self._close_session()
            raise RuntimeError(f"Failed to start GStreamer: {e}")

        # Brief delay to check for immediate failure
        await asyncio.sleep(0.2)
        if self.gst_process.poll() is not None:
            stderr = (
                self.gst_process.stderr.read().decode()
                if self.gst_process.stderr
                else ""
            )
            await self._close_session()
            raise RuntimeError(f"GStreamer exited immediately: {stderr[:200]}")

        self._started = True
        return self.streams

    async def stop(self) -> list[StreamInfo]:
        """
        Stop screencast recording gracefully.

        Returns:
            List of StreamInfo with temp_path for finalization.
        """
        streams = self.streams.copy()

        # Stop GStreamer with SIGINT for clean EOS
        if self.gst_process and self.gst_process.poll() is None:
            try:
                self.gst_process.send_signal(signal.SIGINT)
                # Wait up to 5 seconds for clean shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.gst_process.wait),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("GStreamer did not exit cleanly, killing")
                    self.gst_process.kill()
                    self.gst_process.wait()
            except Exception as e:
                logger.warning(f"Error stopping GStreamer: {e}")

        self.gst_process = None

        # Close PipeWire fd
        if self.pw_fd is not None:
            try:
                os.close(self.pw_fd)
            except OSError:
                pass
            self.pw_fd = None

        # Close portal session
        await self._close_session()

        self.streams = []
        self._started = False

        return streams

    async def _close_session(self):
        """Close the portal session."""
        if self.session_handle and self.bus:
            try:
                session_intro = await self.bus.introspect(
                    PORTAL_BUS, self.session_handle
                )
                session_obj = self.bus.get_proxy_object(
                    PORTAL_BUS, self.session_handle, session_intro
                )
                session_iface = session_obj.get_interface(SESSION_IFACE)
                await session_iface.call_close()
            except Exception:
                pass
        self.session_handle = None

    def is_healthy(self) -> bool:
        """Check if recording is still running."""
        if not self._started:
            return False
        if self.gst_process is None:
            return False
        return self.gst_process.poll() is None
