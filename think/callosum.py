# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Callosum: WebSocket-like broadcast message bus over Unix domain sockets.

Provides real-time event distribution across solstone services using a simple
broadcast protocol. All messages require 'tract' and 'event' fields.
"""

import json
import logging
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

from think.utils import get_journal, now_ms

logger = logging.getLogger(__name__)


class CallosumServer:
    """Broadcast message bus over Unix domain socket.

    Uses a single writer thread to serialize all broadcasts, preventing
    race conditions when multiple client handler threads call broadcast()
    concurrently.
    """

    def __init__(self, socket_path: Path | None = None):
        if socket_path is None:
            socket_path = Path(get_journal()) / "health" / "callosum.sock"

        self.socket_path = Path(socket_path)
        self.clients: list[socket.socket] = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.server_socket: socket.socket | None = None

        # Broadcast queue and writer thread for serialized sends
        self.broadcast_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.writer_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the broadcast server."""
        # Ensure health directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Create Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(str(self.socket_path))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic checks for stop_event

        # Start writer thread before accepting connections
        self.writer_thread = threading.Thread(
            target=self._writer_loop, name="callosum-writer", daemon=True
        )
        self.writer_thread.start()

        logger.info(f"Callosum listening on {self.socket_path}")

        try:
            while not self.stop_event.is_set():
                try:
                    conn, _ = self.server_socket.accept()
                    # Handle client in background thread
                    threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.error(f"Accept error: {e}")
        finally:
            self.server_socket.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

    def _handle_client(self, conn: socket.socket) -> None:
        """Handle a client connection."""
        with self.lock:
            self.clients.append(conn)

        logger.debug(f"Client connected ({len(self.clients)} total)")

        try:
            # Read from client (they might send messages or just listen)
            # Short timeout allows periodic stop_event checks; also used by _writer_loop for sends
            conn.settimeout(2.0)
            buffer = ""
            while not self.stop_event.is_set():
                try:
                    data = conn.recv(4096)
                    if not data:
                        break

                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                message = json.loads(line)
                                self.broadcast(message)
                            except json.JSONDecodeError:
                                pass  # Silent failure - avoid feedback loops
                except socket.timeout:
                    continue
        except Exception as e:
            logger.debug(f"Client error: {e}")
        finally:
            with self.lock:
                if conn in self.clients:
                    self.clients.remove(conn)
            try:
                conn.close()
            except Exception:
                pass
            logger.debug(f"Client disconnected ({len(self.clients)} remaining)")

    def _writer_loop(self) -> None:
        """Dedicated writer thread that serializes all broadcasts.

        Drains the broadcast queue and sends each message to all clients.
        This ensures no interleaving of messages when multiple client
        handler threads call broadcast() concurrently.
        """
        while not self.stop_event.is_set():
            try:
                message = self.broadcast_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._send_to_clients(message)

    def _send_to_clients(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients, removing dead ones.

        This method handles the actual socket I/O and dead client cleanup.
        Called by _writer_loop for each queued message.

        Args:
            message: The message dict to send (will be JSON serialized)
        """
        # Serialize once for all clients
        data = (json.dumps(message) + "\n").encode("utf-8")

        # Snapshot client list under lock
        with self.lock:
            clients_to_send = list(self.clients)

        # Send to all clients, tracking failures
        dead_clients = []
        for client in clients_to_send:
            try:
                # Set per-send timeout to prevent blocking on slow clients
                client.settimeout(2.0)
                client.sendall(data)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                dead_clients.append(client)

        # Clean up dead clients under lock
        if dead_clients:
            with self.lock:
                for client in dead_clients:
                    if client in self.clients:
                        self.clients.remove(client)
                    try:
                        client.close()
                    except Exception:
                        pass

    def broadcast(self, message: dict[str, Any]) -> bool:
        """Queue message for broadcast to all connected clients.

        Returns immediately after queueing. The writer thread handles
        actual transmission to ensure serialized, non-interleaved sends.

        Args:
            message: dict with required 'tract' and 'event' fields

        Returns:
            True if queued successfully, False if validation failed or queue full
        """
        # Validate required fields
        if "tract" not in message or "event" not in message:
            logger.warning("Skipping message without tract/event fields")
            return False

        # Add timestamp if not present
        if "ts" not in message:
            message["ts"] = now_ms()

        # Queue for writer thread
        try:
            self.broadcast_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning(
                f"Broadcast queue full, dropping: {message.get('tract')}/{message.get('event')}"
            )
            return False

    def stop(self) -> None:
        """Stop the server and writer thread."""
        self.stop_event.set()

        # Wait for writer thread to finish
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=1.0)
            if self.writer_thread.is_alive():
                logger.warning("Writer thread did not stop cleanly")


class CallosumConnection:
    """Lock-free bidirectional connection to Callosum.

    Messages are sent via a queue to avoid blocking. A background thread handles
    connection management, queue draining, and message receiving. Messages are
    dropped (with debug logging) when disconnected.
    """

    def __init__(self, socket_path: Path | None = None):
        """Initialize connection (does not connect immediately).

        Args:
            socket_path: Path to Unix socket (defaults to $JOURNAL_PATH/health/callosum.sock)
        """
        if socket_path is None:
            socket_path = Path(get_journal()) / "health" / "callosum.sock"

        self.socket_path = Path(socket_path)
        self.send_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.callback: Callable[[dict[str, Any]], Any] | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self, callback: Callable[[dict[str, Any]], Any] | None = None) -> None:
        """Start background thread for sending and receiving.

        Thread will auto-connect with retry and drain the send queue even when
        disconnected (dropping messages with debug logging).

        Args:
            callback: Optional function to process received messages
        """
        if self.thread and self.thread.is_alive():
            return  # Already started

        self.callback = callback
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self) -> None:
        """Main loop: drain queue, connect/reconnect, receive when connected."""
        sock: socket.socket | None = None
        buffer = ""
        last_connect_attempt = 0.0

        while True:
            # Try to connect if not connected (rate limited to 1/sec)
            if not sock and time.time() - last_connect_attempt > 1.0:
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.connect(str(self.socket_path))
                    sock.settimeout(0.1)  # Short timeout for responsive queue draining
                except Exception as e:
                    logger.info(f"Connection attempt failed to {self.socket_path}: {e}")
                    if sock:
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = None
                    last_connect_attempt = time.time()

            # ALWAYS drain queue (send if connected, drop if not)
            try:
                msg = self.send_queue.get(timeout=0.1)
                if sock:
                    try:
                        line = json.dumps(msg) + "\n"
                        sock.sendall(line.encode("utf-8"))
                    except Exception as e:
                        detail = ""
                        if msg.get("tract") == "logs" and msg.get("event") == "line":
                            detail = f": {msg.get('line', '')[:100]}"
                        logger.info(
                            f"Send {e} for {msg.get('tract')}/{msg.get('event')}{detail}"
                        )
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = None
                else:
                    # Not connected, drop message
                    logger.info(
                        f"Dropping message (not connected): "
                        f"{msg.get('tract')}/{msg.get('event')}"
                    )
            except queue.Empty:
                # Queue is empty - check if we should exit
                if self.stop_event.is_set():
                    break
                # Otherwise continue to receive

            # Receive incoming messages (only if connected)
            if sock:
                try:
                    data = sock.recv(4096)
                    if not data:
                        # Connection closed by server
                        logger.debug("Connection closed by server")
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = None
                        buffer = ""  # Clear partial data from old connection
                        continue

                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip() and self.callback:
                            try:
                                message = json.loads(line)
                                self.callback(message)
                            except json.JSONDecodeError:
                                pass  # Silent failure - avoid feedback loops
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                except socket.timeout:
                    continue  # Normal, just loop back to drain queue
                except Exception as e:
                    logger.info(f"Receive error: {e}")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                    buffer = ""  # Clear partial data from old connection

        # Cleanup on stop
        if sock:
            try:
                sock.close()
            except Exception:
                pass

    def emit(self, tract: str, event: str, **fields) -> bool:
        """Emit message via send queue.

        Returns immediately after queueing. Requires start() to be called first.

        Args:
            tract: Message category/namespace
            event: Event type
            **fields: Additional message fields

        Returns:
            True if queued successfully, False if thread not running or queue full
        """
        if not self.thread or not self.thread.is_alive():
            logger.warning(f"Thread not running, dropping emit: {tract}/{event}")
            return False

        message = {"tract": tract, "event": event, **fields}
        try:
            self.send_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning(f"Queue full, dropping emit: {tract}/{event}")
            return False

    def stop(self) -> None:
        """Stop background thread gracefully, draining queue first."""
        if not self.thread:
            return

        self.stop_event.set()
        self.thread.join(timeout=0.5)

        if self.thread.is_alive():
            logger.warning("Background thread did not stop cleanly")


def callosum_send(
    tract: str,
    event: str,
    socket_path: Path | None = None,
    timeout: float = 2.0,
    **fields,
) -> bool:
    """Send single message via ephemeral Callosum connection.

    Opens connection, sends message, closes. For one-off sends.
    For frequent sends, use CallosumConnection with start() + emit().

    Args:
        tract: Message category/namespace
        event: Event type
        socket_path: Optional socket path (defaults to $JOURNAL_PATH/health/callosum.sock)
        timeout: Connection timeout in seconds (default: 2.0)
        **fields: Additional message fields

    Returns:
        True if sent successfully, False if connection/send failed
    """
    if socket_path is None:
        socket_path = Path(get_journal()) / "health" / "callosum.sock"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(str(socket_path))

        message = {"tract": tract, "event": event, **fields}
        line = json.dumps(message) + "\n"
        sock.sendall(line.encode("utf-8"))
        sock.close()
        return True
    except Exception as e:
        logger.debug(f"callosum_send() failed: {e}")
        return False


def main() -> None:
    """CLI entry point."""
    import argparse

    from think.utils import setup_cli

    parser = argparse.ArgumentParser(description="Callosum message bus")
    setup_cli(parser)  # Handles logging setup based on -v/-d flags

    server = CallosumServer()

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Callosum")
        server.stop()


if __name__ == "__main__":
    main()
