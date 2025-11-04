"""Callosum: WebSocket-like broadcast message bus over Unix facet sockets.

Provides real-time event distribution across Sunstone services using a simple
broadcast protocol. All messages require 'tract' and 'event' fields.
"""

import json
import logging
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CallosumServer:
    """Broadcast message bus over Unix facet socket."""

    def __init__(self, socket_path: Optional[Path] = None):
        if socket_path is None:
            journal = os.getenv("JOURNAL_PATH")
            if not journal:
                raise ValueError("JOURNAL_PATH not set")
            socket_path = Path(journal) / "health" / "callosum.sock"

        self.socket_path = Path(socket_path)
        self.clients: List[socket.socket] = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.server_socket: Optional[socket.socket] = None

    def start(self) -> None:
        """Start the broadcast server."""
        # Ensure health directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Create Unix facet socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(str(self.socket_path))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic checks for stop_event

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
            conn.settimeout(60.0)
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
                                logger.warning(f"Invalid JSON: {line}")
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

    def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        # Validate required fields
        if "tract" not in message or "event" not in message:
            logger.warning("Skipping message without tract/event fields")
            return

        # Add timestamp if not present
        if "ts" not in message:
            message["ts"] = int(time.time() * 1000)

        # Serialize to JSON line
        line = json.dumps(message) + "\n"
        data = line.encode("utf-8")

        # Broadcast to all clients
        with self.lock:
            dead_clients = []
            for client in self.clients:
                try:
                    client.sendall(data)
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    dead_clients.append(client)

            # Clean up dead clients
            for client in dead_clients:
                if client in self.clients:
                    self.clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop the server."""
        self.stop_event.set()


class CallosumConnection:
    """Lock-free bidirectional connection to Callosum.

    Messages are sent via a queue to avoid blocking. A background thread handles
    connection management, queue draining, and message receiving. Messages are
    dropped (with debug logging) when disconnected.
    """

    def __init__(self, socket_path: Optional[Path] = None):
        """Initialize connection (does not connect immediately).

        Args:
            socket_path: Path to Unix socket (defaults to $JOURNAL_PATH/health/callosum.sock)
        """
        if socket_path is None:
            journal = os.getenv("JOURNAL_PATH")
            if not journal:
                raise ValueError("JOURNAL_PATH not set")
            socket_path = Path(journal) / "health" / "callosum.sock"

        self.socket_path = Path(socket_path)
        self.send_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.callback: Optional[Callable[[Dict[str, Any]], Any]] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self, callback: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
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
        sock: Optional[socket.socket] = None
        buffer = ""
        last_connect_attempt = 0.0

        while not self.stop_event.is_set():
            # Try to connect if not connected (rate limited to 1/sec)
            if not sock and time.time() - last_connect_attempt > 1.0:
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.connect(str(self.socket_path))
                    sock.settimeout(0.1)  # Short timeout for responsive queue draining
                except Exception as e:
                    logger.info(f"Connection attempt failed: {e}")
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
                        logger.info(f"Send failed, reconnecting: {e}")
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
                pass  # Normal, continue to receive

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
                        continue

                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                message = json.loads(line)
                                if self.callback:
                                    try:
                                        self.callback(message)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON: {line}")
                except socket.timeout:
                    continue  # Normal, just loop back to drain queue
                except Exception as e:
                    logger.info(f"Receive error: {e}")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None

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
        """Stop background thread gracefully."""
        if not self.thread:
            return

        self.stop_event.set()
        self.thread.join(timeout=2)

        if self.thread.is_alive():
            logger.warning("Background thread did not stop cleanly")


def callosum_send(
    tract: str,
    event: str,
    socket_path: Optional[Path] = None,
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
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            logger.warning("JOURNAL_PATH not set, cannot send message")
            return False
        socket_path = Path(journal) / "health" / "callosum.sock"

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
    args = setup_cli(parser)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = CallosumServer()

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Callosum")
        server.stop()


if __name__ == "__main__":
    main()
