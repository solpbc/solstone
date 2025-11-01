"""Callosum: WebSocket-like broadcast message bus over Unix domain sockets.

Provides real-time event distribution across Sunstone services using a simple
broadcast protocol. All messages require 'tract' and 'event' fields.
"""

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CallosumServer:
    """Broadcast message bus over Unix domain socket."""

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

        # Create Unix domain socket
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
    """Unified bidirectional connection to Callosum.

    Every connection can both emit and receive messages. A background receive loop
    always runs to drain the socket buffer (preventing TCP backpressure), with
    optional message processing via callback.
    """

    def __init__(
        self,
        socket_path: Optional[Path] = None,
        callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        if socket_path is None:
            journal = os.getenv("JOURNAL_PATH")
            if not journal:
                raise ValueError("JOURNAL_PATH not set")
            socket_path = Path(journal) / "health" / "callosum.sock"

        self.socket_path = Path(socket_path)
        self.callback = callback
        self.sock: Optional[socket.socket] = None
        self.receive_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

    def connect(self) -> None:
        """Establish connection and start background receive loop."""
        with self.lock:
            if self.sock:
                return  # Already connected

            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(str(self.socket_path))
                self.sock.settimeout(1.0)

                # Always start receive loop to drain socket buffer
                self.receive_thread = threading.Thread(
                    target=self._receive_loop, daemon=True
                )
                self.receive_thread.start()

                logger.debug(f"Connected to Callosum at {self.socket_path}")
            except Exception as e:
                logger.debug(f"Failed to connect to Callosum: {e}")
                self.sock = None
                raise

    def _receive_loop(self) -> None:
        """Background thread that continuously drains socket buffer."""
        buffer = ""
        while not self.stop_event.is_set():
            try:
                data = self.sock.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            # Process if callback provided, otherwise discard
                            if self.callback:
                                self.callback(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {line}")
            except socket.timeout:
                continue  # Allows checking stop_event periodically
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.debug(f"Receive loop error: {e}")
                break

    def emit(self, tract: str, event: str, **fields) -> None:
        """Emit event to Callosum (auto-connects if needed)."""
        message = {"tract": tract, "event": event, **fields}

        with self.lock:
            # Auto-connect on first emit
            if not self.sock:
                try:
                    self.connect()
                except Exception:
                    return  # Silent failure - bus not running

            try:
                line = json.dumps(message) + "\n"
                self.sock.sendall(line.encode("utf-8"))
            except Exception as e:
                logger.debug(f"Failed to emit to Callosum: {e}")
                # Mark connection as dead - will reconnect on next emit
                self.sock = None

    def close(self) -> None:
        """Close connection and stop receive loop."""
        self.stop_event.set()

        if self.receive_thread:
            self.receive_thread.join(timeout=2)
            if self.receive_thread.is_alive():
                logger.warning("Receive thread did not stop cleanly")

        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None


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
