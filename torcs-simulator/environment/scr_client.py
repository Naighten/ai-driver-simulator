"""UDP client for the TORCS SCR protocol.

Init format (from working reference):
  SCR(init <19_focus_angles>) -> server responds "***identified***"
  Socket timeout: 1.0s, init retried up to 120 times.
"""

import socket
import time
import logging

from msgParser import MsgParser
from carState import CarState
from carControl import CarControl

logger = logging.getLogger(__name__)


class SCRClient:
    """UDP client for the TORCS SCR protocol.

    Inputs:
        host: TORCS hostname.
        port: SCR UDP port.
        timeout: socket timeout in seconds (default 1.0).
    """

    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.addr = (host, port)
        self.timeout = timeout
        self.parser = MsgParser()
        self.state = CarState()
        self.sock = None

    def _ensure_socket(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(self.timeout)

    def identify(self, attempts: int = 120, delay: float = 0.02) -> bool:
        """Send SCR(init <19_angles>) and wait for ***identified***.

        Retries up to `attempts` times with `delay` seconds between sends.
        """
        self._ensure_socket()

        angles = [-90, -75, -60, -45, -30, -20, -15, -10, -5,
                   0,   5,  10,  15,  20,  30,  45,  60,  75, 90]
        init_msg = self.parser.stringify({"init": angles})
        buf = ("SCR" + init_msg).encode("utf-8")

        for attempt in range(attempts):
            try:
                self.sock.sendto(buf, self.addr)
                resp, _ = self.sock.recvfrom(1000)
            except (OSError, socket.timeout):
                if delay > 0:
                    time.sleep(delay)
                continue

            resp_str = resp.decode("utf-8", errors="ignore") if isinstance(resp, bytes) else str(resp)

            if "***identified***" in resp_str:
                logger.debug(f"Identified after {attempt + 1} attempts")
                return True
            if "***shutdown***" in resp_str or "***restart***" in resp_str:
                logger.warning("TORCS restarting, waiting 1s...")
                time.sleep(1.0)

        return False

    def recv_state(self) -> CarState:
        """Receive and parse sensor state from TORCS."""
        self._ensure_socket()
        raw, _ = self.sock.recvfrom(4096)
        msg = raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
        self.state.setFromMsg(msg)
        return self.state

    def send_action(self, control: CarControl) -> None:
        """Send control commands to TORCS."""
        self._ensure_socket()
        msg = control.toMsg()
        self.sock.sendto(msg.encode("utf-8"), self.addr)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def reset(self) -> None:
        """Send meta=1 to request track reset."""
        control = CarControl(accel=0.0, brake=0.0, gear=1, steer=0.0, meta=1)
        self.send_action(control)
