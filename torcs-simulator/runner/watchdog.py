"""Watchdog thread for monitoring TORCS environment health.

Monitors two conditions in a background thread:
  1. UDP timeout — no response from TORCS within timeout_sec.
  2. Stuck detection — speed below threshold for stuck_steps consecutive steps.

On timeout: triggers reconnect up to max_reconnect_attempts.
On stuck: triggers env reset (meta=1).

All events are logged with timestamp and reason.
"""

import time
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class Watchdog:
    """Monitor a TorcsEnv instance for timeouts and stuck conditions.

    Runs in a separate thread, checking a shared status dictionary
    updated by the environment worker.

    Inputs:
        timeout_sec: max seconds without UDP response before reconnect.
        stuck_speed_threshold: speed (km/h) below which car is stuck.
        stuck_steps: consecutive stuck steps before reset.
        max_reconnect_attempts: max reconnection attempts before marking dead.
        reconnect_delay_sec: delay between reconnect attempts.
        on_timeout: callback(stage: str, attempt: int) for reconnect logic.
        on_stuck: callback() for reset logic.
        on_dead: callback() when max reconnects exhausted.
    """

    def __init__(
        self,
        timeout_sec: float = 30.0,
        stuck_speed_threshold: float = 2.0,
        stuck_steps: int = 100,
        max_reconnect_attempts: int = 5,
        reconnect_delay_sec: float = 3.0,
        on_timeout: Optional[Callable] = None,
        on_stuck: Optional[Callable] = None,
        on_dead: Optional[Callable] = None,
    ):
        self.timeout_sec = timeout_sec
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_steps = stuck_steps
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self._on_timeout = on_timeout
        self._on_stuck = on_stuck
        self._on_dead = on_dead

        self._last_response_time = time.time()
        self._stuck_counter = 0
        self._reconnect_attempts = 0
        self._dead = False
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._current_speed = 0.0

    def update_heartbeat(self) -> None:
        """Called by the worker on every successful UDP receive."""
        with self._lock:
            self._last_response_time = time.time()
            self._reconnect_attempts = 0

    def update_speed(self, speed: float) -> None:
        """Called by the worker with current speedX."""
        with self._lock:
            self._current_speed = speed

    def is_dead(self) -> bool:
        """Return True if max reconnect attempts exceeded."""
        with self._lock:
            return self._dead

    def start(self) -> None:
        """Start the watchdog monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Watchdog stopped")

    def _run(self) -> None:
        """Main monitoring loop."""
        while self._running:
            time.sleep(0.5)

            do_timeout_cb = False
            timeout_args = None
            do_dead_cb = False
            do_stuck_cb = False

            with self._lock:
                if self._dead:
                    continue

                now = time.time()
                time_since_response = now - self._last_response_time

                if time_since_response > self.timeout_sec:
                    self._reconnect_attempts += 1
                    attempt = self._reconnect_attempts
                    logger.warning(
                        f"UDP timeout ({time_since_response:.1f}s) "
                        f"reconnect attempt {attempt}/{self.max_reconnect_attempts}"
                    )

                    if attempt >= self.max_reconnect_attempts:
                        self._dead = True
                        do_dead_cb = True
                    elif self._on_timeout:
                        do_timeout_cb = True
                        timeout_args = ("timeout", attempt)

                if self._current_speed < self.stuck_speed_threshold:
                    self._stuck_counter += 1
                    if self._stuck_counter >= self.stuck_steps:
                        logger.warning(
                            f"Car stuck (speed={self._current_speed:.1f} km/h "
                            f"for {self._stuck_counter} steps). Resetting."
                        )
                        self._stuck_counter = 0
                        do_stuck_cb = self._on_stuck is not None
                else:
                    self._stuck_counter = 0

            if do_dead_cb and self._on_dead:
                self._on_dead()
            if do_timeout_cb:
                self._on_timeout(*timeout_args)
                time.sleep(self.reconnect_delay_sec)
            if do_stuck_cb:
                self._on_stuck()
