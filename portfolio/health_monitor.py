from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Any


class HealthMonitor:
    def __init__(
        self,
        poll_seconds: float,
        divergence_threshold_bps: float,
        fetch_metrics: Callable[[], Dict[str, Any]],
        on_emergency_close: Callable[[str], None],
    ):
        self.poll_seconds = poll_seconds
        self.divergence_threshold_bps = divergence_threshold_bps
        self.fetch_metrics = fetch_metrics
        self.on_emergency_close = on_emergency_close
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            m = self.fetch_metrics()
            div_bps = float(m.get("leg_divergence_bps", 0.0))
            if abs(div_bps) > self.divergence_threshold_bps:
                self.on_emergency_close(f"divergence {div_bps:.2f}bps > {self.divergence_threshold_bps:.2f}bps")
            time.sleep(self.poll_seconds)

