"""
Interruption Logger — Cloud Preemption/Spot Notice Monitor
============================================================
Monitors cloud provider metadata endpoints to detect and log
interruption/preemption notices with millisecond precision.

Supports:
    - AWS Spot: IMDSv2 instance-action metadata
    - GCP Preemptible: maintenance-event + preempted metadata
    - Local: simulated interruption for testing
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

try:
    import requests
except ImportError:
    requests = None


@dataclass
class InterruptionEvent:
    """Recorded interruption/preemption event."""
    provider: str
    event_type: str         # 'spot_interruption', 'preemption', 'maintenance', 'simulated'
    detected_at: float      # epoch timestamp
    notice_content: str     # raw metadata response
    estimated_termination_at: float = 0.0
    notice_window_seconds: float = 0.0
    actual_termination_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class InterruptionLogger:
    """
    Monitors cloud metadata endpoints for interruption notices.
    Runs in a background thread with configurable polling interval.
    """

    # Metadata endpoints
    AWS_SPOT_ACTION = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    AWS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
    GCP_PREEMPTED = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
    GCP_MAINTENANCE = "http://metadata.google.internal/computeMetadata/v1/instance/maintenance-event"

    def __init__(
        self,
        provider: str = "aws",
        polling_interval: float = 5.0,
        output_dir: str = "data/timing_logs",
    ):
        self.provider = provider
        self.polling_interval = polling_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events: list[InterruptionEvent] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._imds_token: Optional[str] = None

    def start(self) -> None:
        """Start polling in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Interruption logger started: provider={self.provider}, "
            f"interval={self.polling_interval}s"
        )

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"Interruption logger stopped. Events logged: {len(self.events)}")

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                if self.provider == "aws":
                    self._check_aws_spot()
                elif self.provider == "gcp":
                    self._check_gcp_preemption()
                else:
                    pass  # local mode — no polling
            except Exception as e:
                logger.debug(f"Poll error (expected if not on cloud): {e}")

            time.sleep(self.polling_interval)

    def _check_aws_spot(self) -> None:
        """Check AWS Spot interruption notice via IMDSv2."""
        if requests is None:
            return

        # Get IMDSv2 token
        if not self._imds_token:
            try:
                resp = requests.put(
                    self.AWS_TOKEN_URL,
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
                    timeout=2,
                )
                if resp.status_code == 200:
                    self._imds_token = resp.text
            except Exception:
                return

        # Check for interruption notice
        try:
            resp = requests.get(
                self.AWS_SPOT_ACTION,
                headers={"X-aws-ec2-metadata-token": self._imds_token or ""},
                timeout=2,
            )
            if resp.status_code == 200:
                data = resp.json()
                event = InterruptionEvent(
                    provider="aws",
                    event_type="spot_interruption",
                    detected_at=time.time(),
                    notice_content=json.dumps(data),
                    estimated_termination_at=time.time() + 120,  # ~2 min
                    notice_window_seconds=120.0,
                )
                self.events.append(event)
                self._save_event(event)
                logger.warning(f"AWS SPOT INTERRUPTION DETECTED: {data}")
        except requests.exceptions.HTTPError:
            pass  # 404 = no interruption, expected

    def _check_gcp_preemption(self) -> None:
        """Check GCP preemption via metadata server."""
        if requests is None:
            return

        headers = {"Metadata-Flavor": "Google"}

        # Check preemption flag
        try:
            resp = requests.get(
                self.GCP_PREEMPTED,
                headers=headers,
                timeout=1,
            )
            if resp.status_code == 200 and resp.text.strip().lower() == "true":
                event = InterruptionEvent(
                    provider="gcp",
                    event_type="preemption",
                    detected_at=time.time(),
                    notice_content=resp.text,
                    estimated_termination_at=time.time() + 30,  # ~30 sec
                    notice_window_seconds=30.0,
                )
                self.events.append(event)
                self._save_event(event)
                logger.warning("GCP PREEMPTION DETECTED")
        except Exception:
            pass

        # Check maintenance event
        try:
            resp = requests.get(
                self.GCP_MAINTENANCE,
                headers=headers,
                timeout=1,
            )
            if resp.status_code == 200 and resp.text.strip() != "NONE":
                event = InterruptionEvent(
                    provider="gcp",
                    event_type="maintenance",
                    detected_at=time.time(),
                    notice_content=resp.text,
                    notice_window_seconds=30.0,
                )
                self.events.append(event)
                self._save_event(event)
                logger.warning(f"GCP MAINTENANCE EVENT: {resp.text}")
        except Exception:
            pass

    def simulate_interruption(
        self,
        delay_seconds: float = 0.0,
        notice_window: float = 120.0,
    ) -> InterruptionEvent:
        """Simulate an interruption event for local testing."""
        event = InterruptionEvent(
            provider="local",
            event_type="simulated",
            detected_at=time.time() + delay_seconds,
            notice_content="simulated interruption for testing",
            estimated_termination_at=time.time() + delay_seconds + notice_window,
            notice_window_seconds=notice_window,
        )
        self.events.append(event)
        self._save_event(event)
        logger.info(f"Simulated interruption: window={notice_window}s")
        return event

    def _save_event(self, event: InterruptionEvent) -> None:
        """Save an event to disk."""
        event_file = self.output_dir / f"interruption_{int(event.detected_at)}.json"
        with open(event_file, "w") as f:
            json.dump(event.to_dict(), f, indent=2)

    def get_timing_summary(self) -> Dict[str, Any]:
        """Summarize all recorded interruption events."""
        if not self.events:
            return {"n_events": 0}

        windows = [e.notice_window_seconds for e in self.events]
        return {
            "n_events": len(self.events),
            "providers": list(set(e.provider for e in self.events)),
            "avg_notice_window_s": float(sum(windows) / len(windows)),
            "min_notice_window_s": float(min(windows)),
            "max_notice_window_s": float(max(windows)),
            "events": [e.to_dict() for e in self.events],
        }

    def save_summary(self) -> None:
        """Save full timing summary to disk."""
        summary = self.get_timing_summary()
        summary_file = self.output_dir / "timing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Timing summary saved to {summary_file}")
