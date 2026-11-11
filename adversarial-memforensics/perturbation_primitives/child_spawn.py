"""
Primitive 1: Benign Child-Process Spawning
==========================================
Forks short-lived, legitimate system binaries (signed OS utilities)
that increase `pslist.nproc` and `pslist.avg_threads` without touching
the malware's code path.

Risk Level: LOW — purely additive, does not interact with payload.

Mechanism:
    Spawn benign system executables (hostname.exe, whoami.exe, etc.)
    whose presence in a memory dump inflates process-count and thread-
    count features toward benign-profile ranges, reducing the ensemble's
    malicious-class confidence.

Target VolMemLyzer Features:
    - pslist.nproc          (total number of processes)
    - pslist.avg_threads    (average threads per process)
    - pslist.nppid          (parent PID count, secondary effect)
    - pslist.avg_handlers   (average handlers per process, secondary)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from perturbation_primitives import (
    BasePrimitive,
    FeatureDelta,
    PrimitiveResult,
    RiskLevel,
)


# System binaries that are:
#   1. Signed by Microsoft (won't trigger AV)
#   2. Present on all Windows installations
#   3. Lightweight, exit quickly or can be held alive
#   4. Do NOT require elevation
DEFAULT_BENIGN_BINARIES = [
    "hostname.exe",
    "whoami.exe",
    "where.exe",
    "find.exe",
    "sort.exe",
    "fc.exe",
    "comp.exe",
    "tree.exe",
    "more.com",
    "choice.exe",
]

# Binaries that can be held alive (block on stdin or run longer)
HOLDABLE_BINARIES = [
    ("cmd.exe", ["/k", "echo idle"]),       # stays alive until killed
    ("ping.exe", ["-t", "127.0.0.1"]),      # continuous ping
    ("timeout.exe", ["/t", "300", "/nobreak"]),
]


class BenignChildSpawner(BasePrimitive):
    """
    Perturbation primitive that spawns benign child processes to inflate
    process-listing features in the VolMemLyzer output.

    The spawned processes are legitimate OS utilities that do not touch
    the malware's code path. They are either:
      - Short-lived: execute and exit quickly (inflating nproc at snapshot time
        only if spawning is continuous or timed to acquisition)
      - Long-lived: held alive to guarantee presence during memory capture

    Parameters (apply kwargs):
        n_shortlived:   Number of short-lived binary invocations (default: 15)
        n_longlived:    Number of long-lived processes to hold (default: 5)
        hold_time:      Seconds to wait between short-lived spawns (default: 0.1)
        continuous:     If True, keep respawning short-lived procs in a loop
                        until cleanup() is called (default: False)
    """

    @property
    def name(self) -> str:
        return "benign_child_spawn"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def target_features(self) -> List[str]:
        return [
            "pslist.nproc",
            "pslist.avg_threads",
            "pslist.nppid",
            "pslist.avg_handlers",
        ]

    @property
    def scope(self) -> str:
        return "core"

    def predict_deltas(self, **params) -> List[FeatureDelta]:
        """
        Predict feature shifts based on spawn parameters.

        Reasoning:
          - Each long-lived process adds +1 to pslist.nproc
          - Short-lived processes add transiently (captured only if timing aligns)
          - avg_threads is diluted toward 1.0 (system utils are single-threaded)
          - nppid increases as spawned procs create new parent relationships
        """
        n_shortlived = params.get("n_shortlived", 15)
        n_longlived = params.get("n_longlived", 5)

        # Conservative: only long-lived processes are guaranteed visible
        guaranteed_new_procs = n_longlived
        # Likely visible: some short-lived ones captured mid-execution
        likely_transient_procs = int(n_shortlived * 0.3)

        total_estimated_new = guaranteed_new_procs + likely_transient_procs

        return [
            FeatureDelta(
                feature_name="pslist.nproc",
                direction=+1,
                magnitude=float(total_estimated_new),
                confidence=0.85,
            ),
            FeatureDelta(
                feature_name="pslist.avg_threads",
                direction=-1,  # diluted toward 1.0 (single-threaded utils)
                magnitude=0.5,  # approximate dilution effect
                confidence=0.60,
            ),
            FeatureDelta(
                feature_name="pslist.nppid",
                direction=+1,
                magnitude=float(min(total_estimated_new, 3)),  # bounded by unique parents
                confidence=0.70,
            ),
            FeatureDelta(
                feature_name="pslist.avg_handlers",
                direction=-1,  # diluted (system utils have fewer handlers)
                magnitude=2.0,
                confidence=0.50,
            ),
        ]

    def apply(self, **params) -> PrimitiveResult:
        """
        Apply the benign child-process spawning perturbation.

        Steps:
          1. Spawn n_longlived persistent processes (cmd.exe /k, ping -t, etc.)
          2. Spawn n_shortlived transient processes in rapid succession
          3. Record PIDs and timing for later verification

        Returns:
            PrimitiveResult with held_handles containing long-lived Popen objects
        """
        n_shortlived = params.get("n_shortlived", 15)
        n_longlived = params.get("n_longlived", 5)
        hold_time = params.get("hold_time", 0.1)

        result = PrimitiveResult(
            primitive_name=self.name,
            params=params,
            timestamp=time.time(),
        )

        longlived_procs: List[subprocess.Popen] = []
        shortlived_pids: List[int] = []

        try:
            # ── Phase 1: Spawn long-lived processes ──
            logger.info(f"[{self.name}] Spawning {n_longlived} long-lived processes...")
            for i in range(n_longlived):
                binary, args = HOLDABLE_BINARIES[i % len(HOLDABLE_BINARIES)]
                try:
                    proc = subprocess.Popen(
                        [binary] + args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32"
                            else 0
                        ),
                    )
                    longlived_procs.append(proc)
                    self._held_resources.append(proc)
                    logger.debug(f"  Long-lived PID {proc.pid}: {binary} {args}")
                except FileNotFoundError:
                    logger.warning(f"  Binary not found: {binary}, skipping")
                except Exception as e:
                    logger.warning(f"  Failed to spawn {binary}: {e}")

            # ── Phase 2: Spawn short-lived processes ──
            logger.info(f"[{self.name}] Spawning {n_shortlived} short-lived processes...")
            binaries = DEFAULT_BENIGN_BINARIES
            for i in range(n_shortlived):
                binary = binaries[i % len(binaries)]
                try:
                    proc = subprocess.Popen(
                        [binary],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32"
                            else 0
                        ),
                    )
                    shortlived_pids.append(proc.pid)
                    # Don't hold — let them exit naturally
                    time.sleep(hold_time)
                except FileNotFoundError:
                    pass  # Binary may not exist on all systems
                except Exception as e:
                    logger.debug(f"  Short-lived spawn failed ({binary}): {e}")

            elapsed = (time.time() - result.timestamp) * 1000  # ms
            result.elapsed_ms = elapsed
            result.handles_held = longlived_procs
            result.predicted_deltas = self.predict_deltas(**params)
            result.success = True

            logger.info(
                f"[{self.name}] Complete: {len(longlived_procs)} long-lived, "
                f"{len(shortlived_pids)} short-lived spawned in {elapsed:.1f}ms"
            )

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"[{self.name}] Failed: {e}")

        return result

    def cleanup(self) -> None:
        """Terminate all held long-lived processes."""
        logger.info(f"[{self.name}] Cleaning up {len(self._held_resources)} held processes")
        for proc in self._held_resources:
            try:
                if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception as e:
                logger.warning(f"  Cleanup failed for PID {getattr(proc, 'pid', '?')}: {e}")
                try:
                    proc.kill()
                except Exception:
                    pass
        self._held_resources.clear()
