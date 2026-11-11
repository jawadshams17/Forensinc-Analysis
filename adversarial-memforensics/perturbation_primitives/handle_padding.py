"""
Primitive 3: Handle-Table Padding
==================================
Opens and holds extra benign handles (files, registry keys, events,
mutexes) with no functional use, inflating handles.nhandles and
handles.avg_handles_per_proc toward benign-profile distributions.

Risk Level: LOW — purely additive, does not interact with payload.

Mechanism:
    CreateFile() on temp files, RegOpenKeyEx() on common registry paths,
    CreateEvent() for named events — all benign, all additive-only.

Target VolMemLyzer Features:
    - handles.nhandles               (total handle count)
    - handles.avg_handles_per_proc   (average handles per process)
    - handles.nfile                  (file handles, secondary)
    - handles.nkey                   (registry key handles, secondary)
    - handles.nevent                 (event handles, secondary)
    - handles.nmutant                (mutex handles, secondary)
"""

from __future__ import annotations

import os
import sys
import tempfile
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

# Common, safe registry paths to open (read-only)
SAFE_REGISTRY_PATHS = [
    r"SOFTWARE\Microsoft\Windows\CurrentVersion",
    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
    r"SYSTEM\CurrentControlSet\Control",
    r"SOFTWARE\Classes\CLSID",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
    r"SYSTEM\CurrentControlSet\Services",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer",
]


class HandleTablePadder(BasePrimitive):
    """
    Perturbation primitive that opens and holds benign OS handles
    to inflate handle-count features.

    Handle types created:
      - File handles:     tempfile.NamedTemporaryFile (persistent open)
      - Registry handles: winreg.OpenKey (HKLM, read-only)
      - Event handles:    CreateEvent (named events)
      - Mutex handles:    CreateEvent acting as named mutexes

    Parameters (apply kwargs):
        n_file_handles:   Number of temp file handles to open (default: 20)
        n_reg_handles:    Number of registry key handles (default: 10)
        n_event_handles:  Number of named events (default: 10)
        n_mutex_handles:  Number of named mutexes (default: 5)
    """

    @property
    def name(self) -> str:
        return "handle_table_padding"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def target_features(self) -> List[str]:
        return [
            "handles.nhandles",
            "handles.avg_handles_per_proc",
            "handles.nfile",
            "handles.nkey",
            "handles.nevent",
            "handles.nmutant",
        ]

    @property
    def scope(self) -> str:
        return "core"

    def predict_deltas(self, **params) -> List[FeatureDelta]:
        """
        Predict feature shifts from handle padding.

        Each handle type contributes to its own counter AND total nhandles.
        """
        n_file = params.get("n_file_handles", 20)
        n_reg = params.get("n_reg_handles", 10)
        n_event = params.get("n_event_handles", 10)
        n_mutex = params.get("n_mutex_handles", 5)
        total = n_file + n_reg + n_event + n_mutex

        return [
            FeatureDelta(
                feature_name="handles.nhandles",
                direction=+1,
                magnitude=float(total),
                confidence=0.95,
            ),
            FeatureDelta(
                feature_name="handles.avg_handles_per_proc",
                direction=+1,
                # All handles in one process: effect ≈ total / nproc
                magnitude=float(total) / 50.0,
                confidence=0.70,
            ),
            FeatureDelta(
                feature_name="handles.nfile",
                direction=+1,
                magnitude=float(n_file),
                confidence=0.90,
            ),
            FeatureDelta(
                feature_name="handles.nkey",
                direction=+1,
                magnitude=float(n_reg),
                confidence=0.90,
            ),
            FeatureDelta(
                feature_name="handles.nevent",
                direction=+1,
                magnitude=float(n_event),
                confidence=0.85,
            ),
            FeatureDelta(
                feature_name="handles.nmutant",
                direction=+1,
                magnitude=float(n_mutex),
                confidence=0.85,
            ),
        ]

    def apply(self, **params) -> PrimitiveResult:
        """
        Open and hold benign handles of multiple types.

        Returns:
            PrimitiveResult with all handles as held_handles
        """
        n_file = params.get("n_file_handles", 20)
        n_reg = params.get("n_reg_handles", 10)
        n_event = params.get("n_event_handles", 10)
        n_mutex = params.get("n_mutex_handles", 5)

        result = PrimitiveResult(
            primitive_name=self.name,
            params=params,
            timestamp=time.time(),
        )

        all_handles: List[Any] = []
        counts = {"file": 0, "reg": 0, "event": 0, "mutex": 0}

        try:
            # ── File Handles ──
            logger.info(f"[{self.name}] Opening {n_file} file handles...")
            for i in range(n_file):
                try:
                    f = tempfile.NamedTemporaryFile(
                        delete=False,
                        prefix=f"amf_pad_{i}_",
                        suffix=".tmp",
                    )
                    all_handles.append(("file", f))
                    counts["file"] += 1
                except Exception as e:
                    logger.debug(f"  File handle {i} failed: {e}")

            # ── Registry Handles (Windows only) ──
            if sys.platform == "win32":
                import winreg
                logger.info(f"[{self.name}] Opening {n_reg} registry handles...")
                for i in range(n_reg):
                    reg_path = SAFE_REGISTRY_PATHS[i % len(SAFE_REGISTRY_PATHS)]
                    try:
                        key = winreg.OpenKey(
                            winreg.HKEY_LOCAL_MACHINE,
                            reg_path,
                            0,
                            winreg.KEY_READ,
                        )
                        all_handles.append(("reg", key))
                        counts["reg"] += 1
                    except Exception as e:
                        logger.debug(f"  Registry handle ({reg_path}) failed: {e}")
            else:
                logger.info(f"[{self.name}] SIMULATION: {n_reg} registry handles (non-Windows)")
                counts["reg"] = n_reg

            # ── Event Handles (Windows only) ──
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                logger.info(f"[{self.name}] Creating {n_event} event handles...")
                for i in range(n_event):
                    try:
                        event_name = f"AMF_PadEvent_{i}_{int(time.time())}"
                        handle = kernel32.CreateEventW(
                            None,   # security attributes
                            True,   # manual reset
                            False,  # initial state
                            event_name,
                        )
                        if handle:
                            all_handles.append(("event", handle))
                            counts["event"] += 1
                    except Exception as e:
                        logger.debug(f"  Event handle {i} failed: {e}")

                # ── Mutex Handles ──
                logger.info(f"[{self.name}] Creating {n_mutex} mutex handles...")
                for i in range(n_mutex):
                    try:
                        mutex_name = f"AMF_PadMutex_{i}_{int(time.time())}"
                        handle = kernel32.CreateMutexW(
                            None,
                            False,
                            mutex_name,
                        )
                        if handle:
                            all_handles.append(("mutex", handle))
                            counts["mutex"] += 1
                    except Exception as e:
                        logger.debug(f"  Mutex handle {i} failed: {e}")
            else:
                logger.info(f"[{self.name}] SIMULATION: event/mutex handles (non-Windows)")
                counts["event"] = n_event
                counts["mutex"] = n_mutex

            self._held_resources = all_handles

            elapsed = (time.time() - result.timestamp) * 1000
            result.elapsed_ms = elapsed
            result.handles_held = all_handles
            result.predicted_deltas = self.predict_deltas(**params)
            result.success = True

            total_opened = sum(counts.values())
            logger.info(
                f"[{self.name}] Complete: {total_opened} handles opened in "
                f"{elapsed:.1f}ms — file:{counts['file']}, reg:{counts['reg']}, "
                f"event:{counts['event']}, mutex:{counts['mutex']}"
            )

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"[{self.name}] Failed: {e}")

        return result

    def cleanup(self) -> None:
        """Close all held handles and delete temp files."""
        logger.info(f"[{self.name}] Cleaning up {len(self._held_resources)} handles")

        for handle_type, handle in self._held_resources:
            try:
                if handle_type == "file":
                    name = getattr(handle, 'name', None)
                    handle.close()
                    if name and os.path.exists(name):
                        os.unlink(name)
                elif handle_type == "reg":
                    if sys.platform == "win32":
                        import winreg
                        winreg.CloseKey(handle)
                elif handle_type in ("event", "mutex"):
                    if sys.platform == "win32":
                        import ctypes
                        ctypes.WinDLL("kernel32").CloseHandle(handle)
            except Exception as e:
                logger.warning(f"  Cleanup ({handle_type}) failed: {e}")

        self._held_resources.clear()
