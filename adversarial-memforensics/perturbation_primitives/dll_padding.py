"""
Primitive 2: DLL Load-List Padding
===================================
Loads additional legitimate, already-resident system DLLs into the
malicious process (or a helper process in the same session) to inflate
dlllist features toward benign profiles.

Risk Level: LOW-MEDIUM — must not collide with payload's own imports.

Mechanism:
    Call LoadLibrary() for signed Windows system DLLs that are already
    memory-resident (cached by the system loader). This increases the
    DLL count per process without loading foreign or suspicious modules.

Target VolMemLyzer Features:
    - dlllist.ndlls              (total DLLs loaded across all processes)
    - dlllist.avg_dlls_per_proc  (average DLLs per process)
    - dlllist.avg_path_len       (average path length - secondary effect)
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from perturbation_primitives import (
    BasePrimitive,
    FeatureDelta,
    PrimitiveResult,
    RiskLevel,
)


# Curated list of SAFE system DLLs:
#   - Signed by Microsoft
#   - Present on all Windows 10+ installations
#   - Already loaded by the system (low overhead)
#   - Unlikely to conflict with malware imports
#   - Represent a diverse set of functionality (crypto, networking, UI, etc.)
SAFE_SYSTEM_DLLS = [
    # Common subsystem DLLs
    "advapi32.dll",       # Advanced Windows API
    "shell32.dll",        # Shell operations
    "crypt32.dll",        # Certificate/crypto
    "wintrust.dll",       # Trust verification
    "imagehlp.dll",       # Image helper
    "version.dll",        # Version information
    "userenv.dll",        # User environment
    "winhttp.dll",        # HTTP client
    "urlmon.dll",         # URL moniker
    "mswsock.dll",        # Winsock helper
    "dnsapi.dll",         # DNS client
    "iphlpapi.dll",       # IP helper
    "netapi32.dll",       # Network management
    "secur32.dll",        # Security support
    "wtsapi32.dll",       # Terminal services
    "pdh.dll",            # Performance data helper
    "dbghelp.dll",        # Debug helper
    "oleaut32.dll",       # OLE automation
    "shlwapi.dll",        # Shell lightweight utilities
    "cabinet.dll",        # Cabinet file extraction
]

# Average system DLL path length (C:\Windows\System32\xxx.dll ≈ 35 chars)
AVG_SYSTEM_DLL_PATH_LEN = 35.0


class DLLLoadPadder(BasePrimitive):
    """
    Perturbation primitive that loads additional legitimate system DLLs
    to inflate DLL-listing features.

    The loaded DLLs are curated to be:
      - Safe (signed, system-resident)
      - Unlikely to conflict with malware's own imports
      - Diverse in functionality (to appear as normal system activity)

    Parameters (apply kwargs):
        n_dlls:         Number of additional DLLs to load (default: 8)
        target_pid:     PID to load into (None = current process, default: None)
        select_random:  Randomly select from the pool (default: True)
    """

    @property
    def name(self) -> str:
        return "dll_load_padding"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW_MEDIUM

    @property
    def target_features(self) -> List[str]:
        return [
            "dlllist.ndlls",
            "dlllist.avg_dlls_per_proc",
            "dlllist.avg_path_len",
        ]

    @property
    def scope(self) -> str:
        return "core"

    def predict_deltas(self, **params) -> List[FeatureDelta]:
        """
        Predict feature shifts from DLL padding.

        Reasoning:
          - Each loaded DLL adds +1 to per-process DLL count
          - ndlls increases by n_dlls (system-wide total)
          - avg_dlls_per_proc is diluted if only affecting one process
          - avg_path_len shifts toward system DLL avg (~35 chars)
        """
        n_dlls = params.get("n_dlls", 8)

        return [
            FeatureDelta(
                feature_name="dlllist.ndlls",
                direction=+1,
                magnitude=float(n_dlls),
                confidence=0.90,
            ),
            FeatureDelta(
                feature_name="dlllist.avg_dlls_per_proc",
                direction=+1,
                # Effect is n_dlls / nproc — approximate with nproc ≈ 50
                magnitude=float(n_dlls) / 50.0,
                confidence=0.70,
            ),
            FeatureDelta(
                feature_name="dlllist.avg_path_len",
                direction=-1 if params.get("__current_avg_path_len", 40.0) > AVG_SYSTEM_DLL_PATH_LEN else +1,
                magnitude=abs(params.get("__current_avg_path_len", 40.0) - AVG_SYSTEM_DLL_PATH_LEN) * 0.1,
                confidence=0.50,
            ),
        ]

    def apply(self, **params) -> PrimitiveResult:
        """
        Load additional system DLLs via LoadLibrary.

        On Windows, uses ctypes to call kernel32.LoadLibraryW.
        On non-Windows (testing/simulation), simulates the operation.

        Returns:
            PrimitiveResult with dl handles as held resources
        """
        n_dlls = params.get("n_dlls", 8)
        select_random = params.get("select_random", True)

        result = PrimitiveResult(
            primitive_name=self.name,
            params=params,
            timestamp=time.time(),
        )

        loaded_handles: List[Any] = []
        loaded_names: List[str] = []

        try:
            # Select DLLs to load
            available = list(SAFE_SYSTEM_DLLS)
            if select_random:
                rng = np.random.default_rng()
                rng.shuffle(available)
            dlls_to_load = available[:n_dlls]

            if sys.platform == "win32":
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.LoadLibraryW.restype = ctypes.wintypes.HMODULE
                kernel32.LoadLibraryW.argtypes = [ctypes.wintypes.LPCWSTR]

                for dll_name in dlls_to_load:
                    try:
                        handle = kernel32.LoadLibraryW(dll_name)
                        if handle:
                            loaded_handles.append((dll_name, handle))
                            loaded_names.append(dll_name)
                            logger.debug(f"  Loaded {dll_name} → handle 0x{handle:X}")
                        else:
                            error_code = ctypes.get_last_error()
                            logger.warning(
                                f"  LoadLibrary({dll_name}) returned NULL, "
                                f"error={error_code}"
                            )
                    except Exception as e:
                        logger.warning(f"  Failed to load {dll_name}: {e}")
            else:
                # Simulation mode for non-Windows development/testing
                logger.info(f"[{self.name}] SIMULATION MODE (non-Windows)")
                for dll_name in dlls_to_load:
                    loaded_handles.append((dll_name, None))
                    loaded_names.append(dll_name)
                    logger.debug(f"  [SIM] Loaded {dll_name}")

            # Store handles for cleanup
            self._held_resources = loaded_handles

            elapsed = (time.time() - result.timestamp) * 1000
            result.elapsed_ms = elapsed
            result.handles_held = loaded_handles
            result.predicted_deltas = self.predict_deltas(**params)
            result.success = True

            logger.info(
                f"[{self.name}] Complete: {len(loaded_names)}/{n_dlls} DLLs "
                f"loaded in {elapsed:.1f}ms — {loaded_names}"
            )

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"[{self.name}] Failed: {e}")

        return result

    def cleanup(self) -> None:
        """Free loaded DLL handles via FreeLibrary."""
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            for dll_name, handle in self._held_resources:
                if handle:
                    try:
                        kernel32.FreeLibrary(handle)
                        logger.debug(f"  Freed {dll_name} (0x{handle:X})")
                    except Exception as e:
                        logger.warning(f"  FreeLibrary({dll_name}) failed: {e}")
        else:
            logger.info(f"[{self.name}] SIMULATION cleanup")
        self._held_resources.clear()
