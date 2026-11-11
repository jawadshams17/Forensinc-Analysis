"""
Primitive 4: Malfind Region Dilution (EXTENDED SCOPE)
=====================================================
Allocates additional executable-flagged but INERT memory regions
alongside the payload's injected region, diluting malfind indicators.

Risk Level: HIGH — most likely to interact with the malware's actual
                    injection mechanism. EXPERIMENTAL, NOT BASELINE.

⚠️  This primitive is DISABLED by default and only enabled after
    Phase 0 extended validation confirms it does not disrupt the
    malware's functionality.

Mechanism:
    VirtualAllocEx() with PAGE_EXECUTE_READWRITE protection, filled
    with benign NOP/RET patterns or legitimate code stubs, creating
    "noise" regions that dilute the malfind signal (ninjections,
    commitCharge) across more regions.

Target VolMemLyzer Features:
    - malfind.ninjections     (count of injected/suspicious regions)
    - malfind.commitCharge    (total committed memory in suspect regions)
    - malfind.uniqueInjections (unique injection patterns)
    - malfind.protection      (protection flags aggregate)
"""

from __future__ import annotations

import ctypes
import struct
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


# Memory protection constants (Windows)
PAGE_EXECUTE_READWRITE = 0x40
PAGE_READWRITE = 0x04
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000

# Benign fill patterns for allocated regions
FILL_PATTERNS = {
    "nop_sled": b"\x90" * 4096,                      # x86 NOP sled
    "ret_stub": b"\xC3" + b"\xCC" * 4095,            # RET + INT3 padding
    "zero_page": b"\x00" * 4096,                      # zeroed page
    "benign_loop": (                                   # infinite loop stub
        b"\xEB\xFE"                                    # JMP $-2
        + b"\xCC" * 4094                               # INT3 padding
    ),
}

# Default region size: 4KB (single page) — minimum footprint
DEFAULT_REGION_SIZE = 4096


class MalfindDiluter(BasePrimitive):
    """
    EXTENDED SCOPE perturbation primitive that allocates executable
    but inert memory regions to dilute malfind indicators.

    This is the highest-risk primitive because:
      1. Executable memory regions are the primary malfind signal
      2. Interaction with the malware's injection mechanism is possible
      3. Security products may flag the allocation itself

    The strategy is DILUTION, not hiding:
      - By creating many benign executable regions, the ratio of
        malicious-to-total injections decreases
      - The aggregate commitCharge increases, potentially pushing
        the feature value into benign ranges

    Parameters (apply kwargs):
        n_regions:     Number of executable regions to allocate (default: 5)
        region_size:   Size in bytes per region (default: 4096)
        fill_pattern:  Pattern key from FILL_PATTERNS (default: 'ret_stub')
        target_pid:    PID to allocate in (None = current process)
    """

    @property
    def name(self) -> str:
        return "malfind_dilution"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    @property
    def target_features(self) -> List[str]:
        return [
            "malfind.ninjections",
            "malfind.commitCharge",
            "malfind.uniqueInjections",
            "malfind.protection",
        ]

    @property
    def scope(self) -> str:
        return "extended"

    def predict_deltas(self, **params) -> List[FeatureDelta]:
        """
        Predict feature shifts from malfind dilution.

        Key insight: malfind.ninjections will INCREASE (not decrease),
        but the distribution shifts — more regions with benign content
        dilutes the classifier's overall signal.
        """
        n_regions = params.get("n_regions", 5)
        region_size = params.get("region_size", DEFAULT_REGION_SIZE)

        return [
            FeatureDelta(
                feature_name="malfind.ninjections",
                direction=+1,
                magnitude=float(n_regions),
                confidence=0.80,
            ),
            FeatureDelta(
                feature_name="malfind.commitCharge",
                direction=+1,
                magnitude=float(n_regions * region_size),
                confidence=0.85,
            ),
            FeatureDelta(
                feature_name="malfind.uniqueInjections",
                direction=+1,
                # Benign patterns are distinct from malware patterns
                magnitude=float(min(n_regions, 3)),
                confidence=0.60,
            ),
            FeatureDelta(
                feature_name="malfind.protection",
                direction=+1,
                # More RWX regions shift the aggregate
                magnitude=float(n_regions) * 0.5,
                confidence=0.50,
            ),
        ]

    def apply(self, **params) -> PrimitiveResult:
        """
        Allocate executable memory regions with benign fill patterns.

        On Windows, uses VirtualAlloc for current process or
        VirtualAllocEx for remote process injection.
        """
        n_regions = params.get("n_regions", 5)
        region_size = params.get("region_size", DEFAULT_REGION_SIZE)
        fill_pattern_key = params.get("fill_pattern", "ret_stub")
        target_pid = params.get("target_pid", None)

        result = PrimitiveResult(
            primitive_name=self.name,
            params=params,
            timestamp=time.time(),
        )

        allocated_regions: List[Any] = []

        try:
            fill_data = FILL_PATTERNS.get(fill_pattern_key, FILL_PATTERNS["ret_stub"])
            # Tile the pattern to fill the region
            fill_data = (fill_data * (region_size // len(fill_data) + 1))[:region_size]

            if sys.platform == "win32":
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

                if target_pid is None:
                    # Allocate in current process
                    logger.info(
                        f"[{self.name}] Allocating {n_regions} × {region_size}B "
                        f"executable regions in current process..."
                    )
                    for i in range(n_regions):
                        addr = kernel32.VirtualAlloc(
                            None,
                            region_size,
                            MEM_COMMIT | MEM_RESERVE,
                            PAGE_EXECUTE_READWRITE,
                        )
                        if addr:
                            # Write benign fill pattern
                            ctypes.memmove(addr, fill_data, region_size)
                            allocated_regions.append(("local", addr, region_size))
                            logger.debug(
                                f"  Region {i}: addr=0x{addr:X}, "
                                f"size={region_size}, pattern={fill_pattern_key}"
                            )
                        else:
                            error = ctypes.get_last_error()
                            logger.warning(f"  VirtualAlloc failed, error={error}")
                else:
                    # Allocate in remote process
                    logger.info(
                        f"[{self.name}] Allocating {n_regions} × {region_size}B "
                        f"executable regions in PID {target_pid}..."
                    )
                    PROCESS_ALL_ACCESS = 0x1F0FFF
                    proc_handle = kernel32.OpenProcess(
                        PROCESS_ALL_ACCESS, False, target_pid
                    )
                    if not proc_handle:
                        raise OSError(
                            f"OpenProcess({target_pid}) failed: "
                            f"{ctypes.get_last_error()}"
                        )

                    for i in range(n_regions):
                        addr = kernel32.VirtualAllocEx(
                            proc_handle,
                            None,
                            region_size,
                            MEM_COMMIT | MEM_RESERVE,
                            PAGE_EXECUTE_READWRITE,
                        )
                        if addr:
                            # Write fill pattern
                            written = ctypes.c_size_t(0)
                            kernel32.WriteProcessMemory(
                                proc_handle, addr, fill_data,
                                region_size, ctypes.byref(written)
                            )
                            allocated_regions.append(
                                ("remote", addr, region_size, proc_handle)
                            )
                            logger.debug(f"  Remote region {i}: addr=0x{addr:X}")
                        else:
                            logger.warning(f"  VirtualAllocEx failed: {ctypes.get_last_error()}")
            else:
                # Simulation mode
                logger.info(f"[{self.name}] SIMULATION MODE (non-Windows)")
                for i in range(n_regions):
                    allocated_regions.append(("sim", i, region_size))
                    logger.debug(f"  [SIM] Region {i}: {region_size}B")

            self._held_resources = allocated_regions

            elapsed = (time.time() - result.timestamp) * 1000
            result.elapsed_ms = elapsed
            result.handles_held = allocated_regions
            result.predicted_deltas = self.predict_deltas(**params)
            result.success = True

            total_allocated = len(allocated_regions) * region_size
            logger.info(
                f"[{self.name}] Complete: {len(allocated_regions)}/{n_regions} "
                f"regions allocated ({total_allocated / 1024:.1f} KB) in {elapsed:.1f}ms"
            )

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"[{self.name}] Failed: {e}")

        return result

    def cleanup(self) -> None:
        """Free allocated memory regions."""
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            for region_info in self._held_resources:
                try:
                    if region_info[0] == "local":
                        _, addr, size = region_info
                        kernel32.VirtualFree(addr, 0, MEM_RELEASE)
                        logger.debug(f"  Freed local region 0x{addr:X}")
                    elif region_info[0] == "remote":
                        _, addr, size, proc_handle = region_info
                        kernel32.VirtualFreeEx(proc_handle, addr, 0, MEM_RELEASE)
                        logger.debug(f"  Freed remote region 0x{addr:X}")
                except Exception as e:
                    logger.warning(f"  Region cleanup failed: {e}")
        else:
            logger.info(f"[{self.name}] SIMULATION cleanup")
        self._held_resources.clear()
