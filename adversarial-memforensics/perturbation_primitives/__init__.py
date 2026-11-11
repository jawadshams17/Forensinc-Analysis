"""
Perturbation Primitives Package
================================
Implements functionality-preserving adversarial perturbations for memory
feature-space evasion, adapted from Demetrio et al.'s GAMMA framework
to the VolMemLyzer feature domain.

Core Primitives (committed scope):
  1. BenignChildSpawner   — fork short-lived legitimate OS binaries
  2. DLLLoadPadder        — load extra legitimate system DLLs
  3. HandleTablePadder    — open and hold benign file/registry handles

Extended Primitive (contingent on Phase 0):
  4. MalfindDiluter       — allocate inert executable memory regions

All primitives implement the BasePrimitive interface and are composable
via the PerturbationSequence class.
"""

from __future__ import annotations

import abc
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ── Feature Domain ──────────────────────────────────────────────────

class FeatureGroup(enum.Enum):
    """VolMemLyzer feature groups targeted by perturbation primitives."""
    PSLIST = "pslist"
    DLLLIST = "dlllist"
    HANDLES = "handles"
    LDRMODULES = "ldrmodules"
    MALFIND = "malfind"
    MODULES = "modules"
    SVCSCAN = "svcscan"
    CALLBACKS = "callbacks"
    NETSCAN = "netscan"


@dataclass(frozen=True)
class FeatureDelta:
    """
    Expected change to a single VolMemLyzer feature after applying a
    perturbation primitive.

    Attributes:
        feature_name: Fully qualified feature (e.g. 'pslist.nproc')
        direction:    Expected change direction (+1 increase, -1 decrease, 0 none)
        magnitude:    Estimated magnitude of change (absolute)
        confidence:   How confident we are in this prediction (0.0–1.0)
    """
    feature_name: str
    direction: int          # +1, -1, or 0
    magnitude: float        # estimated absolute shift
    confidence: float       # 0.0 to 1.0


@dataclass
class PrimitiveResult:
    """
    Result of applying a single perturbation primitive.

    Attributes:
        primitive_id:      Unique identifier for this application
        primitive_name:    Name of the primitive class
        params:            Parameters used for this application
        timestamp:         When the primitive was applied
        elapsed_ms:        Wall-clock milliseconds to apply
        handles_held:      Any resources that must remain alive during capture
        predicted_deltas:  Expected feature changes
        measured_deltas:   Actual feature changes (populated after extraction)
        success:           Whether the primitive executed without error
        error:             Error message if failed
    """
    primitive_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    primitive_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    elapsed_ms: float = 0.0
    handles_held: List[Any] = field(default_factory=list)
    predicted_deltas: List[FeatureDelta] = field(default_factory=list)
    measured_deltas: Optional[List[FeatureDelta]] = None
    success: bool = False
    error: Optional[str] = None


class RiskLevel(enum.Enum):
    """Risk classification for perturbation primitives."""
    LOW = "low"
    LOW_MEDIUM = "low-medium"
    MEDIUM = "medium"
    HIGH = "high"


# ── Base Primitive ──────────────────────────────────────────────────

class BasePrimitive(abc.ABC):
    """
    Abstract base class for all perturbation primitives.

    Each primitive must:
      1. Define its target features and risk level
      2. Predict the feature deltas it will cause
      3. Apply the perturbation and return a result
      4. Support cleanup of any held resources
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._held_resources: List[Any] = []

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of the primitive."""
        ...

    @property
    @abc.abstractmethod
    def risk_level(self) -> RiskLevel:
        """Risk level classification."""
        ...

    @property
    @abc.abstractmethod
    def target_features(self) -> List[str]:
        """List of VolMemLyzer feature names this primitive targets."""
        ...

    @property
    @abc.abstractmethod
    def scope(self) -> str:
        """'core' or 'extended' scope."""
        ...

    @abc.abstractmethod
    def predict_deltas(self, **params) -> List[FeatureDelta]:
        """
        Predict the feature deltas this primitive will cause with the
        given parameters, WITHOUT actually applying the perturbation.
        """
        ...

    @abc.abstractmethod
    def apply(self, **params) -> PrimitiveResult:
        """
        Apply the perturbation and return a result.
        Resources that must stay alive during memory capture should be
        stored in result.handles_held.
        """
        ...

    def cleanup(self) -> None:
        """Release any held resources (handles, processes, etc.)."""
        for resource in self._held_resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'terminate'):
                    resource.terminate()
                elif hasattr(resource, 'kill'):
                    resource.kill()
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to cleanup resource: {e}")
        self._held_resources.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(risk={self.risk_level.value}, targets={self.target_features})>"


# ── Perturbation Sequence ──────────────────────────────────────────

@dataclass
class PerturbationAction:
    """A single action in a perturbation sequence."""
    primitive: BasePrimitive
    params: Dict[str, Any]


@dataclass
class PerturbationSequence:
    """
    An ordered sequence of perturbation primitive applications.
    Represents the δ* output of the GAMMA-adapted optimizer.
    """
    actions: List[PerturbationAction] = field(default_factory=list)
    total_cost: float = 0.0
    fitness: float = 0.0
    creation_timestamp: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return len(self.actions)

    def predict_combined_deltas(self) -> Dict[str, FeatureDelta]:
        """
        Predict the combined feature deltas from all actions.
        Assumes additive composition for non-conflicting features.
        """
        combined: Dict[str, FeatureDelta] = {}
        for action in self.actions:
            deltas = action.primitive.predict_deltas(**action.params)
            for delta in deltas:
                if delta.feature_name in combined:
                    existing = combined[delta.feature_name]
                    combined[delta.feature_name] = FeatureDelta(
                        feature_name=delta.feature_name,
                        direction=np.sign(existing.magnitude * existing.direction
                                          + delta.magnitude * delta.direction),
                        magnitude=abs(existing.magnitude + delta.magnitude),
                        confidence=min(existing.confidence, delta.confidence),
                    )
                else:
                    combined[delta.feature_name] = delta
        return combined

    def apply_all(self) -> List[PrimitiveResult]:
        """Apply all actions in sequence, returning results."""
        results: List[PrimitiveResult] = []
        for i, action in enumerate(self.actions):
            logger.info(
                f"Applying perturbation {i+1}/{self.length}: "
                f"{action.primitive.name} with {action.params}"
            )
            result = action.primitive.apply(**action.params)
            results.append(result)
            if not result.success:
                logger.error(
                    f"Perturbation {action.primitive.name} failed: {result.error}"
                )
                break
        return results

    def cleanup_all(self) -> None:
        """Clean up all held resources from all primitives."""
        for action in self.actions:
            action.primitive.cleanup()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the sequence for logging/storage."""
        return {
            "length": self.length,
            "total_cost": self.total_cost,
            "fitness": self.fitness,
            "timestamp": self.creation_timestamp,
            "actions": [
                {
                    "primitive": action.primitive.name,
                    "params": action.params,
                    "risk_level": action.primitive.risk_level.value,
                    "target_features": action.primitive.target_features,
                }
                for action in self.actions
            ],
        }


# ── Feature Vector Utilities ───────────────────────────────────────

# Complete list of the 55 VolMemLyzer features from CIC-MalMem-2022
VOLMEMLYZER_FEATURES: List[str] = [
    # pslist (process listing)
    "pslist.nproc", "pslist.nppid", "pslist.avg_threads",
    "pslist.nprocs64bit", "pslist.avg_handlers",
    # dlllist (loaded DLLs)
    "dlllist.ndlls", "dlllist.avg_dlls_per_proc",
    "dlllist.avg_path_len",
    # handles
    "handles.nhandles", "handles.avg_handles_per_proc",
    "handles.nport", "handles.nfile", "handles.nevent",
    "handles.ndesktop", "handles.nkey", "handles.nthread",
    "handles.ndirectory", "handles.nsemaphore",
    "handles.ntimer", "handles.nsection",
    "handles.nmutant",
    # ldrmodules
    "ldrmodules.not_in_load", "ldrmodules.not_in_init",
    "ldrmodules.not_in_mem", "ldrmodules.not_in_load_avg",
    "ldrmodules.not_in_init_avg", "ldrmodules.not_in_mem_avg",
    # malfind
    "malfind.ninjections", "malfind.commitCharge",
    "malfind.protection", "malfind.uniqueInjections",
    # modules
    "modules.nmodules",
    # svcscan (service scanning)
    "svcscan.nservices", "svcscan.kernel_drivers",
    "svcscan.fs_drivers", "svcscan.process_services",
    "svcscan.shared_process_services",
    "svcscan.interactive_process_services",
    "svcscan.nactive", "svcscan.nstopped",
    # callbacks
    "callbacks.ncallbacks", "callbacks.nanonymous",
    "callbacks.ngeneric",
    # Additional statistical features
    "pslist.avg_priority", "pslist.avg_start_offset",
    "dlllist.nfwd_imports", "dlllist.avg_fwd_imports",
    "handles.avg_file_handles", "handles.avg_event_handles",
    "handles.avg_key_handles",
    # netscan (network connections)
    "netscan.nconns_established", "netscan.nconns_listening",
    "netscan.nconns_closed", "netscan.avg_local_port",
    "netscan.avg_remote_port",
    "netscan.nconns_total",
]

# Mapping from feature group prefix to feature names
FEATURE_GROUPS: Dict[str, List[str]] = {}
for feat in VOLMEMLYZER_FEATURES:
    group = feat.split(".")[0]
    FEATURE_GROUPS.setdefault(group, []).append(feat)


def apply_feature_deltas(
    feature_vector: np.ndarray,
    feature_names: List[str],
    deltas: Dict[str, FeatureDelta],
) -> np.ndarray:
    """
    Apply predicted feature deltas to a feature vector.
    This is the ⊕ operator: x' = x ⊕ δ

    Args:
        feature_vector: Original 55-dim feature vector
        feature_names:  Ordered list of feature names
        deltas:         Dict of feature name → FeatureDelta

    Returns:
        Modified feature vector with deltas applied
    """
    x_prime = feature_vector.copy().astype(np.float64)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    for feat_name, delta in deltas.items():
        if feat_name in name_to_idx:
            idx = name_to_idx[feat_name]
            if not np.isnan(x_prime[idx]):
                shift = delta.direction * delta.magnitude
                x_prime[idx] = max(0.0, x_prime[idx] + shift)  # features are non-negative
                logger.debug(
                    f"  {feat_name}: {feature_vector[idx]:.4f} → {x_prime[idx]:.4f} "
                    f"(δ={shift:+.4f})"
                )
        else:
            logger.warning(f"Feature '{feat_name}' not found in feature vector")

    return x_prime


# ── Exports ────────────────────────────────────────────────────────

__all__ = [
    "BasePrimitive",
    "FeatureGroup",
    "FeatureDelta",
    "PrimitiveResult",
    "RiskLevel",
    "PerturbationAction",
    "PerturbationSequence",
    "VOLMEMLYZER_FEATURES",
    "FEATURE_GROUPS",
    "apply_feature_deltas",
]
