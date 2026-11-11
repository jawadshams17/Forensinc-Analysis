"""
Acquisition Orchestrator — End-to-End Trial Runner
====================================================
Automates the full acquisition pipeline for each experimental trial:
    1. Launch ephemeral VM (Spot/Preemptible)
    2. Deploy malware sample
    3. Wait for steady-state
    4. Apply perturbation (if adversarial condition)
    5. Trigger LiME at designated truncation level
    6. Upload dump + metadata
    7. Run feature extraction + inference
    8. Record results

Supports both cloud (AWS/GCP) and local (KVM/QEMU) backends.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TrialConfig:
    """Configuration for a single acquisition trial."""
    trial_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    provider: str = "local"              # 'aws', 'gcp', 'local'
    truncation_pct: float = 1.0
    condition: str = "clean"             # 'clean' or 'adversarial'
    malware_family: str = ""
    malware_category: str = ""
    malware_hash: str = ""
    perturbation_sequence: Optional[Dict] = None
    steady_state_wait_seconds: float = 60.0
    repetition: int = 1
    output_dir: str = "data"


@dataclass
class TrialResult:
    """Result of a single acquisition trial."""
    trial_id: str = ""
    timestamp: str = ""
    provider: str = ""
    truncation_pct: float = 1.0
    condition: str = ""
    malware_family: str = ""
    malware_category: str = ""

    # Timing
    vm_launch_time_ms: float = 0.0
    malware_deploy_time_ms: float = 0.0
    steady_state_wait_ms: float = 0.0
    perturbation_time_ms: float = 0.0
    acquisition_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Cloud-specific
    actual_notice_window_ms: float = 0.0
    instance_id: str = ""

    # Data
    dump_path: str = ""
    dump_size_bytes: int = 0
    features_path: str = ""
    feature_vector: Optional[List[float]] = None
    n_nan_features: int = 0

    # Prediction
    predicted_class: int = -1
    predicted_label: str = ""
    malicious_probability: float = 0.0
    ensemble_probabilities: Optional[List[float]] = None

    # Perturbation details
    perturbation_applied: Optional[List[str]] = None
    perturbation_results: Optional[List[Dict]] = None

    # Status
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class AcquisitionOrchestrator:
    """
    Orchestrates end-to-end acquisition trials across cloud/local backends.
    """

    CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "data",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[TrialResult] = []

    def generate_trial_matrix(
        self,
        truncation_levels: List[float] = [1.0, 0.75, 0.50, 0.25, 0.10],
        conditions: List[str] = ["clean", "adversarial"],
        families: Optional[Dict[str, List[str]]] = None,
        providers: List[str] = ["local"],
        repetitions: int = 5,
    ) -> List[TrialConfig]:
        """Generate the full experimental trial matrix."""

        if families is None:
            families = {
                "Ransomware": ["wannacry_variant", "cerber_loader", "lockbit_mem"],
                "Spyware": ["agent_tesla", "formbook", "snake_keylogger"],
                "Trojan": ["emotet_stage", "trickbot_mem", "qakbot_injector"],
            }

        trials = []
        for provider in providers:
            for category, fam_list in families.items():
                for family in fam_list:
                    for condition in conditions:
                        for pct in truncation_levels:
                            for rep in range(1, repetitions + 1):
                                trial = TrialConfig(
                                    provider=provider,
                                    truncation_pct=pct,
                                    condition=condition,
                                    malware_family=family,
                                    malware_category=category,
                                    repetition=rep,
                                    output_dir=str(self.output_dir),
                                )
                                trials.append(trial)

        logger.info(
            f"Generated {len(trials)} trials: "
            f"{len(providers)} providers × {sum(len(v) for v in families.values())} families × "
            f"{len(conditions)} conditions × {len(truncation_levels)} truncation levels × "
            f"{repetitions} reps"
        )
        return trials

    def run_trial(self, config: TrialConfig) -> TrialResult:
        """
        Execute a single acquisition trial.

        In production, this launches real VMs and acquires memory.
        This implementation provides the framework; cloud-specific
        backends are pluggable.
        """
        result = TrialResult(
            trial_id=config.trial_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            provider=config.provider,
            truncation_pct=config.truncation_pct,
            condition=config.condition,
            malware_family=config.malware_family,
            malware_category=config.malware_category,
        )

        total_start = time.time()

        try:
            # Step 1: Launch VM
            logger.info(f"[{config.trial_id}] Launching {config.provider} VM...")
            vm_start = time.time()
            instance_id = self._launch_vm(config)
            result.instance_id = instance_id
            result.vm_launch_time_ms = (time.time() - vm_start) * 1000

            # Step 2: Deploy malware
            logger.info(f"[{config.trial_id}] Deploying {config.malware_family}...")
            deploy_start = time.time()
            self._deploy_malware(config, instance_id)
            result.malware_deploy_time_ms = (time.time() - deploy_start) * 1000

            # Step 3: Wait for steady-state
            logger.info(
                f"[{config.trial_id}] Waiting {config.steady_state_wait_seconds}s "
                f"for steady-state..."
            )
            time.sleep(min(config.steady_state_wait_seconds, 1.0))  # capped for testing
            result.steady_state_wait_ms = config.steady_state_wait_seconds * 1000

            # Step 4: Apply perturbation (adversarial only)
            if config.condition == "adversarial" and config.perturbation_sequence:
                logger.info(f"[{config.trial_id}] Applying perturbation sequence...")
                pert_start = time.time()
                pert_results = self._apply_perturbation(config, instance_id)
                result.perturbation_time_ms = (time.time() - pert_start) * 1000
                result.perturbation_applied = [
                    p.get("primitive", "unknown") for p in (pert_results or [])
                ]
                result.perturbation_results = pert_results

            # Step 5: Acquire memory at truncation level
            logger.info(
                f"[{config.trial_id}] Acquiring memory at {config.truncation_pct:.0%}..."
            )
            acq_start = time.time()
            dump_path = self._acquire_memory(config, instance_id)
            result.acquisition_time_ms = (time.time() - acq_start) * 1000
            result.dump_path = str(dump_path)

            if dump_path and Path(dump_path).exists():
                result.dump_size_bytes = Path(dump_path).stat().st_size

            # Step 6: Feature extraction + inference would happen here
            # (In production, this calls VolMemLyzer + ensemble)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"[{config.trial_id}] Trial failed: {e}")

        finally:
            # Step 7: Cleanup VM
            try:
                self._terminate_vm(config, result.instance_id)
            except Exception:
                pass

        result.total_time_ms = (time.time() - total_start) * 1000
        self.results.append(result)

        # Save individual result
        result_path = self.output_dir / "metadata" / f"{config.trial_id}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        return result

    def run_batch(
        self,
        trials: List[TrialConfig],
        max_concurrent: int = 1,
    ) -> List[TrialResult]:
        """Run a batch of trials sequentially (concurrent support is future work)."""
        results = []
        for i, trial in enumerate(trials):
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {i+1}/{len(trials)}: {trial.trial_id}")
            logger.info(f"  {trial.malware_family} | {trial.condition} | "
                        f"{trial.truncation_pct:.0%} | rep={trial.repetition}")
            logger.info(f"{'='*60}")

            result = self.run_trial(trial)
            results.append(result)

        self._save_batch_summary(results)
        return results

    # ── Backend Methods (pluggable) ────────────────────────────────

    def _launch_vm(self, config: TrialConfig) -> str:
        """Launch a VM. Returns instance ID. Override for real backends."""
        if config.provider == "local":
            return f"local-vm-{config.trial_id}"
        elif config.provider == "aws":
            # TODO: Implement AWS Spot launch via boto3
            logger.info("[AWS] Would launch Spot Instance here")
            return f"i-{config.trial_id}"
        elif config.provider == "gcp":
            # TODO: Implement GCP preemptible launch
            logger.info("[GCP] Would launch preemptible VM here")
            return f"gcp-{config.trial_id}"
        return f"unknown-{config.trial_id}"

    def _deploy_malware(self, config: TrialConfig, instance_id: str) -> None:
        """Deploy malware to the VM. Override for real backends."""
        logger.debug(f"[{config.provider}] Deploying {config.malware_family} to {instance_id}")

    def _apply_perturbation(self, config: TrialConfig, instance_id: str) -> List[Dict]:
        """Apply perturbation sequence. Override for real backends."""
        logger.debug(f"[{config.provider}] Applying perturbation to {instance_id}")
        return config.perturbation_sequence.get("actions", []) if config.perturbation_sequence else []

    def _acquire_memory(self, config: TrialConfig, instance_id: str) -> str:
        """Acquire memory dump. Override for real backends."""
        dump_dir = Path(config.output_dir) / "raw_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"{config.trial_id}.lime"
        # In production: execute truncation_acquirer.sh remotely
        dump_path.touch()  # placeholder
        return str(dump_path)

    def _terminate_vm(self, config: TrialConfig, instance_id: str) -> None:
        """Terminate the VM. Override for real backends."""
        logger.debug(f"[{config.provider}] Terminating {instance_id}")

    # ── Reporting ──────────────────────────────────────────────────

    def _save_batch_summary(self, results: List[TrialResult]) -> None:
        """Save batch results as a CSV summary."""
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        summary_path = self.output_dir / "batch_summary.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"Batch summary saved to {summary_path}")

        n_success = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {n_success}/{len(results)} successful")
