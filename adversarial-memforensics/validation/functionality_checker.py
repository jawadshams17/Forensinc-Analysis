"""
Functionality Checker — Dynamic Analysis Verification
======================================================
Provides tools to verify that adversarial perturbation primitives
preserve the malware's original malicious behavior.

This module defines the verification framework; actual dynamic analysis
must be performed in an isolated sandbox (VM snapshot → infect → perturb
→ observe → compare against baseline).

Verified behaviors per malware category:
    - Ransomware: file encryption activity, ransom note creation
    - Spyware:    keylogging, screen capture, data exfiltration
    - Trojan:     C2 contact, persistence mechanism, credential theft
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class MalwareCategory(enum.Enum):
    RANSOMWARE = "Ransomware"
    SPYWARE = "Spyware"
    TROJAN = "Trojan"
    BENIGN = "Benign"


class BehaviorType(enum.Enum):
    """Observable malicious behaviors for verification."""
    # Ransomware behaviors
    FILE_ENCRYPTION = "file_encryption"
    RANSOM_NOTE = "ransom_note_creation"
    FILE_DELETION = "shadow_copy_deletion"

    # Spyware behaviors
    KEYLOGGING = "keylogging"
    SCREEN_CAPTURE = "screen_capture"
    DATA_EXFILTRATION = "data_exfiltration"
    CLIPBOARD_MONITORING = "clipboard_monitoring"

    # Trojan behaviors
    C2_CONTACT = "c2_contact"
    PERSISTENCE = "persistence_mechanism"
    CREDENTIAL_THEFT = "credential_theft"
    LATERAL_MOVEMENT = "lateral_movement"

    # Generic behaviors
    NETWORK_ACTIVITY = "network_activity"
    PROCESS_INJECTION = "process_injection"
    REGISTRY_MODIFICATION = "registry_modification"


# Default expected behaviors per category
EXPECTED_BEHAVIORS: Dict[MalwareCategory, List[BehaviorType]] = {
    MalwareCategory.RANSOMWARE: [
        BehaviorType.FILE_ENCRYPTION,
        BehaviorType.C2_CONTACT,
        BehaviorType.PERSISTENCE,
    ],
    MalwareCategory.SPYWARE: [
        BehaviorType.DATA_EXFILTRATION,
        BehaviorType.C2_CONTACT,
        BehaviorType.NETWORK_ACTIVITY,
    ],
    MalwareCategory.TROJAN: [
        BehaviorType.C2_CONTACT,
        BehaviorType.PERSISTENCE,
        BehaviorType.PROCESS_INJECTION,
    ],
}


@dataclass
class BehaviorObservation:
    """Single observed behavior during dynamic analysis."""
    behavior_type: BehaviorType
    observed: bool
    timestamp: float = 0.0
    evidence: str = ""       # description of evidence (pcap ref, file path, etc.)
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class FunctionalityCheckResult:
    """
    Complete result of functionality verification for one
    (primitive, malware_sample) combination.
    """
    sample_id: str
    malware_family: str
    malware_category: MalwareCategory
    primitive_name: str
    primitive_params: Dict[str, Any]

    # Baseline (pre-perturbation)
    baseline_observations: List[BehaviorObservation] = field(default_factory=list)

    # Post-perturbation
    post_perturbation_observations: List[BehaviorObservation] = field(default_factory=list)

    # Analysis
    preserved_behaviors: List[BehaviorType] = field(default_factory=list)
    lost_behaviors: List[BehaviorType] = field(default_factory=list)
    new_behaviors: List[BehaviorType] = field(default_factory=list)  # unexpected

    # Verdict
    functionality_preserved: bool = False
    preservation_ratio: float = 0.0
    notes: str = ""
    analyst_name: str = ""
    analysis_timestamp: float = field(default_factory=time.time)

    def evaluate(self) -> None:
        """Compare baseline vs post-perturbation and set verdict."""
        baseline_set = {
            obs.behavior_type for obs in self.baseline_observations if obs.observed
        }
        post_set = {
            obs.behavior_type for obs in self.post_perturbation_observations if obs.observed
        }

        self.preserved_behaviors = list(baseline_set & post_set)
        self.lost_behaviors = list(baseline_set - post_set)
        self.new_behaviors = list(post_set - baseline_set)

        if baseline_set:
            self.preservation_ratio = len(self.preserved_behaviors) / len(baseline_set)
        else:
            self.preservation_ratio = 1.0

        # Functionality is preserved if ALL baseline behaviors are still present
        self.functionality_preserved = len(self.lost_behaviors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "malware_family": self.malware_family,
            "malware_category": self.malware_category.value,
            "primitive_name": self.primitive_name,
            "primitive_params": self.primitive_params,
            "preserved": [b.value for b in self.preserved_behaviors],
            "lost": [b.value for b in self.lost_behaviors],
            "new": [b.value for b in self.new_behaviors],
            "functionality_preserved": self.functionality_preserved,
            "preservation_ratio": self.preservation_ratio,
            "notes": self.notes,
        }


class FunctionalityChecker:
    """
    Coordinates functionality verification across multiple
    (primitive, sample) combinations.

    Workflow:
        1. Register malware samples with expected behaviors
        2. Record baseline observations (pre-perturbation)
        3. Apply perturbation
        4. Record post-perturbation observations
        5. Evaluate and compare
    """

    def __init__(self):
        self.samples: Dict[str, Dict[str, Any]] = {}
        self.results: List[FunctionalityCheckResult] = []

    def register_sample(
        self,
        sample_id: str,
        family: str,
        category: MalwareCategory,
        expected_behaviors: Optional[List[BehaviorType]] = None,
        hash_sha256: str = "",
    ) -> None:
        """Register a malware sample for functionality testing."""
        self.samples[sample_id] = {
            "family": family,
            "category": category,
            "expected_behaviors": expected_behaviors or EXPECTED_BEHAVIORS.get(category, []),
            "hash": hash_sha256,
        }
        logger.info(
            f"Registered sample {sample_id}: {family} ({category.value}), "
            f"{len(self.samples[sample_id]['expected_behaviors'])} expected behaviors"
        )

    def record_result(self, result: FunctionalityCheckResult) -> None:
        """Record a completed functionality check result."""
        result.evaluate()
        self.results.append(result)

        status = "✅ PRESERVED" if result.functionality_preserved else "❌ BROKEN"
        logger.info(
            f"[{result.primitive_name} × {result.sample_id}] {status} "
            f"({result.preservation_ratio:.0%})"
        )
        if result.lost_behaviors:
            logger.warning(f"  Lost behaviors: {[b.value for b in result.lost_behaviors]}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all results."""
        n_total = len(self.results)
        n_preserved = sum(1 for r in self.results if r.functionality_preserved)

        by_primitive = {}
        by_category = {}
        for r in self.results:
            by_primitive.setdefault(r.primitive_name, []).append(r.functionality_preserved)
            by_category.setdefault(r.malware_category.value, []).append(r.functionality_preserved)

        return {
            "total_tests": n_total,
            "total_preserved": n_preserved,
            "overall_ratio": n_preserved / n_total if n_total else 0.0,
            "by_primitive": {
                k: sum(v) / len(v) for k, v in by_primitive.items()
            },
            "by_category": {
                k: sum(v) / len(v) for k, v in by_category.items()
            },
        }

    def save_results(self, output_dir: Path) -> None:
        """Save all results to a JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "functionality_check_results.json"

        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Functionality check results saved to {output_path}")

    def generate_checklist(self, output_dir: Path) -> None:
        """
        Generate a manual testing checklist for the analyst.
        This is used when dynamic analysis must be performed manually.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        checklist_path = output_dir / "functionality_checklist.md"

        lines = [
            "# Phase 0 Functionality Verification Checklist",
            "",
            "For each combination below, verify in an isolated sandbox VM:",
            "",
        ]

        primitives = ["benign_child_spawn", "dll_load_padding", "handle_table_padding"]
        test_num = 0

        for sample_id, info in self.samples.items():
            for prim in primitives:
                test_num += 1
                behaviors = info["expected_behaviors"]
                lines.extend([
                    f"## Test {test_num}: {prim} × {sample_id}",
                    f"- Family: {info['family']}",
                    f"- Category: {info['category'].value}",
                    f"- SHA256: {info['hash'] or 'N/A'}",
                    "",
                    "### Baseline (No Perturbation)",
                    *[f"- [ ] {b.value}" for b in behaviors],
                    "",
                    "### Post-Perturbation",
                    *[f"- [ ] {b.value}" for b in behaviors],
                    "",
                    "### Verdict: [ ] Preserved / [ ] Broken",
                    "### Notes:",
                    "",
                    "---",
                    "",
                ])

        with open(checklist_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(
            f"Generated checklist: {test_num} tests across "
            f"{len(self.samples)} samples → {checklist_path}"
        )
