# Adversarial Robustness of Memory-Forensic Ensemble Classifiers Under Cloud Ephemerality

> **Research-grade experimental platform** for evaluating how ensemble-based malware classifiers degrade under (1) memory-capture truncation from cloud ephemerality and (2) functionality-preserving adversarial feature perturbations.

## Research Questions

| RQ | Question | Module |
|----|----------|--------|
| **RQ1** | How do accuracy/F1 degrade with truncation (absent adversarial manipulation)? | `analysis/rq1_truncation_curves.py` |
| **RQ2** | Does adversarial evasion increase *disproportionately* on truncated captures? | `analysis/rq2_evasion_interaction.py` |
| **RQ3** | How do SHAP feature-importance rankings drift under combined stress? | `analysis/rq3_shap_drift.py` |
| **RQ4** | At what capture completeness does correct detection first occur? | `analysis/rq4_time_to_detection.py` |
| **RQ5** | Do adversarial perturbations transfer across classifier architectures? | `analysis/rq5_cross_architecture.py` |

## Architecture

```
adversarial-memforensics/
├── perturbation_primitives/       # Adversarial perturbation engine
│   ├── __init__.py                # BasePrimitive ABC, 55 features, ⊕ operator
│   ├── child_spawn.py             # P1: Benign child-process spawning
│   ├── dll_padding.py             # P2: DLL load-list padding
│   ├── handle_padding.py          # P3: Handle-table padding
│   ├── malfind_dilution.py        # P4: Malfind region dilution (extended)
│   └── optimizer.py               # GAMMA-adapted black-box optimizer
│
├── pipeline/                      # ML pipeline
│   ├── ensemble_model.py          # RF+XGBoost+LightGBM soft-voting ensemble
│   ├── feature_extraction.py      # VolMemLyzer wrapper + truncation simulation
│   ├── shap_analysis.py           # SHAP explainability as robustness diagnostic
│   └── missing_value_handler.py   # NaN encoding for truncated features
│
├── analysis/                      # RQ analysis pipelines
│   ├── rq1_truncation_curves.py   # Degradation curves with bootstrap CIs
│   ├── rq2_evasion_interaction.py # Interaction effect Δ
│   ├── rq3_shap_drift.py          # Feature-importance rank correlation
│   ├── rq4_time_to_detection.py   # Streaming detection latency
│   ├── rq5_cross_architecture.py  # Transfer attack analysis
│   ├── ablation_runner.py         # Component contribution isolation
│   ├── statistical_tests.py       # Bootstrap, Friedman, effect sizes
│   └── visualization.py           # Publication-quality figures
│
├── validation/                    # Phase 0 feasibility gate
│   ├── phase0_feasibility.py      # 3-criteria go/no-go evaluator
│   └── functionality_checker.py   # Behavior preservation verification
│
├── infrastructure/
│   ├── scripts/
│   │   ├── truncation_acquirer.sh        # LiME acquisition with time control
│   │   ├── acquisition_orchestrator.py   # Trial runner (local/AWS/GCP)
│   │   └── interruption_logger.py        # Cloud preemption monitor
│   ├── docker/Dockerfile                 # Multi-stage container
│   └── terraform/
│       ├── aws/main.tf                   # Spot Fleet + S3 + billing
│       └── gcp/main.tf                   # Preemptible + GCS + budget
│
├── config/
│   ├── experiment_config.yaml     # Master experiment configuration
│   ├── aws_spot_config.yaml       # AWS Spot parameters
│   └── gcp_preemptible_config.yaml # GCP Preemptible parameters
│
├── samples/
│   └── sample_registry.json       # Malware sample metadata manifest
│
├── tests/
│   ├── test_primitives.py         # 26 unit tests for primitives + optimizer
│   └── test_pipeline.py           # 13 integration tests for pipeline
│
├── run.py                         # CLI entry point
├── setup.py                       # Package installation
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
python run.py test

# Phase 0 feasibility (dry-run with synthetic data)
python run.py phase0 --dry-run

# Train ensemble (with real CIC-MalMem-2022 data)
python run.py train --data data/features/CIC-MalMem-2022.csv

# Run specific RQ analysis
python run.py analyze --rq 1 --output-dir results/

# Full experiment dry-run
python run.py full-experiment
```

## Key Design Decisions

1. **Phase 0 Gate**: Mandatory feasibility pilot must pass 3 criteria before cloud funding
2. **NaN Encoding**: Truncated features are `NaN`, not imputed — exercises native missing-value handling
3. **Offline Optimization**: GAMMA optimizer runs offline against a surrogate; only δ* is deployed at runtime
4. **Acquisition-Order Truncation**: Features masked in VolMemLyzer traversal order (pslist first, netscan last)

## Threat Model

- **Attacker**: Offline black-box optimization (GAMMA-adapted)
- **Capability**: Pre-computed perturbation sequences deployed as fixed runtime actions
- **Constraint**: Functionality-preserving (malicious behavior must survive perturbation)
- **Budget**: ≤50 ensemble queries, ≤5 perturbation actions

## Statistical Framework

- **Primary**: 95% bootstrap confidence intervals for all metrics
- **Comparative**: Friedman test + Nemenyi post-hoc across truncation levels
- **Effect Sizes**: Cohen's d and Cliff's delta
- **SHAP Stability**: Spearman's ρ with permutation-based significance testing
- **Multiple Comparisons**: Bonferroni correction

## Dataset

**CIC-MalMem-2022** (~58,596 samples, 55 VolMemLyzer features)
- Categories: Benign, Ransomware, Spyware, Trojan
- 3 families per malicious category (9 families total)
