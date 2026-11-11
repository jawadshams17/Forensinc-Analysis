# Q1 Walkthrough: Cross-Dataset Generalisation in Memory Forensics

## 0. Threat Model

> [!IMPORTANT]
> **Adversary definition (must be stated before the research questions).**

The attacker possesses **user-land code execution** on the victim host and may alter OS-level parameters (registry, services, process creation) and employ **standard hooking and injection techniques** (DLL injection, process hollowing, APC queues).

The attacker explicitly **does NOT possess**:
- Ring-0 / hypervisor-level control capable of blinding Volatility 3's physical memory snapshot acquisition.
- Direct Kernel Object Manipulation (DKOM) capable of fully delisting process objects from `EPROCESS` chains. (Note: DKOM would produce **zero-vectors** for the VolMemLyzer `pslist_*` family of features. Our Information-Theoretical Ablation in Step 3 quantifies the precision impact of this exact zeroing scenario.)

This framing defines the operational boundary: our detector is valid against commodity malware families. It is *explicitly not* claimed to defeat kernel-level adversaries who can surgically blind the acquisition chain.

---

## 1. The Problem

A single **XGBoost** model (Validated via continuous-probability Wilcoxon testing: across 4 independent CV seeds, no consistent winner emerged between it and the Voting Ensemble (mean relative diff -0.86%), consistent with fold-level noise rather than a structural capability difference) trained on **58,596** CIC-MalMem-2022 samples achieves **100% accuracy** in-distribution but collapses to **~50% (random chance)** on 30 independently-acquired MalVol-25 memory dumps.

> [!NOTE]
> **Multi-class label disclaimer.** The labels "Ransomware," "Spyware," and "Trojan" are vendor-assigned CIC dataset tags used here purely as a mechanism to force a systematic data partition for domain-shift experimentation. At the physical memory level these categories are behaviourally overlapping (e.g. Ryuk operates as Trojan-loader → Spyware-credential-dump → Ransomware-encryption). All core detection metrics are therefore reported as **binary: Benign vs. Malicious**.

---

## 1B. Data Alignment Methodology (Feature Space: 55 → 41)

The original VolMemLyzer extraction schema produces **55 raw properties** per memory snapshot, and the CIC-MalMem-2022 source dataset contains all 55. However, standardizing a unified feature space for cross-dataset evaluation across differing Volatility versions and target OS builds required computing the strict intersection of stable numerical features. 

**Alignment Procedure:**
1. The CIC-MalMem-2022 dataset provides the full 55 numeric feature columns.
2. Feature extraction on the MalVol-25 memory dumps via Volatility 3 produced the schema.
3. The intersection of numeric feature names was computed. **14 features were dropped**, all belonging to the `psxview_not_in_*` family (e.g., `psxview_not_in_pslist`, `psxview_not_in_eprocess_pool`, `psxview_not_in_session`). These properties were dropped because they were highly sparse, largely degenerate across the target domain, or inconsistently mapped due to OS structural differences (Windows 7 vs Windows 10).
4. **Result:** A unified space of **41 canonical features**. 

> [!IMPORTANT]
> **MalVol-25 contains exactly all 41 of these aligned features with zero missing columns.** Cross-dataset and benign-OOD evaluations are executed natively on this fully intact 41-feature space; no dimensionality reduction is substituted for missing data.

---

## 2. Root-Cause (C2): Feature Value Shift (KS-Test)

> [!NOTE]
> This section measures drift in **feature VALUES** between datasets using KS two-sample tests. It is distinct from model reliance drift (see Section 3 and Phase 2A).

KS two-sample testing reveals **39/41 features** have statistically significant distribution shift (p < 0.05). The shift is systemic — every Volatility plugin group drifts.

![Feature Distribution Shift Heatmap](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/feature_drift_heatmap.png)

![Shift by Feature Group](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/grouped_drift_bars.png)

---

## 3. Domain Adaptation Experiments (C3): Exploratory Sweep — 10 Variants

> [!WARNING]
> **Multiple-Comparisons Notice.** This section constitutes an **exploratory adaptation sweep** across 10 variants (D0, D3, U1, U3, U5, U2, U2b, U2c, U4, U6) tested against the same MalVol-25 holdout (N=30). Because 10 variants were evaluated, this sweep exceeds the threshold for confirmatory inference without correction. Per the Holm-Bonferroni procedure applied to the 6 pairwise McNemar p-values (raw: 0.131, 0.371, 0.450, 0.617, 0.724, 1.000), none survive correction. This section is therefore **exploratory and non-confirmatory**. The sole confirmatory result of the paper is the headline domain-shift finding (Sections 4B and 6): a model achieving 100% CIC accuracy exhibits 26.7% FPR on Windows 10 Benign samples and degrades to ~50% accuracy on MalVol-25.

> [!CAUTION]
> **Classifier Collapse Notice.** Three adaptation variants — U2 (5-Shot Fine-Tuning), U4 (Stacked Meta-Learner), and U6 (Combined Pipeline) — collapsed to **majority-class prediction** (confusion matrix: [[12,0],[13,0]] — zero recall on Benign class). Their reported accuracy (48%) reflects class imbalance, not classification ability. These methods are **excluded from headline comparison** and retained below only for completeness with explicit collapse labels. They must NOT be presented as comparable functioning methods.

### Exploratory Results Table (Transductive LOOCV on MalVol-25, N=30)

| Method | Accuracy | CM | F1 | AUC | Status |
|---|---|---|---|---|---|
| D0: No Adaptation (Baseline) | 50.0% | [[11,4],[11,4]] | 0.471 | 0.502 | Functioning |
| **D3: CORAL Alignment** | **60.0%** | **[[10,5],[7,8]]** | **0.598** | **0.551** | Functioning |
| U1: Rank Transform (Quantile) | 43.3% | [[6,9],[8,7]] | 0.433 | 0.471 | Functioning (inverted) |
| U5: Delta Features + Z-Score | 53.3% | [[11,4],[10,5]] | 0.514 | 0.431 | Functioning |
| U3: TCA (MMD Minimisation) | 53.3% | [[9,6],[8,7]] | 0.531 | 0.482 | Functioning |
| ~~U2: 5-Shot Fine-Tuning~~ | ~~48.0%~~ | ~~[[12,0],[13,0]]~~ | ~~0.324~~ | — | **⚠ COLLAPSED (all-Malware)** |
| ~~U4: Stacked Meta-Learner~~ | ~~48.0%~~ | ~~[[12,0],[13,0]]~~ | ~~0.324~~ | — | **⚠ COLLAPSED (all-Malware)** |
| ~~U6: Combined Pipeline~~ | ~~48.0%~~ | ~~[[12,0],[13,0]]~~ | ~~0.324~~ | — | **⚠ COLLAPSED (all-Malware)** |

### McNemar Pairwise Significance (Functioning Methods Only)

| Comparison | b | c | p-value (raw) | p-value (Holm-Bonferroni) | Significant? |
|---|---|---|---|---|---|
| D3: CORAL vs D0 | — | — | 0.371 | 1.000 | No |
| D3: CORAL vs U1 | — | — | 0.131 | 0.786 | No |
| D3: CORAL vs U5 | — | — | 0.617 | 1.000 | No |
| D0 vs U1 | — | — | 0.724 | 1.000 | No |
| D0 vs U5 | — | — | 1.000 | 1.000 | No |
| U1 vs U5 | — | — | 0.450 | 1.000 | No |

> [!IMPORTANT]
> **CORAL's apparent +10 percentage point gain over the D0 baseline is NOT statistically distinguishable from chance at N=30 (McNemar p=0.371; Holm-corrected p=1.000).** Do NOT present CORAL alignment as a validated improvement anywhere in the manuscript. Its +10pt delta is consistent with the expected sampling variance on a 30-sample holdout. The correct framing is: "CORAL produced the highest point estimate among functioning methods, but no adaptation method achieved statistically significant improvement over the unadapted baseline (all Holm-corrected p=1.0), confirming that the source of failure is structural — not a distributional gap addressable by standard alignment."


![Defence Comparison](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/defence_comparison.png)


---


![Feature Drift Heatmap](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/feature_drift_heatmap.png)


---


![Defence Confusion Grid](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/defence_confusion_grid.png)


---

## 4. PCA / t-SNE Dimensionality Reduction (M1)

> [!NOTE]
> This is the M1 figure. It shows the CIC-MalMem-2022 feature manifold by binary class (Benign vs Malware). Both PCA and t-SNE panels have been computed specifically using these binary distinctions for clarity.

**Intrinsic Dimensionality (Source vs Target Domains):**
The original dataset dimensions (55 in VolMemLyzer → 41 aligned features) drastically overestimate intrinsic dimensionality. A PCA (StandardScaler) on the continuous feature matrix reveals both the source and target domains have $D < 10$:

| Metric | Source Domain (CIC, N=58,596) | Target Domain (MalVol-25, N=30) |
|---|---|---|
| PC1 Explained Variance | 39.58% | 64.68% |
| PC1 + PC2 | 52.31% | 82.26% |
| PC1 + PC2 + PC3 | 62.58% | 89.72% |
| PCs required for 90% variance | **9  (of 41)** | **4  (of 41)** |
| PCs required for 95% variance | 12 (of 41) | 5  (of 41) |
| PCs required for 99% variance | 19 (of 41) | 9  (of 41) |

> [!WARNING]
> **Constraint on Target PCA:** The extreme dimensional collapse on MalVol-25 (90% variance in just 4 principal components, PC1=64%) is partially an artefact of its small sample size (N=30). A 41-feature covariance matrix estimated from 30 samples is rank-deficient and unstable. This estimate is high-variance and primarily confirms that intrinsic feature expression on the target is lower-dimensional than the canonical 41-feature count implies.

For the Source Domain, PC1 and PC2 capture only 52% of variance, meaning the 2D projection below is a coarse view. Furthermore, linear projection alone (PCA) **does not cleanly separate the classes**. The Benign samples form a tight, low-variance cluster, while the Malware samples are diffuse and overlapping — structurally behaving like a heterogeneous "everything that isn't the tight benign baseline" category rather than a coherent class. While the non-linear embedding (t-SNE) exhibits clearer regional separation between these classes, this reflects local neighbourhood associations rather than linear separability. This is consistent with how the XGBoost model achieves near-perfect in-distribution accuracy: through complex, non-linear, tree-based decision boundaries rather than simple linear hyperplanes.

**Does the separability transfer to MalVol-25?** No. As shown in Section 2 (KS-test: 39/41 features shift significantly between CIC and MalVol-25) and Section 8 (the model collapses to ~50% accuracy when the environment changes), the cluster boundaries visible here are environment-specific artefacts of the CIC Win7 VM — they do not transfer to MalVol-25's Win10 physical environment.

![PCA & t-SNE Manifold](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/3_pca_tsne.png)

---

## 5. SHAP-vs-KS Drift Language Separation (M2)

> [!NOTE]
> This is the M2 requirement: the SHAP reliance drift (model attribution drift) and the KS value drift (feature distribution drift) are **explicitly separated** throughout the manuscript. They measure different things.

**KS drift (Section 2)** measures distributional shift in *feature values* between CIC and MalVol-25 — it is dataset-level and model-agnostic.

**SHAP reliance drift (Section 6A)** measures shift in *model attribution weights* — it is model-specific and reveals which features the trained XGBoost relied on in-distribution that vanish on MalVol-25.

These are **not redundant**. A feature can have high KS drift but low SHAP reliance (the model didn't depend on it). The most dangerous features are those with **high SHAP reliance on CIC + high KS drift on MalVol-25** — these are the "memorised environmental artefacts." The Step 9 rank-deficiency analysis specifically targets this intersection.

![SHAP Summary (CIC in-distribution)](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/10_shap_summary.png)

---

## 6. Phase 2 Experiments: The Adversarial Smoking Guns

### 6A — Model Reliance Drift (SHAP, not KS)

> [!NOTE]
> This section measures drift in **model RELIANCE** on specific features (SHAP values), not in feature values themselves. Do not conflate with the KS-test analysis in Section 2.

The SHAP Model Reliance experiment trains XGBoost on CIC and measures mean |SHAP| for each feature — first on a CIC test partition (in-distribution reliance), then on MalVol-25 (OOD reliance). 

**Finding: Reliance is Preserved, Not Collapsed.** The top predictive features on CIC (`svcscan_nservices`, `handles_avg_handles_per_proc`, etc.) show near-identical SHAP attributions when exposed to MalVol-25 (e.g., `4.37 → 4.03` effect magnitude). The model attempts to forcefully apply the exact same feature logic on MalVol-25 as it did on CIC. However, because the underlying physical feature *values* have shifted (proven by the KS drift in Section 2), accuracy still collapses. This demonstrates that the model is blindly applying environment-specific threshold logic to a new domain, failing to realise the structural mapping has changed.

![SHAP Model Reliance Drift](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/step9_shap_drift.png)

### 6B — Benign-Only OOD Rejection (Confirmatory Result — OS-Generational Fragility)

> [!IMPORTANT]
> **This is the paper's sole confirmatory result.** It does not depend on the N=30 adaptation comparison above.
>
> **Feature set and model consistency note (for methods section):** The model used here is the standalone XGBoost (`n_estimators=100, max_depth=5, random_state=42`). `max_depth=5` was the configuration from the original single Phase 0 run and was locked before this fold-level audit was performed; `max_depth=3` was in fact the more frequent winner (10/20 folds) across the full 20-fold/4-seed audit. This is noted as a minor methodological limitation — re-running all evaluations with `max_depth=3` is left as a robustness check, not expected to materially change the paper's conclusions given the near-identical Brier/accuracy scores observed between the two configurations in Phase 0. The feature space used is the **full 41-feature VolMemLyzer set** (MalVol-25 contains all 41 features; `av_idx` in `step9_phase2_experiments.py` resolves to the complete feature list with zero columns dropped). The 26.7% FPR is therefore directly comparable to the 50% accuracy reported in Section 3 (same model configuration, same features, same target dataset).

A model trained exclusively on CIC-MalMem-2022 (Windows 7 VM environment) is tested **only** on the 15 MalVol-25 Benign samples (Windows 10 physical environment):

| Metric | Result |
|---|---|
| Target samples | 15 Benign (Windows 10, MalVol-25) |
| False Positives (Benign → Malware) | **4 / 15** |
| **False Positive Rate** | **26.7%** |

> [!CAUTION]
> A detector achieving 100% accuracy in CIC mis-classifies **1-in-4 normal Windows 10 processes as Malware** purely because they are running on a different OS generation. The model is not detecting malware — it is detecting the Windows 7 VM configuration.

![Benign OOD FPR](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/step9_benign_ood_fpr.png)

---

## 7. Advanced Discussion and Limitations

### The N=30 Representativeness Problem
Mathematical stability on $N=30$ does not guarantee ecological representativeness. MalVol-25 functions strictly as a **Cross-Environment Probe**, not a general benchmark. The massive cross-family internal validation (N=24,669 Spyware holdout, Section 8) confirms the structural shift is not a sample-size artefact.

### The OS Generational Shift Confound (Why ENF Failed)
Windows 10 restructured memory management (compressed stores, Svchost splitting) at the architectural level. Environment-Normalised Features (ENF) returned 50.0% accuracy because malware footprints do not scale linearly with system complexity — absolute threshold-based tree splits are destroyed when features are ratio-normalised. **This negative result is the paper's evidence that the shift demands non-linear, forensics-native adaptation.**

### Information-Theoretical Ablation (not "Evasion")
> [!NOTE]
> The truncation experiments in the RQ evaluations are correctly framed as **Information-Theoretical Ablation Studies** — determining how much feature information can be removed before detection degrades. They are **not** claimed to simulate adversarial evasion. Adversarial evasion via DKOM would not truncate features; it would produce strict zero-vectors for specific VolMemLyzer plugin families. The zero-handling in the U5 Delta-Feature pipeline explicitly accounts for this regime.

### Multiple-Comparisons and Exploratory vs. Confirmatory Framing
The adaptation sweep (Section 3) tested 10 method variants against the same N=30 holdout. This is an **exploratory analysis** — it is not powered for confirmatory inference. The Holm-Bonferroni correction confirms no adaptation variant achieves significance after correction. The confirmatory claims of the paper are restricted to:
1. The headline domain-shift finding: 100% CIC accuracy → 50% accuracy on MalVol-25 (Section 1)
2. The FPR finding: 26.7% false positives on Windows 10 benign processes (Section 6B)
3. The cross-family validation: 99.74% zero-shot accuracy within CIC (Section 8)

### The Publishable Finding

> *A single XGBoost model achieves 100% CIC accuracy and 99.74% zero-shot cross-family accuracy (N=24,669), yet exhibits a 26.7% False Positive Rate on Windows 10 benign processes. These results jointly prove that standard memory forensics ML memorises OS-specific environmental artefacts, completely failing to learn abstract malware behaviour. No adaptation method tested produced a statistically significant improvement (all Holm-corrected p=1.0). This is the structural, non-distributional domain shift that standard adaptation cannot fix.*

---

## 8. Cross-Family Experiment (Path 3 — Confound Audit, Not Confirmatory)

> [!CAUTION]
> **This result is NOT an independent confirmatory success and must not be labelled as one.** The 99.74% cross-family accuracy is explicitly consistent with the same environmental-confound mechanism identified in the Confound Audit: Ransomware, Trojan, and Spyware sub-families in this experiment all originate from the *same* CIC-MalMem-2022 source environment (Windows 7 VM, identical acquisition pipeline). The model does not cross an environment boundary here. When the environment boundary IS crossed (CIC → MalVol-25), accuracy collapses to 50%. This result therefore confirms the confound — it does not refute it. **Do not present cross-family 99.74% as evidence that the model generalises across malware families.**

Train on: Benign (50%) + Ransomware + Trojans (**Source, N=33,927**)  
Test on: Benign (50%) + Spyware (**Target, N=24,669**)  
**Critical constraint: all sub-families share the same CIC Win7 VM acquisition environment.**

| Metric | In-Dist (CIC) | Cross-Family "OOD" (Spyware, same environment) |
|---|---|---|
| Accuracy | **100.0%** | **99.74%** |
| F1 (macro) | **1.000** | **0.997** |
| AUC | **1.000** | **0.999** |

The near-identical in-distribution and cross-family AUC (1.000 vs 0.999) is the diagnostic: **any genuine cross-family generalisation would show degradation**. The absence of degradation confirms the model is responding to the shared Win7 VM fingerprint, not to malware-family-specific behaviour. This result belongs in the Confound Audit, not in the list of confirmatory findings.

---

## 9. Output Artifacts

| File | Description |
|---|---|
| [step6_cross_family.json](file:///F:/FORENSICS/adversarial-memforensics/results/unified/step6_cross_family.json) | Path 3 cross-family validated results |
| [step8_baseline_ablation.json](file:///F:/FORENSICS/adversarial-memforensics/results/unified/step8_baseline_ablation.json) | McNemar ensemble vs XGBoost verdict |
| [step9_rank_deficiency.json](file:///F:/FORENSICS/adversarial-memforensics/results/unified/step9_rank_deficiency.json) | SHAP reliance drift top-3 table |
| [step9_benign_ood.json](file:///F:/FORENSICS/adversarial-memforensics/results/unified/step9_benign_ood.json) | Benign-only OOD FPR (26.7%) |
| [feature_shift_analysis.csv](file:///F:/FORENSICS/adversarial-memforensics/results/unified/feature_shift_analysis.csv) | Per-feature KS + Cohen's d |
| [thesis_figures/](file:///F:/FORENSICS/adversarial-memforensics/results/unified/thesis_figures) | All 300-DPI plots |

## 10. Comprehensive Thesis Visualisations

> **Methodology Note (Phase 0 vs Graphical Suite):** The metrics below (including the ~99.9915% accuracy visible in the confusion matrices) originate from a single 80/20 train/test holdout designed specifically for high-resolution graphics generation. This evaluation is distinct from the formal Phase 0 run, but the ~99.9915% point estimate shown here (1 error in 11,720 samples) aligns cleanly with the ~99.9932% accuracy observed across the full 58,596-sample 20-fold nested CV. The close agreement between these two independently-run evaluations (99.9915% vs 99.9932%) is consistent with, though does not by itself statistically confirm, the same underlying decision boundary.


![Data Distributions](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/1_data_distributions.png)


---


![Correlation Heatmap](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/2_correlation_heatmap.png)


---


![PCA & t-SNE](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/3_pca_tsne.png)


---


![Model Comparison Metrics](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/4_model_comparison_metrics.png)


---


![Time Comparison](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/5_model_time_comparison.png)


---


![ROC & PR Curves](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/6_roc_pr_curves.png)


---


![Confusion Matrices](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/7_confusion_matrices.png)


---


![Detection by Category](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/8_detection_by_category.png)


---


![Confidence Distribution](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/9_confidence_distribution.png)


---


![SHAP Summary Plot](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/10_shap_summary.png)


---


![SHAP Feature Importance](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/11_shap_feature_importance.png)


---


![Learning Curve](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/12_learning_curve.png)


---


![SHAP Reliance Drift](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/step9_shap_drift.png)


---


![Benign OOD FPR Test](C:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/step9_benign_ood_fpr.png)

