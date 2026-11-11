# Spurious Correlations in RAM Forensics: Exposing the Cross-Environment Generalization Collapse of Memory-Based Malware Classifiers

---

### Figure 1: The "Illusion of Perfection" (In-Distribution Success)
**File to Use:** `4_model_comparison_metrics.png` (or `6_roc_pr_curves.png`)  
**Where to put it:** Early in the paper (Methods / Initial Evaluation).  
**What it proves:** Establishes the baseline expectation. Standard ML works flawlessly when the environment doesn't change.

**Recommended Caption for Manuscript:**
> *Figure 1: In-distribution performance metrics evaluated on the CIC-MalMem-2022 dataset using an 80/20 partition. The standalone XGBoost classifier achieves near-perfect classification (Accuracy ≈ 99.99%, AUC = 1.000). While these metrics suggest a solved problem under traditional independent and identically distributed (i.i.d.) assumptions, this performance ceiling is an artefact of environmental uniformity. The model successfully memorises the homogeneous Windows 7 OS state rather than learning generalized adversarial behaviours.*

---

### Figure 2: The Structural Geometry (PCA & t-SNE)
**File to Use:** `3_pca_tsne.png` (The updated binary version)  
**Where to put it:** Right after Figure 1.  
**What it proves:** Demystifies *why* XGBoost achieves 99.9% accuracy. It shows that Benign memory is a tight, predictable cluster, while Malware is just "everything else."

**Recommended Caption for Manuscript:**
> *Figure 2: 2D dimensionality reduction (PCA and t-SNE) of the 41-feature VolMemLyzer space by binary class. Crucially, linear projection (PCA, left) reveals that the classes are not cleanly linearly separable. Instead, the Benign samples form a highly stable, low-variance central cluster, while Malware samples are diffuse and overlapping. The model achieves near-perfect accuracy not by learning a "malware signature," but by mapping the tight boundaries of the specific benign OS environment (Windows 7). Anything falling outside this narrow, memorised physical state is flagged as anomalous.*

---

### Figure 3: The Environmental Shift (KS-Test Drift)
**File to Use:** `feature_drift_heatmap.png` (or `grouped_drift_bars.png`)  
**Where to put it:** The start of the Cross-Dataset/Failure analysis.  
**What it proves:** The root cause of the paper's entire premise. The memory structures physically change when moving from Win7 (CIC) to Win10 (MalVol-25).

**Recommended Caption for Manuscript:**
> *Figure 3: Kolmogorov-Smirnov (KS) divergence and Cohen’s d effect sizes comparing feature distributions between the source domain (Win7 VM) and target domain (Win10 Physical). Analysis reveals catastrophic covariate shift: 39 of the 41 canonical features exhibit statistically significant distribution drift (p < 0.05). The physical memory layout fundamentally changes across OS versions, meaning a model trained on source data is mathematically destined to process out-of-distribution inputs when deployed on modern physical hardware.*

---

### Figure 4: The Blind Reliance (SHAP Model Reliance)
**File to Use:** `step9_shap_drift.png`  
**Where to put it:** The "Smoking Gun" / Root Cause Analysis section.  
**What it proves:** Proves that the model didn't just break randomly—it actively tried to use Win7 logic on Win10 memory.

**Recommended Caption for Manuscript:**
> *Figure 4: Global feature reliance (mean |SHAP| value) measured across both the in-distribution (CIC) and out-of-distribution (MalVol-25) datasets. Feature importance ranking is preserved across domains; for example, the primary predictive feature (`svcscan_nservices`) maintains a near-identical SHAP magnitude (4.37 vs 4.03) despite the domain shift. This confirms the classifier does not fail organically; rather, it forcefully and blindly applies environment-specific baseline thresholds to a structurally foreign OS environment, resulting in systemic classification collapse.*

---

### Figure 5: The Ultimate Proof (Benign False Positive Rate)
**File to Use:** `step9_benign_ood_fpr.png`  
**Where to put it:** Immediately following Figure 4.  
**What it proves:** It isolates the variable. By feeding the model totally harmless Win10 memory, the model flags it as malware purely because the OS looks different.

**Recommended Caption for Manuscript:**
> *Figure 5: False Positive Rate (FPR) isolated exclusively on pristine, benign target memory dumps (MalVol-25 Benign, N=15). Despite containing zero malicious artefacts, the model exhibits a 26.7% false positive rate. Because no adversarial variables are present, this misclassification is definitively caused by the model misinterpreting standard Windows 10 OS structures as malicious deviations from its learned Windows 7 baseline, confirming that traditional memory forensics classifiers learn the environment rather than the threat.*

---

### Figure 6: The Failure of Algorithms (Domain Adaptation Defences)
**File to Use:** `defence_comparison.png` (or `upgrade_progression.png`)  
**Where to put it:** The Mitigation / Towards Generalisation section.  
**What it proves:** That you can't just slap an algorithm (like 5-shot learning or TCA) on this physical problem to fix it.

**Recommended Caption for Manuscript:**
> *Figure 6: Evaluation of statistical domain adaptation techniques (CORAL, TCA) and few-shot fine-tuning applied to bridge the source and target domains. While CORAL alignment provides a modest baseline improvement, all techniques fail to restore performance to practical operational levels, plateauing well below 65% accuracy. This indicates that the domain gap between OS memory environments is non-linear and structurally severe, resisting superficial statistical feature-space alignments. Future research must move away from purely mathematical mappings and integrate semantic, OS-aware feature normalisation prior to model ingestion.*
