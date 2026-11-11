"""
step5_eval_tables.py
====================
Compiles all JSON results from Step 2, 3, and 4 into a clean LaTeX table
and a formatted text summary for copying straight into the manuscript.

Outputs formatting for IEEE/Elsevier class standards.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "results" / "unified"

def main():
    with open(OUT_DIR / "step2_statistical.json") as f:
        d2 = json.load(f)
    
    with open(OUT_DIR / "step4_enf.json") as f:
        d4 = json.load(f)
        
    print("="*80)
    print("FINAL Q1 EXPERIMENTAL RESULTS: CROSS-DATASET EVALUATION")
    print("="*80)
    
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cross-Dataset Transfer Performance: CIC-MalMem-2022 \\rightarrow MalVol-25}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l l l l c}",
        "\\toprule",
        "\\textbf{Adaptation Family} & \\textbf{Method} & \\textbf{Evaluation Protocol} & \\textbf{Accuracy (\\pm\\%} CI/Std) & \\textbf{F1-Macro} \\\\",
        "\\midrule"
    ]
    
    # Zero shot
    print(f"\n--- ZERO SHOT TRANSDUCTIVE ---")
    for r in d2["loocv"]:
        print(f"{r['method']:<30s} {r['accuracy']:.4f}")
        tex.append(f"Unsupervised DA & {r['method'].replace('D0: ', '').replace('D3: ', '').replace('U1: ', '').replace('U5: ', '')} & Transductive & {r['accuracy']:.3f} & {r['f1_macro']:.3f} \\\\")
        
    print(f"\n--- FEW-SHOT FINETUNING (100 Trials) ---")
    for r in d2["trials"]:
        print(f"{r['method']:<30s} {r['accuracy_mean']:.4f} \u00b1 {r['accuracy_std']:.4f}")
        tex.append(f"Supervised 5-Shot & {r['method'].replace('U2: ', '').replace('U2b: ', '').replace('U2c: ', '')} & 100-Trial Repeated & {r['accuracy_mean']:.3f} \\pm {r['accuracy_std']:.3f} & {r['f1_mean']:.3f} \\\\")
        
    print(f"\n--- FORENSICS-AWARE DOMAIN ADAPTATION ---")
    print(f"{d4['method']:<30s} {d4['accuracy']:.4f}")
    tex.append(f"Domain-Aware & {d4['method']} & Transductive & {d4['accuracy']:.3f} & {d4['f1_macro']:.3f} \\\\")
    
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\label{tab:final_results}",
        "\\vspace{-2mm}",
        "\\end{table*}"
    ])
    
    print("\n\n" + "="*80)
    print("LATEX SOURCE (COPY-PASTE READY FOR OVERLEAF)")
    print("="*80)
    print("\n".join(tex))

if __name__ == "__main__":
    main()
