import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from pathlib import Path

def read_json(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("GENERATING MANUSCRIPT TABLES AND FIGURES (1500 Scenes - Orijinal)")
    print("=" * 60)

    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
    TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
    FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load data
    summary = read_json(RESULTS_DIR / "evaluation_summary.json")
    extra = read_json(RESULTS_DIR / "extra_metrics.json")
    speed = read_json(RESULTS_DIR / "inference_speed.json")

    if not summary or not extra:
        print("Missing evaluation files. Cannot generate assets.")
        return

    # Map keys for models
    configs = {
        "specialist_only": "Tekil Uzman (Sızıntılı Baseline)",
        "generalist_only": "Tekil Genelci",
        "hybrid_late_nms": "Hibrit (Late Fusion - Önerilen)",
        "hybrid_separate_nms": "Hibrit (Separate NMS - Alternatif)"
    }

    # ====================================================================
    # TABLO I: Genişletilmiş Model Karşılaştırma (LaTeX)
    # ====================================================================
    print("Generating Table I (LaTeX)...")
    table_1_path = TABLES_DIR / "table_1_model_comparison_1500.tex"
    
    tex_1 = r"""\begin{table}[htbp]
\caption{Orijinal Değerlendirme Bölümünde Model Karşılaştırma Sonuçları (1500 Sahnede Sızıntılı Baseline)}
\label{tab:model_comparison_1500}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{mAP (\%)} & \textbf{mAP50 (\%)} & \textbf{AR@100 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} & \textbf{FPR (\%)} \\ \hline
"""
    for key, display_name in configs.items():
        s = summary.get(key, {})
        
        bold_start = "\\textbf{" if "Late Fusion" in display_name else ""
        bold_end = "}" if "Late Fusion" in display_name else ""
        
        tex_1 += f"{bold_start}{display_name}{bold_end} & {bold_start}{s.get('mAP_50_95', 0)*100:.1f}{bold_end} & {bold_start}{s.get('mAP_50', 0)*100:.1f}{bold_end} & {bold_start}{s.get('AR_100', 0)*100:.1f}{bold_end} & {bold_start}{s.get('Precision', 0):.1f}{bold_end} & {bold_start}{s.get('Recall', 0):.1f}{bold_end} & {bold_start}{s.get('F1', 0):.1f}{bold_end} & {bold_start}{s.get('FPR', 0):.1f}{bold_end} \\\\\n"

    tex_1 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_1_path, "w") as f:
        f.write(tex_1)
    print(f"Saved: {table_1_path}")

    # ====================================================================
    # TABLO III: Sınıf Bazlı Detaylı Performans (LaTeX)
    # ====================================================================
    print("Generating Table III (LaTeX)...")
    table_3_path = TABLES_DIR / "table_3_class_breakdown_1500.tex"
    
    tex_3 = r"""\begin{table}[htbp]
\caption{Sınıf Bazlı Detaylı F1-Skor ve Doğruluk Karşılaştırması (1500 Sahne)}
\label{tab:class_breakdown_1500}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Nesne Sınıfı} & \multicolumn{3}{c}{\textbf{Tekil Uzman (Sızıntılı Baseline) (\%)}} & \multicolumn{3}{c}{\textbf{Önerilen Hibrit Model (\%)}} \\ \cline{2-7} 
 & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} \\ \hline
"""
    spec_pc = extra.get("specialist_only", {}).get("per_class", {})
    hybrid_pc = extra.get("hybrid_late_nms", {}).get("per_class", {})
    
    # Class names mapping from IDs or keeping as names if present
    CODA_CLASSES = {
        "1": "Pedestrian", "2": "Cyclist", "3": "Car", "4": "Truck", "6": "Bus",
        "17": "Dog", "18": "Barrier", "23": "Cone", "26": "Traffic Sign",
        "27": "Debris", "30": "Concrete Block"
    }

    for cid_str in sorted(spec_pc.keys(), key=lambda x: int(x)):
        cname = CODA_CLASSES.get(cid_str, f"Sınıf {cid_str}")
        sp = spec_pc[cid_str]
        hb = hybrid_pc.get(cid_str, {"Precision": 0.0, "Recall": 0.0, "F1": 0.0})
        
        # Only show classes that have some ground truth annotations or predictions
        if sp.get("TP", 0) > 0 or sp.get("FP", 0) > 0 or hb.get("TP", 0) > 0 or hb.get("FP", 0) > 0:
            tex_3 += f"{cname} & {sp.get('Precision', 0):.1f} & {sp.get('Recall', 0):.1f} & {sp.get('F1', 0):.1f} & {hb.get('Precision', 0):.1f} & {hb.get('Recall', 0):.1f} & {hb.get('F1', 0):.1f} \\\\\n"

    tex_3 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_3_path, "w") as f:
        f.write(tex_3)
    print(f"Saved: {table_3_path}")

    # ====================================================================
    # TABLO IV: Inference Hızı Karşılaştırması (LaTeX)
    # ====================================================================
    print("Generating Table IV (LaTeX)...")
    table_4_path = TABLES_DIR / "table_4_inference_speed_1500.tex"
    
    tex_4 = r"""\begin{table}[htbp]
\caption{Inference Gecikmesi ve FPS Karşılaştırmaları (1500 Sahne)}
\label{tab:inference_speed_1500}
\centering
\begin{tabular}{lcc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{Ortalama Gecikme (ms)} & \textbf{Çıkarım Hızı (FPS)} \\ \hline
"""
    if speed:
        for key, display_name in configs.items():
            sp_data = speed.get(key, {})
            bold_start = "\\textbf{" if "Late Fusion" in display_name else ""
            bold_end = "}" if "Late Fusion" in display_name else ""
            
            tex_4 += f"{bold_start}{display_name}{bold_end} & {bold_start}{sp_data.get('avg_latency_ms', 0):.2f} $\\pm$ {sp_data.get('std_latency_ms', 0):.2f}{bold_end} & {bold_start}{sp_data.get('fps', 0):.2f}{bold_end} \\\\\n"

    tex_4 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_4_path, "w") as f:
        f.write(tex_4)
    print(f"Saved: {table_4_path}")

    print("\n✅ All 1500-scene paper assets have been generated successfully!")

if __name__ == "__main__":
    main()
