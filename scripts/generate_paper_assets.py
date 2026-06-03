import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

def read_json(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("GENERATING MANUSCRIPT TABLES AND FIGURES (Fair split)")
    print("=" * 60)

    RESULTS_DIR = "/workspace/siu_revision/outputs/results"
    TABLES_DIR = "/workspace/siu_revision/outputs/tables"
    FIGURES_DIR = "/workspace/siu_revision/outputs/figures"

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load data
    summary = read_json(f"{RESULTS_DIR}/fair_evaluation_summary.json")
    extra = read_json(f"{RESULTS_DIR}/extra_metrics.json")
    speed = read_json(f"{RESULTS_DIR}/inference_speed.json")

    if not summary or not extra:
        print("Missing evaluation files. Cannot generate assets.")
        return

    # Map keys for models
    configs = {
        "specialist_only": "Tekil Uzman (Yeniden Eğitilmiş)",
        "generalist_only": "Tekil Genelci",
        "hybrid_late_nms": "Hibrit (Late Fusion - Önerilen)",
        "hybrid_separate_nms": "Hibrit (Separate NMS - Alternatif)"
    }

    # ====================================================================
    # TABLO I: Genişletilmiş Model Karşılaştırma (LaTeX)
    # ====================================================================
    print("Generating Table I (LaTeX)...")
    table_1_path = f"{TABLES_DIR}/table_1_model_comparison.tex"
    
    tex_1 = r"""\begin{table}[htbp]
\caption{Adil Değerlendirme Bölümünde Model Karşılaştırma Sonuçları (300 Test Sahnesi)}
\label{tab:model_comparison}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{mAP (\%)} & \textbf{mAP50 (\%)} & \textbf{AR@100 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} & \textbf{FPR (\%)} \\ \hline
"""
    for key, display_name in configs.items():
        s = summary.get(key, {})
        ex = extra.get(key, {}).get("overall", {})
        
        bold_start = "\\textbf{" if "Late Fusion" in display_name else ""
        bold_end = "}" if "Late Fusion" in display_name else ""
        
        tex_1 += f"{bold_start}{display_name}{bold_end} & {bold_start}{s.get('mAP@[0.5:0.95]', 0):.1f}{bold_end} & {bold_start}{s.get('mAP@0.50', 0):.1f}{bold_end} & {bold_start}{s.get('AR@100', 0):.1f}{bold_end} & {bold_start}{ex.get('Precision', 0):.1f}{bold_end} & {bold_start}{ex.get('Recall', 0):.1f}{bold_end} & {bold_start}{ex.get('F1', 0):.1f}{bold_end} & {bold_start}{ex.get('FPR', 0):.1f}{bold_end} \\\\\n"

    tex_1 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_1_path, "w") as f:
        f.write(tex_1)
    print(f"Saved: {table_1_path}")

    # ====================================================================
    # TABLO II: Nesne Boyutuna Göre AP/AR Dağılımı (LaTeX)
    # ====================================================================
    print("Generating Table II (LaTeX)...")
    table_2_path = f"{TABLES_DIR}/table_2_object_size.tex"
    
    tex_2 = r"""\begin{table}[htbp]
\caption{Nesne Boyutuna Göre AP ve AR Metrikleri Karşılaştırması}
\label{tab:object_size}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{AP\_small (\%)} & \textbf{AP\_medium (\%)} & \textbf{AP\_large (\%)} & \textbf{AR\_small (\%)} & \textbf{AR\_medium (\%)} & \textbf{AR\_large (\%)} \\ \hline
"""
    for key, display_name in configs.items():
        s = summary.get(key, {})
        bold_start = "\\textbf{" if "Late Fusion" in display_name else ""
        bold_end = "}" if "Late Fusion" in display_name else ""
        
        tex_2 += f"{bold_start}{display_name}{bold_end} & {bold_start}{s.get('AP_small', 0):.1f}{bold_end} & {bold_start}{s.get('AP_medium', 0):.1f}{bold_end} & {bold_start}{s.get('AP_large', 0):.1f}{bold_end} & {bold_start}{s.get('AR_small', 0):.1f}{bold_end} & {bold_start}{s.get('AR_medium', 0):.1f}{bold_end} & {bold_start}{s.get('AR_large', 0):.1f}{bold_end} \\\\\n"

    tex_2 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_2_path, "w") as f:
        f.write(tex_2)
    print(f"Saved: {table_2_path}")

    # ====================================================================
    # TABLO III: Sınıf Bazlı Detaylı Performans (LaTeX)
    # ====================================================================
    print("Generating Table III (LaTeX)...")
    table_3_path = f"{TABLES_DIR}/table_3_class_breakdown.tex"
    
    tex_3 = r"""\begin{table}[htbp]
\caption{Sınıf Bazlı Detaylı F1-Skor ve Doğruluk Karşılaştırması}
\label{tab:class_breakdown}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Nesne Sınıfı} & \multicolumn{3}{c}{\textbf{Tekil Uzman (Yeniden Eğitilmiş) (\%)}} & \multicolumn{3}{c}{\textbf{Önerilen Hibrit Model (\%)}} \\ \cline{2-7} 
 & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} \\ \hline
"""
    spec_pc = extra.get("specialist_only", {}).get("per_class", {})
    hybrid_pc = extra.get("hybrid_late_nms", {}).get("per_class", {})
    
    for cname in sorted(spec_pc.keys()):
        sp = spec_pc[cname]
        hb = hybrid_pc.get(cname, {"Precision": 0.0, "Recall": 0.0, "F1": 0.0})
        
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
    table_4_path = f"{TABLES_DIR}/table_4_inference_speed.tex"
    
    tex_4 = r"""\begin{table}[htbp]
\caption{Inference Gecikmesi ve FPS Karşılaştırmaları}
\label{tab:inference_speed}
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

    # ====================================================================
    # ŞEKİL 1: Precision-Recall Curves
    # ====================================================================
    print("Plotting Precision-Recall curves...")
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    for key, display_name in configs.items():
        ex_data = extra.get(key, {})
        pr_data = ex_data.get("pr_curve_data", {})
        scores = np.array(pr_data.get("scores", []))
        matches = np.array(pr_data.get("matches", []))
        
        if len(matches) > 0:
            precision, recall, _ = precision_recall_curve(matches, scores)
            plt.plot(recall, precision, label=display_name, lw=2.5)

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curves (Fair Split Evaluation)", fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    
    fig_pr_path = f"{FIGURES_DIR}/precision_recall_curves.png"
    plt.savefig(fig_pr_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_pr_path}")

    # ====================================================================
    # ŞEKİL 2: F1 Bar Chart Comparison
    # ====================================================================
    print("Plotting class-wise F1 bar chart comparison...")
    classes = sorted(spec_pc.keys())
    spec_f1 = [spec_pc[c]["F1"] for c in classes]
    hybrid_f1 = [hybrid_pc.get(c, {"F1": 0.0})["F1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(15, 8))
    plt.bar(x - width/2, spec_f1, width, label="Tekil Uzman", color="#1f77b4")
    plt.bar(x + width/2, hybrid_f1, width, label="Hibrit Model", color="#aec7e8")

    plt.xlabel("Nesne Sınıfları", fontsize=14)
    plt.ylabel("F1-Skor (%)", fontsize=14)
    plt.title("Tekil Uzman vs Hibrit Model - Sınıf Bazlı F1-Skor Karşılaştırması", fontsize=16, fontweight='bold')
    plt.xticks(x, classes, rotation=45, ha="right", fontsize=12)
    plt.ylim([0.0, 105.0])
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    fig_f1_path = f"{FIGURES_DIR}/class_f1_bar_chart.png"
    plt.savefig(fig_f1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_f1_path}")

    print("\n✅ All paper assets have been generated successfully!")

if __name__ == "__main__":
    main()
