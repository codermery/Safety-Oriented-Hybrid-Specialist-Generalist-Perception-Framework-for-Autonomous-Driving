"""
MAKALE TABLO VE GRAFİKLERİ (Reviewer İsekleri & Meta-Reviewer)

Okunan Dosyalar (outputs/results/):
  - evaluation_summary.json
  - inference_speed.json
  - alternative_fusion_results.json
  - combined_metrics.json
  - specialist_metrics.json, generalist_metrics.json, hybrid_late_nms_metrics.json, hybrid_separate_nms_metrics.json

Çıktılar:
  - outputs/tables/*.tex (LaTeX formatında tablolar)
  - outputs/figures/precision_recall_curves.png (Precision-Recall eğrisi)
  - outputs/figures/class_f1_bar_chart.png (Sınıf bazlı F1-Skor bar grafiği)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *

def read_json(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}. Using empty fallback.")
        return {}
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("MAKALE MATERYALLERİ OLUŞTURULUYOR")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR / "tables", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "figures", exist_ok=True)

    # 1. JSON'ları yükle
    summary = read_json(OUTPUT_DIR / "results" / "evaluation_summary.json")
    speed = read_json(OUTPUT_DIR / "results" / "inference_speed.json")
    alt_fusion = read_json(OUTPUT_DIR / "results" / "alternative_fusion_results.json")
    combined = read_json(OUTPUT_DIR / "results" / "combined_metrics.json")
    
    spec_detailed = read_json(OUTPUT_DIR / "results" / "specialist_metrics.json")
    gen_detailed = read_json(OUTPUT_DIR / "results" / "generalist_metrics.json")
    hybrid_detailed = read_json(OUTPUT_DIR / "results" / "hybrid_late_nms_metrics.json")
    sep_detailed = read_json(OUTPUT_DIR / "results" / "hybrid_separate_nms_metrics.json")

    # Fallback to combined summary if files are empty
    if not summary:
        print("Evaluation summary is empty. Generate assets cannot run. Exiting.")
        return

    # ====================================================================
    # TABLO I: Genişletilmiş Model Karşılaştırma
    # ====================================================================
    print("Tablo I: Genişletilmiş Model Karşılaştırma LaTeX Tablosu oluşturuluyor...")
    table_1_path = OUTPUT_DIR / "tables" / "table_1_model_comparison.tex"
    
    # Check if combined baseline is available
    combined_mAP_50 = combined.get("mAP@0.50", 0.0) * 100 if combined else 0.0
    combined_AR = combined.get("AR@100", 0.0) * 100 if combined else 0.0

    tex_1 = r"""\begin{table}[htbp]
\caption{Genişletilmiş Model Karşılaştırma Sonuçları (CODA ve COCO veri setleri üzerinde)}
\label{tab:model_comparison}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{mAP@50 (\%)} & \textbf{AR@100 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} & \textbf{FPR (\%)} \\ \hline
"""
    # Specialist
    s = summary.get("specialist", {})
    tex_1 += f"Tekil Uzman (Baseline) & {s.get('mAP_50',0)*100:.2f} & {s.get('AR_100',0)*100:.2f} & {s.get('precision',0)*100:.2f} & {s.get('recall',0)*100:.2f} & {s.get('f1',0)*100:.2f} & {s.get('fpr',0)*100:.2f} \\\\\n"
    # Generalist
    g = summary.get("generalist", {})
    tex_1 += f"Tekil Genelci & {g.get('mAP_50',0)*100:.2f} & {g.get('AR_100',0)*100:.2f} & {g.get('precision',0)*100:.2f} & {g.get('recall',0)*100:.2f} & {g.get('f1',0)*100:.2f} & {g.get('fpr',0)*100:.2f} \\\\\n"
    # Combined training baseline
    if combined:
        c_det = combined.get("detection_metrics", {}).get("overall", {})
        tex_1 += f"COCO+CODA Birleşik Eğitim Baseline & {combined_mAP_50:.2f} & {combined_AR:.2f} & {c_det.get('Precision',0)*100:.2f} & {c_det.get('Recall',0)*100:.2f} & {c_det.get('F1',0)*100:.2f} & {c_det.get('FPR',0)*100:.2f} \\\\\n"
    else:
        tex_1 += "COCO+CODA Birleşik Eğitim Baseline & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n"
    # Hybrid Separate NMS
    sep = summary.get("hybrid_separate_nms", {})
    tex_1 += f"Hibrit (Separate NMS - Alternatif) & {sep.get('mAP_50',0)*100:.2f} & {sep.get('AR_100',0)*100:.2f} & {sep.get('precision',0)*100:.2f} & {sep.get('recall',0)*100:.2f} & {sep.get('f1',0)*100:.2f} & {sep.get('fpr',0)*100:.2f} \\\\\n"
    # Hybrid Late NMS (Recommended)
    hl = summary.get("hybrid_late_nms", {})
    tex_1 += f"\\textbf{{Hibrit (Late Fusion - Önerilen)}} & \\textbf{{{hl.get('mAP_50',0)*100:.2f}}} & \\textbf{{{hl.get('AR_100',0)*100:.2f}}} & \\textbf{{{hl.get('precision',0)*100:.2f}}} & \\textbf{{{hl.get('recall',0)*100:.2f}}} & \\textbf{{{hl.get('f1',0)*100:.2f}}} & \\textbf{{{hl.get('fpr',0)*100:.2f}}} \\\\\n"
    
    tex_1 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_1_path, "w") as f:
        f.write(tex_1)

    # ====================================================================
    # TABLO II: Nesne Boyutuna Göre AP Karşılaştırması
    # ====================================================================
    print("Tablo II: Nesne Boyutuna Göre AP Karşılaştırması LaTeX Tablosu oluşturuluyor...")
    table_2_path = OUTPUT_DIR / "tables" / "table_2_object_size.tex"
    
    tex_2 = r"""\begin{table}[htbp]
\caption{Nesne Boyutuna Göre AP ve AR Metrikleri Karşılaştırması}
\label{tab:object_size}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{AP\_small (\%)} & \textbf{AP\_medium (\%)} & \textbf{AP\_large (\%)} & \textbf{AR\_small (\%)} & \textbf{AR\_medium (\%)} & \textbf{AR\_large (\%)} \\ \hline
"""
    for key, name in [("specialist", "Tekil Uzman (Baseline)"), ("generalist", "Tekil Genelci"), ("hybrid_separate_nms", "Hibrit (Separate NMS)"), ("hybrid_late_nms", "Hibrit (Late Fusion)")]:
        det = read_json(OUTPUT_DIR / "results" / f"{key}_metrics.json")
        coco = det.get("coco_metrics", {})
        bold = "\\textbf{" if "Late Fusion" in name else ""
        endbold = "}" if "Late Fusion" in name else ""
        
        tex_2 += f"{bold}{name}{endbold} & {bold}{coco.get('mAP_small',0)*100:.2f}{endbold} & {bold}{coco.get('mAP_medium',0)*100:.2f}{endbold} & {bold}{coco.get('mAP_large',0)*100:.2f}{endbold} & {bold}{coco.get('AR_small',0)*100:.2f}{endbold} & {bold}{coco.get('AR_medium',0)*100:.2f}{endbold} & {bold}{coco.get('AR_large',0)*100:.2f}{endbold} \\\\\n"
        
    tex_2 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_2_path, "w") as f:
        f.write(tex_2)

    # ====================================================================
    # TABLO III: Sınıf Bazlı Detaylı Performans
    # ====================================================================
    print("Tablo III: Sınıf Bazlı Detaylı Performans LaTeX Tablosu oluşturuluyor...")
    table_3_path = OUTPUT_DIR / "tables" / "table_3_class_breakdown.tex"
    
    tex_3 = r"""\begin{table}[htbp]
\caption{Önerilen Hibrit Modelin Sınıf Bazlı Detaylı Performans Dağılımı}
\label{tab:class_breakdown}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Sınıf Adı} & \textbf{Sınıf ID} & \textbf{True Positives} & \textbf{False Positives} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} \\ \hline
"""
    if hybrid_detailed:
        per_class = hybrid_detailed.get("detection_metrics", {}).get("per_class", {})
        for cid_str, met in per_class.items():
            cid = int(cid_str)
            name = CODA_CLASSES.get(cid, f"Sınıf {cid}")
            tex_3 += f"{name} & {cid} & {met.get('TP',0)} & {met.get('FP',0)} & {met.get('Precision',0)*100:.2f} & {met.get('Recall',0)*100:.2f} & {met.get('F1',0)*100:.2f} \\\\\n"
            
    tex_3 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_3_path, "w") as f:
        f.write(tex_3)

    # ====================================================================
    # TABLO IV: Füzyon Stratejisi Karşılaştırması
    # ====================================================================
    print("Tablo IV: Füzyon Stratejisi Karşılaştırması LaTeX Tablosu oluşturuluyor...")
    table_4_path = OUTPUT_DIR / "tables" / "table_4_fusion_strategy.tex"
    
    tex_4 = r"""\begin{table}[htbp]
\caption{Füzyon Stratejileri ve Farklı NMS Eşik Değerlerinin Performans Karşılaştırması}
\label{tab:fusion_strategy}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{NMS Stratejisi} & \textbf{NMS Eşiği} & \textbf{mAP@50 (\%)} & \textbf{mAP@75 (\%)} & \textbf{AR@100 (\%)} \\ \hline
"""
    if alt_fusion:
        for mode, name in [("late_nms", "Geç Füzyon (Late Fusion + NMS)"), ("separate_nms", "Ayrı NMS (Separate NMS + Merge)")]:
            mode_data = alt_fusion.get(mode, {})
            for iou_val in sorted(mode_data.keys()):
                d = mode_data[iou_val]
                tex_4 += f"{name} & IoU={iou_val} & {d.get('mAP_50',0)*100:.2f} & {d.get('mAP_75',0)*100:.2f} & {d.get('AR_100',0)*100:.2f} \\\\\n"
                
    tex_4 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_4_path, "w") as f:
        f.write(tex_4)

    # ====================================================================
    # TABLO V: Inference Hızı Karşılaştırması
    # ====================================================================
    print("Tablo V: Gecikme ve FPS Karşılaştırması LaTeX Tablosu oluşturuluyor...")
    table_5_path = OUTPUT_DIR / "tables" / "table_5_inference_speed.tex"
    
    tex_5 = r"""\begin{table}[htbp]
\caption{Farklı Modellerin Nvidia RTX 4090 GPU Üzerindeki Hız Karşılaştırması}
\label{tab:inference_speed}
\centering
\begin{tabular}{lcc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{Ortalama Gecikme (ms)} & \textbf{Çıkarım Hızı (FPS)} \\ \hline
"""
    if speed:
        for key in ["specialist", "generalist", "hybrid_separate_nms", "hybrid_late_nms"]:
            s_data = speed.get(key, {})
            name = s_data.get("model_name", key)
            tex_5 += f"{name} & {s_data.get('avg_latency_ms',0):.2f} $\\pm$ {s_data.get('std_latency_ms',0):.2f} & {s_data.get('fps',0):.2f} \\\\\n"
            
    tex_5 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(table_5_path, "w") as f:
        f.write(tex_5)

    # ====================================================================
    # ŞEKİL 1: Precision-Recall Eğrisi
    # ====================================================================
    print("Şekil 1: Precision-Recall curves plotting...")
    fig_pr_path = OUTPUT_DIR / "figures" / "precision_recall_curves.png"
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    
    detailed_metrics = {
        "Tekil Uzman (Baseline)": spec_detailed,
        "Tekil Genelci": gen_detailed,
        "Hibrit (Separate NMS)": sep_detailed,
        "Hibrit (Late Fusion - Önerilen)": hybrid_detailed
    }
    
    for label, det in detailed_metrics.items():
        if not det:
            continue
        pr_data = det.get("pr_curve_data", {})
        scores = np.array(pr_data.get("scores", []))
        matches = np.array(pr_data.get("matches", []))
        
        if len(matches) > 0:
            precision, recall, _ = precision_recall_curve(matches, scores)
            plt.plot(recall, precision, label=label, lw=2.5)

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Farklı Model Konfigürasyonlarının Precision-Recall Eğrileri", fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(fig_pr_path, dpi=300)
    plt.close()

    # ====================================================================
    # ŞEKİL 2: Sınıf Bazlı F1-Skor Bar Chart
    # ====================================================================
    print("Şekil 2: Sınıf Bazlı F1-Skor Bar Grafik oluşturuluyor...")
    fig_f1_path = OUTPUT_DIR / "figures" / "class_f1_bar_chart.png"
    
    if hybrid_detailed:
        per_class = hybrid_detailed.get("detection_metrics", {}).get("per_class", {})
        classes = []
        f1_scores = []
        
        for cid_str, met in per_class.items():
            cid = int(cid_str)
            name = CODA_CLASSES.get(cid, f"ID:{cid}")
            classes.append(name)
            f1_scores.append(met.get("F1", 0.0) * 100)

        # Sort by F1 score descending
        sorted_idx = np.argsort(f1_scores)[::-1]
        classes = [classes[i] for i in sorted_idx]
        f1_scores = [f1_scores[i] for i in sorted_idx]

        plt.figure(figsize=(14, 7))
        sns.barplot(x=classes, y=f1_scores, palette="viridis")
        plt.xlabel("Nesne Sınıfları", fontsize=14)
        plt.ylabel("F1-Skor (%)", fontsize=14)
        plt.title("Önerilen Hibrit Modelin Sınıf Bazlı F1-Skor Dağılımı", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylim([0.0, 105.0])
        plt.tight_layout()
        plt.savefig(fig_f1_path, dpi=300)
        plt.close()

    print("\n✅ Tüm makale tablo ve grafik varlıkları başarıyla üretildi!")

if __name__ == "__main__":
    main()
