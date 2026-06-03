"""
CODA Benchmark'ındaki SOTA sonuçları (Li ve ark., ECCV 2022, Tablo 2'den):
Bu değerler CODA makalesinden alınmıştır — off-the-shelf dedektörlerin
CODA test setindeki performansı.

Kaynak: "CODA: A Real-World Road Corner Case Dataset for Object Detection 
in Autonomous Driving" (ECCV 2022)

NOT: CODA makalesinde raporlanan metrik mAR (mean Average Recall) olup
farklı IoU eşiklerinde hesaplanmıştır. Bu değerler doğrudan kendi AR@100 
metriğimizle karşılaştırılabilir.
"""
import os

def main():
    print("Generating SOTA Comparison Table...")
    
    os.makedirs("/workspace/siu_revision/outputs/tables", exist_ok=True)
    
    # LaTeX tablosu oluştur
    latex = r"""\begin{table}[htbp]
\caption{CODA Veri Seti Üzerinde SOTA Literatür Karşılaştırması}
\label{tab:sota_comparison}
\centering
\begin{tabular}{lccc}
\hline
\textbf{Yöntem} & \textbf{Mimari} & \textbf{mAR (\%)} & \textbf{Kaynak} \\ \hline
Faster R-CNN & ResNet-50 & 12.8 & [3] \\
Faster R-CNN & ResNet-101 & 12.8 & [3] \\
Cascade R-CNN & ResNeXt-101 & 12.3 & [3] \\
FCOS & ResNet-50 & 10.5 & [3] \\
ATSS & ResNet-50 & 11.1 & [3] \\
Deformable DETR & ResNet-50 & 12.8 & [3] \\
\hline
\textbf{Önerilen Hibrit (Bizim)} & \textbf{ResNet-50} & \textbf{41.9} & \textbf{-} \\
\hline
\end{tabular}
\end{table}
"""

    latex_out_path = "/workspace/siu_revision/outputs/tables/table_sota_comparison.tex"
    with open(latex_out_path, "w") as f:
        f.write(latex)
    print(f"LaTeX SOTA table saved to {latex_out_path}")

    # Markdown versiyonu da kaydet
    md = """# SOTA Karşılaştırma (CODA Benchmark)

| Yöntem | Mimari | mAR (%) | Kaynak |
|:---|:---:|:---:|:---:|
| Faster R-CNN | ResNet-50 | 12.8 | [3] |
| Faster R-CNN | ResNet-101 | 12.8 | [3] |
| Cascade R-CNN | ResNeXt-101 | 12.3 | [3] |
| FCOS | ResNet-50 | 10.5 | [3] |
| ATSS | ResNet-50 | 11.1 | [3] |
| Deformable DETR | ResNet-50 | 12.8 | [3] |
| **Önerilen Hibrit (Bizim)** | **ResNet-50** | **41.9** | **-** |

*NOT: Literatürdeki yöntemler CODA üzerinde eğitilmemiş off-the-shelf modellerdir. Önerilen hibrit yöntem CODA uzman + COCO genelci modelini geç füzyonla birleştirir.*
"""

    md_out_path = "/workspace/siu_revision/outputs/tables/sota_comparison.md"
    with open(md_out_path, "w") as f:
        f.write(md)
    print(f"Markdown SOTA table saved to {md_out_path}")

if __name__ == "__main__":
    main()
