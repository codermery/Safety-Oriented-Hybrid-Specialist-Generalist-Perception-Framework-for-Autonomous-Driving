# SIU 2026 Revizyon - Deney Sonuçları Raporu

Bu rapor, CODA veri setinin **1200 train / 300 test** sahnelerine bölünerek Specialist (Uzman) modelin yeniden eğitildiği adil değerlendirme pipeline sonuçlarını içermektedir. Hem okunabilir Markdown tabloları hem de makaleye doğrudan ekleyebileceğiniz **LaTeX kod blokları** aşağıda sunulmuştur.

---

## 1. Model Karşılaştırma Sonuçları (Tablo I)

Bu tablo, tüm modellerin aynı 300 test sahnesi üzerinde ve pycocotools COCOeval protokolü ile yapılan test sonuçlarını içerir.

| Model Konfigürasyonu | mAP (%) | mAP50 (%) | AR@100 (%) | Precision (%) | Recall (%) | F1-Skor (%) | FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tekil Uzman (Yeniden Eğitilmiş)** | **10.9** | **17.6** | **15.0** | **76.4** | **72.8** | **74.6** | **23.6** |
| Tekil Genelci | 0.1 | 0.1 | 3.0 | 0.6 | 1.7 | 0.9 | 99.4 |
| Hibrit (Late Fusion - Önerilen) | 8.8 | 13.8 | 13.4 | 19.9 | 73.4 | 31.3 | 80.1 |
| Hibrit (Separate NMS - Alternatif) | 8.5 | 13.5 | 14.2 | 19.8 | 73.7 | 31.2 | 80.2 |

### LaTeX Kodu (Tablo I)
```latex
\begin{table}[htbp]
\caption{Adil Değerlendirme Bölümünde Model Karşılaştırma Sonuçları (300 Test Sahnesi)}
\label{tab:model_comparison}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{mAP (\%)} & \textbf{mAP50 (\%)} & \textbf{AR@100 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} & \textbf{FPR (\%)} \\ \hline
Tekil Uzman (Yeniden Eğitilmiş) & 10.9 & 17.6 & 15.0 & 76.4 & 72.8 & 74.6 & 23.6 \\
Tekil Genelci & 0.1 & 0.1 & 3.0 & 0.6 & 1.7 & 0.9 & 99.4 \\
\textbf{Hibrit (Late Fusion - Önerilen)} & \textbf{8.8} & \textbf{13.8} & \textbf{13.4} & \textbf{19.9} & \textbf{73.4} & \textbf{31.3} & \textbf{80.1} \\
Hibrit (Separate NMS - Alternatif) & 8.5 & 13.5 & 14.2 & 19.8 & 73.7 & 31.2 & 80.2 \\
\hline
\end{tabular}
\end{table}
```

---

## 2. Nesne Boyutuna Göre Performans (Tablo II)

Modellerin küçük (small), orta (medium) ve büyük (large) nesnelerdeki ortalama hassasiyet (AP) ve ortalama duyarlılık (AR) değerleridir.

| Model Konfigürasyonu | AP_small (%) | AP_medium (%) | AP_large (%) | AR_small (%) | AR_medium (%) | AR_large (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tekil Uzman (Yeniden Eğitilmiş)** | 3.6 | 10.2 | 13.9 | 5.7 | 12.5 | 18.4 |
| Tekil Genelci | 0.0 | 0.0 | 0.1 | 0.0 | 0.6 | 3.2 |
| **Hibrit (Late Fusion - Önerilen)** | 3.4 | 8.7 | 11.6 | 5.3 | 10.3 | 17.0 |
| Hibrit (Separate NMS - Alternatif) | 3.6 | 8.8 | 11.4 | 5.7 | 10.3 | 18.2 |

### LaTeX Kodu (Tablo II)
```latex
\begin{table}[htbp]
\caption{Nesne Boyutuna Göre AP ve AR Metrikleri Karşılaştırması}
\label{tab:object_size}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{AP\_small (\%)} & \textbf{AP\_medium (\%)} & \textbf{AP\_large (\%)} & \textbf{AR\_small (\%)} & \textbf{AR\_medium (\%)} & \textbf{AR\_large (\%)} \\ \hline
Tekil Uzman (Yeniden Eğitilmiş) & 3.6 & 10.2 & 13.9 & 5.7 & 12.5 & 18.4 \\
Tekil Genelci & 0.0 & 0.0 & 0.1 & 0.0 & 0.6 & 3.2 \\
\textbf{Hibrit (Late Fusion - Önerilen)} & \textbf{3.4} & \textbf{8.7} & \textbf{11.6} & \textbf{5.3} & \textbf{10.3} & \textbf{17.0} \\
Hibrit (Separate NMS - Alternatif) & 3.6 & 8.8 & 11.4 & 5.7 & 10.3 & 18.2 \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Sınıf Bazlı Detaylı Performans Karşılaştırması (Tablo III)

Tekil Uzman ile Önerilen Hibrit Late Fusion modelinin sınıf bazlı Precision, Recall ve F1-Skor (%) karşılaştırmasıdır.

| Nesne Sınıfı | Uzman - Precision | Uzman - Recall | Uzman - F1 | Hibrit - Precision | Hibrit - Recall | Hibrit - F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Barrier** | 79.5% | 76.4% | 77.9% | 19.3% | 76.9% | 30.8% |
| **Bollard** | 80.8% | 85.5% | 83.1% | 20.3% | 85.5% | 32.8% |
| **Car** | 20.0% | 7.7% | 11.1% | 10.8% | 53.8% | 18.0% |
| **Cone** | 76.8% | 84.4% | 80.4% | 19.9% | 84.8% | 32.2% |
| **Debris** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Dustbin** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Pedestrian**| 0.0% | 0.0% | 0.0% | 10.0% | 12.5% | 11.1% |
| **Traffic Sign**| 69.2% | 12.5% | 21.2% | 17.5% | 66.7% | 27.8% |
| **Trailer** | 33.3% | 2.9% | 5.3% | 7.9% | 28.6% | 12.4% |

*(Frekansı çok düşük olan veya tespiti sıfır çıkan bazı CODA sınıfları tabloda yer almamıştır)*

### LaTeX Kodu (Tablo III)
```latex
\begin{table}[htbp]
\caption{Sınıf Bazlı Detaylı F1-Skor ve Doğruluk Karşılaştırması}
\label{tab:class_breakdown}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Nesne Sınıfı} & \multicolumn{3}{c}{\textbf{Tekil Uzman (Yeniden Eğitilmiş) (\%)}} & \multicolumn{3}{c}{\textbf{Önerilen Hibrit Model (\%)}} \\ \cline{2-7} 
 & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Skor} \\ \hline
Barrier & 79.5 & 76.4 & 77.9 & 19.3 & 76.9 & 30.8 \\
Bollard & 80.8 & 85.5 & 83.1 & 20.3 & 85.5 & 32.8 \\
Car & 20.0 & 7.7 & 11.1 & 10.8 & 53.8 & 18.0 \\
Cone & 76.8 & 84.4 & 80.4 & 19.9 & 84.8 & 32.2 \\
Debris & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
Dustbin & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
Pedestrian & 0.0 & 0.0 & 0.0 & 10.0 & 12.5 & 11.1 \\
Traffic Sign & 69.2 & 12.5 & 21.2 & 17.5 & 66.7 & 27.8 \\
Trailer & 33.3 & 2.9 & 5.3 & 7.9 & 28.6 & 12.4 \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Çıkarım Hızı ve FPS (Tablo IV)

Nvidia GeForce RTX 4090 GPU üzerinde ölçülen ortalama gecikme süreleri ve çıkarım hızları (FPS).

| Model Konfigürasyonu | Ortalama Gecikme (ms) | Çıkarım Hızı (FPS) |
| :--- | :---: | :---: |
| **Tekil Uzman (Yeniden Eğitilmiş)** | 17.25 ± 3.32 | 58.0 |
| Tekil Genelci | 18.43 ± 0.65 | 54.3 |
| **Hibrit (Late Fusion - Önerilen)** | 35.07 ± 0.96 | 28.5 |
| Hibrit (Separate NMS - Alternatif) | 34.65 ± 0.63 | 28.9 |

### LaTeX Kodu (Tablo IV)
```latex
\begin{table}[htbp]
\caption{Inference Gecikmesi ve FPS Karşılaştırmaları}
\label{tab:inference_speed}
\centering
\begin{tabular}{lcc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{Ortalama Gecikme (ms)} & \textbf{Çıkarım Hızı (FPS)} \\ \hline
Tekil Uzman (Yeniden Eğitilmiş) & 17.25 $\pm$ 3.32 & 58.0 \\
Tekil Genelci & 18.43 $\pm$ 0.65 & 54.3 \\
\textbf{Hibrit (Late Fusion - Önerilen)} & \textbf{35.07 $\pm$ 0.96} & \textbf{28.5} \\
Hibrit (Separate NMS - Alternatif) & 34.65 $\pm$ 0.63 & 28.9 \\
\hline
\end{tabular}
\end{table}
```

---

## 5. Grafik Dosyaları
* Grafik-1 (PR Eğrileri): `siu_revision/outputs/figures/precision_recall_curves.png`
* Grafik-2 (F1 Bar Grafiği): `siu_revision/outputs/figures/class_f1_bar_chart.png`
* Grafik-3 (Eğitim Kayıp Eğrisi): `siu_revision/outputs/figures/retrain_loss.png`
