# SIU 2026 Revizyon - 1500 Sahne (Orijinal Değerlendirme / Veri Sızıntılı) Sonuçları Raporu

Bu rapor, makaledeki orijinal deneylerin (veri sızıntılı ve 1500 sahne üzerinden olan) replikasyon ve doğrulama sonuçlarını içermektedir.

## 🎯 Doğrulama Durumu: **MODEL DOĞRULANDI**

Makalede hedeflenen ve yeni eğittiğimiz model ile elde edilen hibrit inference (Late Fusion) sonuçları karşılaştırması:

| Metrik | Makaledeki Hedef Değer | Replikasyon Sonucu (Bizim Elde Ettiğimiz) | Durum |
| :--- | :---: | :---: | :---: |
| **mAP@50-95** | ≈ 0.364 | **0.3667** (+0.27% fark) | **UYUMLU** |
| **AR@100** | ≈ 0.416 | **0.4194** (+0.34% fark) | **UYUMLU** |

Elde edilen fark tolerans sınırları (±0.015) içerisinde olduğundan model eğitimi ve hibrit çıkarım kodunun doğruluğu **resmen kanıtlanmıştır**.

---

## 1. 1500 Sahne Üzerinde Genel Model Karşılaştırma Sonuçları

Aşağıdaki tablo, 1500 sahnenin tamamı üzerinde (skor eşiği = 0.5, IoU eşiği = 0.5) elde edilen tüm model konfigürasyonlarının sonuçlarını göstermektedir:

| Model Konfigürasyonu | mAP (%) | mAP50 (%) | AR@100 (%) | Precision (%) | Recall (%) | F1-Skor (%) | FPR (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tekil Uzman (Specialist Only)** * | **51.46** | **64.54** | **55.44** | **93.85** | **97.80** | **95.78** | **6.15** | **48.23** |
| **Tekil Genelci (Generalist Only)** | 0.03 | 0.07 | 2.46 | 0.57 | 1.68 | 0.85 | 99.43 | 46.19 |
| **Hibrit (Late Fusion - Önerilen)** | **36.67** | **43.66** | **41.94** | **24.89** | **97.30** | **39.64** | **75.11** | **25.38** |
| **Hibrit (Separate NMS - Alternatif)** | 37.71 | 44.85 | 43.51 | 24.67 | 97.55 | 39.38 | 75.33 | 25.23 |

*\* Not: Tekil Uzman modelin bu kadar yüksek sonuçlar almasının sebebi, tüm 1500 resim üzerinde eğitilmesi ve testin de yine aynı veri (veri sızıntısı) üzerinde yapılmasıdır.*

---

## 2. Gecikme ve Çıkarım Hızı Dağılımı

Tüm ölçümler Nvidia GeForce RTX 4090 GPU üzerinde yapılmıştır:

* **Tekil Uzman (Specialist Only)**: Ortalama Gecikme = **20.73 ms ± 1.53 ms** | FPS = **48.23**
* **Tekil Genelci (Generalist Only)**: Ortalama Gecikme = **21.65 ms ± 1.57 ms** | FPS = **46.19**
* **Hibrit (Late Fusion - Önerilen)**: Ortalama Gecikme = **39.40 ms ± 2.32 ms** | FPS = **25.38**
* **Hibrit (Separate NMS - Alternatif)**: Ortalama Gecikme = **39.63 ms ± 2.00 ms** | FPS = **25.23**

---

## 3. LaTeX Tablo Kodları

### LaTeX Kod (Tablo I - 1500 Sahne Model Karşılaştırma)
```latex
\begin{table}[htbp]
\caption{Orijinal Notebook Değerlendirme Protokolü ile 1500 Sahne Üzerinde Karşılaştırma Sonuçları}
\label{tab:1500_model_comparison}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Model Konfigürasyonu} & \textbf{mAP (\%)} & \textbf{mAP50 (\%)} & \textbf{AR@100 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Skor (\%)} & \textbf{FPR (\%)} \\ \hline
Tekil Uzman (Baseline) & 51.5 & 64.5 & 55.4 & 93.9 & 97.8 & 95.8 & 6.2 \\
Tekil Genelci & 0.0 & 0.1 & 2.5 & 0.6 & 1.7 & 0.9 & 99.4 \\
\textbf{Hibrit (Late Fusion - Önerilen)} & \textbf{36.7} & \textbf{43.7} & \textbf{41.9} & \textbf{24.9} & \textbf{97.3} & \textbf{39.6} & \textbf{75.1} \\
Hibrit (Separate NMS - Alternatif) & 37.7 & 44.9 & 43.5 & 24.7 & 97.6 & 39.4 & 75.3 \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Kaydedilen Sonuç Dosyaları (Local & Remote)

Tüm detaylı sonuçlar local workspace ve remote sunucu üzerinde başarıyla kaydedilmiş ve indirilmştir:
- **Genel Özet JSON**: `siu_revision/outputs/results/evaluation_summary.json`
- **Sınıf Bazlı Detaylı Metrikler**: `siu_revision/outputs/results/extra_metrics.json`
- **Sınıf Bazlı Recall Karşılaştırması**: `siu_revision/outputs/results/class_wise_recall_comparison.json`
- **Çıkarım Hızı / Latency JSON**: `siu_revision/outputs/results/inference_speed.json`
