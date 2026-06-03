# SOTA Karşılaştırma (CODA Benchmark)

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
