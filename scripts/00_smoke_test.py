"""
SMOKE TEST: Model ağırlıklarını doğrula.
Beklenen: Uzman model AR ≈ %12.8, Hibrit model AR ≈ %41.6
Eğer bu değerlerden büyük sapma varsa → model yanlış, yeniden eğitim gerekli.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.dataset import CODACustomDataset
from src.models import load_specialist, load_generalist
from src.fusion import fusion_late_nms, fusion_specialist_only
from src.evaluation import generate_coco_results, evaluate_coco_metrics

def main():
    print("=" * 60)
    print("SMOKE TEST — Model Doğrulama")
    print("=" * 60)

    # 1. Veri seti yükle
    dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(ANN_FILE))
    print(f"Veri seti yüklendi: {len(dataset)} sahne")

    # 2. Modelleri yükle
    print("Uzman model yükleniyor...")
    model_spec = load_specialist(str(SPECIALIST_WEIGHTS), NUM_CLASSES_CODA, DEVICE)
    print("Genelci model yükleniyor...")
    model_gen = load_generalist(DEVICE)

    # 3. Hızlı test: 5 görüntü üzerinde
    print("\n--- Uzman Model (Baseline) Testi ---")
    # İlk 5 görüntüde çıktı kontrol et
    for i in range(min(5, len(dataset))):
        img, target = dataset[i]
        boxes, scores, labels = fusion_specialist_only(model_spec, img.to(DEVICE))
        n_det = (scores >= 0.5).sum().item()
        n_gt = len(target["labels"])
        print(f"  Görüntü {i}: GT={n_gt} nesne, Tespit={n_det} nesne")

    print("\n--- Hibrit Model Testi ---")
    for i in range(min(5, len(dataset))):
        img, target = dataset[i]
        boxes, scores, labels = fusion_late_nms(
            model_spec, model_gen, img.to(DEVICE),
            COCO_TO_CODA_MAP, NMS_IOU_THRESH, SCORE_THRESH_EVAL
        )
        n_det = (scores >= 0.5).sum().item()
        n_gt = len(target["labels"])
        print(f"  Görüntü {i}: GT={n_gt} nesne, Tespit={n_det} nesne")

    # 4. Tam COCO eval — sadece uzman model (hızlı doğrulama)
    print("\n--- COCO Eval: Uzman Model (tüm veri seti) ---")
    spec_fn = lambda img: fusion_specialist_only(model_spec, img.to(DEVICE))
    
    # Save smoke test json in outputs/results/
    smoke_json = OUTPUT_DIR / "results" / "smoke_specialist.json"
    generate_coco_results(dataset, spec_fn, str(ANN_FILE), str(smoke_json))
    spec_metrics = evaluate_coco_metrics(str(ANN_FILE), str(smoke_json))

    print(f"\nBEKLENEN AR@100 ≈ 12.8%  →  ÖLÇÜLEN: {spec_metrics['AR@100']*100:.1f}%")
    print(f"BEKLENEN mAP ≈ 9.5%     →  ÖLÇÜLEN: {spec_metrics['mAP@[0.5:0.95]']*100:.1f}%")

    diff_ar = abs(spec_metrics["AR@100"] * 100 - 12.8)
    if diff_ar > 5.0:
        print(f"\n⚠️  UYARI: AR farkı {diff_ar:.1f} puan — model ağırlıkları yanlış olabilir!")
        print("   → Yeniden eğitim gerekli olabilir.")
    else:
        print(f"\n✅ Model doğrulandı. AR farkı {diff_ar:.1f} puan (kabul edilebilir).")
        print("   → Tam değerlendirmeye geçilebilir.")

if __name__ == "__main__":
    main()
