"""
TÜM MODEL DEĞERLENDİRMESİ
4 model konfigürasyonu × tüm metrikler = makaleye hazır sonuçlar

Modeller:
  1. Tekil Uzman (Baseline) — sadece CODA eğitimli
  2. Tekil Genelci — sadece COCO pretrained
  3. Hibrit (Önerilen) — Late Fusion + NMS
  4. Hibrit Alternatif — Separate NMS + Merge (Reviewer 2)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.dataset import CODACustomDataset
from src.models import load_specialist, load_generalist
from src.fusion import (
    fusion_late_nms, 
    fusion_separate_nms, 
    fusion_specialist_only, 
    fusion_generalist_only
)
from src.evaluation import (
    generate_coco_results, 
    evaluate_coco_metrics, 
    compute_detection_metrics
)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")

def main():
    print("=" * 60)
    print("TÜM MODELLERİN DEĞERLENDİRİLMESİ")
    print("=" * 60)

    # 1. Veri seti yükle
    dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(ANN_FILE))
    print(f"Veri seti yüklendi: {len(dataset)} sahne")

    # 2. Modelleri yükle
    print("Modeller yükleniyor...")
    model_spec = load_specialist(str(SPECIALIST_WEIGHTS), NUM_CLASSES_CODA, DEVICE)
    model_gen = load_generalist(DEVICE)

    # 3. Model inference tanımlamaları (SCORE_THRESH_EVAL = 0.05)
    models_to_eval = {
        "specialist": {
            "fn": lambda img: fusion_specialist_only(model_spec, img.to(DEVICE), score_thresh=SCORE_THRESH_EVAL),
            "name": "Tekil Uzman (Baseline)"
        },
        "generalist": {
            "fn": lambda img: fusion_generalist_only(model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, score_thresh=SCORE_THRESH_EVAL),
            "name": "Tekil Genelci"
        },
        "hybrid_late_nms": {
            "fn": lambda img: fusion_late_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_EVAL),
            "name": "Hibrit (Late Fusion - Önerilen)"
        },
        "hybrid_separate_nms": {
            "fn": lambda img: fusion_separate_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_EVAL),
            "name": "Hibrit (Separate NMS - Alternatif)"
        }
    }

    # 4. Her modeli değerlendir
    results_summary = {}

    for key, model_info in models_to_eval.items():
        print(f"\n--- Değerlendiriliyor: {model_info['name']} ---")
        
        # Output paths
        coco_pred_json = OUTPUT_DIR / "results" / f"{key}_coco_predictions.json"
        metrics_json = OUTPUT_DIR / "results" / f"{key}_metrics.json"

        # COCO Eval (mAP, AR)
        print("COCO-style sonuçlar üretiliyor...")
        generate_coco_results(dataset, model_info["fn"], str(ANN_FILE), str(coco_pred_json))
        coco_metrics = evaluate_coco_metrics(str(ANN_FILE), str(coco_pred_json))

        # Ek Metrikler (Precision, Recall, F1, Accuracy, FPR, Miss Rate)
        print("Klasik metrikler hesaplanıyor...")
        det_metrics = compute_detection_metrics(
            dataset, model_info["fn"], DEVICE, 
            iou_thresh=0.5, score_thresh=SCORE_THRESH_INFERENCE
        )

        # Sonuçları birleştir
        combined_results = {
            "model_name": model_info["name"],
            "coco_metrics": coco_metrics,
            "detection_metrics": det_metrics
        }
        
        save_json(combined_results, metrics_json)
        
        # Summary for comparison
        results_summary[key] = {
            "mAP_50_95": coco_metrics["mAP@[0.5:0.95]"],
            "mAP_50": coco_metrics["mAP@0.50"],
            "mAP_75": coco_metrics["mAP@0.75"],
            "AR_100": coco_metrics["AR@100"],
            "precision": det_metrics["overall"]["Precision"],
            "recall": det_metrics["overall"]["Recall"],
            "f1": det_metrics["overall"]["F1"],
            "accuracy": det_metrics["overall"]["Accuracy"],
            "fpr": det_metrics["overall"]["FPR"],
            "miss_rate": det_metrics["overall"]["MissRate"]
        }

    # 5. Sonuçları özetle ve LaTeX formatında kaydet
    summary_path = OUTPUT_DIR / "results" / "evaluation_summary.json"
    save_json(results_summary, summary_path)

    print("\n" + "=" * 60)
    print("ÖZET PERFORMANS TABLOSU")
    print("=" * 60)
    print(f"{'Model':<35} | {'mAP@50':<8} | {'AR@100':<8} | {'F1-Score':<8} | {'FPR':<8}")
    print("-" * 75)
    for key, s in results_summary.items():
        name = models_to_eval[key]["name"]
        print(f"{name:<35} | {s['mAP_50']*100:>7.2f}% | {s['AR_100']*100:>7.2f}% | {s['f1']*100:>7.2f}% | {s['fpr']*100:>7.2f}%")

if __name__ == "__main__":
    main()
