"""
ALTERNATİF FÜZYON DENEYLERİ (Reviewer 2 ve Meta-Reviewer İsteği)

Varyasyonlar:
  1. Strateji A — Late Fusion + NMS (NMS IoU Thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
  2. Strateji B — Separate NMS + Merge (NMS IoU Thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.dataset import CODACustomDataset
from src.models import load_specialist, load_generalist
from src.fusion import fusion_late_nms, fusion_separate_nms
from src.evaluation import generate_coco_results, evaluate_coco_metrics

def main():
    print("=" * 60)
    print("ALTERNATİF FÜZYON DENEYLERİ")
    print("=" * 60)

    # 1. Veri seti yükle
    dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(ANN_FILE))
    print(f"Veri seti: {len(dataset)} sahne")

    # 2. Modelleri yükle
    model_spec = load_specialist(str(SPECIALIST_WEIGHTS), NUM_CLASSES_CODA, DEVICE)
    model_gen = load_generalist(DEVICE)

    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    fusion_results = {
        "late_nms": {},
        "separate_nms": {}
    }

    # Evaluate Late Fusion
    for iou in iou_thresholds:
        print(f"\nEvaluating Late NMS with IoU threshold: {iou}")
        fn_late = lambda img: fusion_late_nms(
            model_spec, model_gen, img.to(DEVICE), 
            COCO_TO_CODA_MAP, nms_thresh=iou, score_thresh=SCORE_THRESH_EVAL
        )
        
        pred_json = OUTPUT_DIR / "results" / f"late_nms_iou_{iou}_predictions.json"
        generate_coco_results(dataset, fn_late, str(ANN_FILE), str(pred_json))
        metrics = evaluate_coco_metrics(str(ANN_FILE), str(pred_json))
        
        fusion_results["late_nms"][str(iou)] = {
            "mAP_50": float(metrics["mAP@0.50"]),
            "mAP_75": float(metrics["mAP@0.75"]),
            "mAP_50_95": float(metrics["mAP@[0.5:0.95]"]),
            "AR_100": float(metrics["AR@100"])
        }

    # Evaluate Separate NMS
    for iou in iou_thresholds:
        print(f"\nEvaluating Separate NMS with IoU threshold: {iou}")
        fn_sep = lambda img: fusion_separate_nms(
            model_spec, model_gen, img.to(DEVICE), 
            COCO_TO_CODA_MAP, nms_thresh=iou, score_thresh=SCORE_THRESH_EVAL
        )
        
        pred_json = OUTPUT_DIR / "results" / f"separate_nms_iou_{iou}_predictions.json"
        generate_coco_results(dataset, fn_sep, str(ANN_FILE), str(pred_json))
        metrics = evaluate_coco_metrics(str(ANN_FILE), str(pred_json))
        
        fusion_results["separate_nms"][str(iou)] = {
            "mAP_50": float(metrics["mAP@0.50"]),
            "mAP_75": float(metrics["mAP@0.75"]),
            "mAP_50_95": float(metrics["mAP@[0.5:0.95]"]),
            "AR_100": float(metrics["AR@100"])
        }

    # Save results
    save_path = OUTPUT_DIR / "results" / "alternative_fusion_results.json"
    with open(save_path, "w") as f:
        json.dump(fusion_results, f, indent=4)
    print(f"\nFusion experiment results saved to {save_path}")

if __name__ == "__main__":
    main()
