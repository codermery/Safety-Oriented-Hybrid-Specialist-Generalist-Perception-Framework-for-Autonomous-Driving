import os
import sys
import json
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/workspace/siu_revision")
sys.path.insert(0, str(PROJECT_ROOT))

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
    compute_detection_metrics,
    measure_inference_speed
)

def main():
    print("=" * 60)
    # Use absolute paths on the remote server
    img_dir = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
    ann_file = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json"
    weights_path = "/workspace/siu_revision/weights/coda_fresh_model.pth"
    output_dir = Path("/workspace/siu_revision/outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {img_dir}")
    print(f"Annotation file: {ann_file}")
    print(f"Specialist model weights: {weights_path}")
    print(f"Results output folder: {output_dir}")
    print("=" * 60)

    # 1. Load dataset
    dataset = CODACustomDataset(root_dir=img_dir, annotation_file=ann_file)
    print(f"Dataset loaded: {len(dataset)} scenes.")

    # 2. Load models
    print("Loading models onto device:", DEVICE)
    model_spec = load_specialist(weights_path, NUM_CLASSES_CODA, DEVICE)
    model_gen = load_generalist(DEVICE)

    # Define the 4 configurations to evaluate
    configs = {
        "hybrid_late_nms": {
            "fn": lambda img: fusion_late_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_EVAL),
            "name": "Hybrid (Late Fusion)"
        },
        "specialist_only": {
            "fn": lambda img: fusion_specialist_only(model_spec, img.to(DEVICE), score_thresh=SCORE_THRESH_EVAL),
            "name": "Specialist Only (Baseline)"
        },
        "generalist_only": {
            "fn": lambda img: fusion_generalist_only(model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, score_thresh=SCORE_THRESH_EVAL),
            "name": "Generalist Only"
        },
        "hybrid_separate_nms": {
            "fn": lambda img: fusion_separate_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_EVAL),
            "name": "Hybrid (Separate NMS)"
        }
    }

    results_summary = {}
    extra_metrics = {}

    # 3. Evaluate Hybrid Late NMS (Proposed) First for Verification
    print("\n--- STEP 1: Verifying Hybrid Late NMS Model ---")
    hybrid_key = "hybrid_late_nms"
    hybrid_info = configs[hybrid_key]
    hybrid_pred_json = str(output_dir / f"{hybrid_key}_predictions.json")
    
    print("Generating COCO results for Hybrid Late NMS...")
    generate_coco_results(dataset, hybrid_info["fn"], ann_file, hybrid_pred_json)
    
    print("Running COCOeval...")
    coco_metrics = evaluate_coco_metrics(ann_file, hybrid_pred_json)
    
    ar_100 = coco_metrics["AR@100"]
    map_50_95 = coco_metrics["mAP@[0.5:0.95]"]
    print(f"\nHybrid Late NMS Results:")
    print(f"  mAP@[0.5:0.95] : {map_50_95:.4f}")
    print(f"  AR@100         : {ar_100:.4f}")
    
    # Check verification criteria: AR@100 ≈ 0.416
    ar_target = 0.416
    map_target = 0.364
    verified = False
    
    # Tolerans limit: 0.015
    if abs(ar_100 - ar_target) <= 0.015:
        print("\n" + "*" * 50)
        print("MODEL DOĞRULANDI")
        print("*" * 50 + "\n")
        verified = True
    else:
        print("\n" + "!" * 50)
        print(f"MODEL DOGRULANAMADI: Beklenen AR@100 ≈ {ar_target}, Elde Edilen = {ar_100:.4f}")
        print("!" * 50 + "\n")

    results_summary[hybrid_key] = coco_metrics

    # 4. Proceed with all other evaluations
    print("\n--- STEP 2: Evaluating other configurations ---")
    for key, info in configs.items():
        if key == hybrid_key:
            continue
        print(f"\nEvaluating: {info['name']}")
        pred_json = str(output_dir / f"{key}_predictions.json")
        generate_coco_results(dataset, info["fn"], ann_file, pred_json)
        coco_metrics = evaluate_coco_metrics(ann_file, pred_json)
        results_summary[key] = coco_metrics

    # 5. Compute extra metrics (Precision, Recall, F1, Accuracy, FPR at score_thresh=0.5)
    print("\n--- STEP 3: Computing extra metrics (IoU=0.5, score_thresh=0.5) ---")
    # Define custom inference functions for the extra metrics that use score_thresh=0.5
    configs_extra = {
        "hybrid_late_nms": lambda img: fusion_late_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=0.5),
        "specialist_only": lambda img: fusion_specialist_only(model_spec, img.to(DEVICE), score_thresh=0.5),
        "generalist_only": lambda img: fusion_generalist_only(model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, score_thresh=0.5),
        "hybrid_separate_nms": lambda img: fusion_separate_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=0.5)
    }

    for key, fn in configs_extra.items():
        print(f"Computing classic metrics for {key}...")
        metrics = compute_detection_metrics(dataset, fn, DEVICE, iou_thresh=0.5, score_thresh=0.5)
        extra_metrics[key] = metrics

    # Save extra_metrics to JSON
    with open(output_dir / "extra_metrics.json", "w") as f:
        json.dump(extra_metrics, f, indent=4)

    # 6. Class-wise Recall Comparison
    print("\n--- STEP 4: Creating Class-wise Recall Comparison ---")
    class_recall_comparison = {}
    
    # Get all class names from the results
    sample_key = list(extra_metrics.keys())[0]
    classes = list(extra_metrics[sample_key]["per_class"].keys())
    
    for c in classes:
        # Resolve class names
        class_recall_comparison[c] = {}
        for key in extra_metrics.keys():
            r = extra_metrics[key]["per_class"][c]["Recall"]
            class_recall_comparison[c][key] = r

    with open(output_dir / "class_wise_recall_comparison.json", "w") as f:
        json.dump(class_recall_comparison, f, indent=4)

    # 7. Speed/FPS Profiling on GPU
    print("\n--- STEP 5: Measuring Inference Speed/FPS ---")
    speed_results = {}
    # Warmup and profile on first 150 images
    for key, info in configs.items():
        print(f"Measuring speed for {info['name']}...")
        res = measure_inference_speed(dataset, info["fn"], DEVICE, num_warmup=15, num_measure=150)
        speed_results[key] = {
            "model_name": info["name"],
            "avg_latency_ms": res["avg_inference_ms"],
            "std_latency_ms": res["std_inference_ms"],
            "fps": res["fps"]
        }
        print(f"  Avg Latency: {res['avg_inference_ms']:.2f} ms | FPS: {res['fps']:.2f}")

    with open(output_dir / "inference_speed.json", "w") as f:
        json.dump(speed_results, f, indent=4)

    # 8. Create combined summary
    combined_summary = {}
    for key in configs.keys():
        coco = results_summary[key]
        extra = extra_metrics[key]["overall"]
        speed = speed_results[key]
        
        combined_summary[key] = {
            "model_name": configs[key]["name"],
            "mAP_50_95": coco["mAP@[0.5:0.95]"],
            "mAP_50": coco["mAP@0.50"],
            "mAP_75": coco["mAP@0.75"],
            "AR_100": coco["AR@100"],
            "Precision": extra["Precision"],
            "Recall": extra["Recall"],
            "F1": extra["F1"],
            "Accuracy": extra["Accuracy"],
            "FPR": extra["FPR"],
            "MissRate": extra["MissRate"],
            "Avg_Latency_ms": speed["avg_latency_ms"],
            "FPS": speed["fps"]
        }

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(combined_summary, f, indent=4)

    # Print clean summary table
    print("\n" + "=" * 100)
    print("FINAL PERFORMANCE SUMMARY TABLE (1500 SCENES)")
    print("=" * 100)
    print(f"{'Model':<30} | {'mAP@50':<8} | {'AR@100':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6} | {'FPR':<6} | {'FPS':<6}")
    print("-" * 100)
    for key, val in combined_summary.items():
        print(f"{val['model_name']:<30} | {val['mAP_50']*100:>7.2f}% | {val['AR_100']*100:>7.2f}% | {val['Precision']:>9.2f}% | {val['Recall']:>7.2f}% | {val['F1']:>5.2f}% | {val['FPR']:>5.2f}% | {val['FPS']:>5.2f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
