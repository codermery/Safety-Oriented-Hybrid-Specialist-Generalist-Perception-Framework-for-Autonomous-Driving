"""evaluate_combined.py — Birleşik modeli CODA 1500 sahne üzerinde değerlendir."""
import os
import sys
import json
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Add project root to path
PROJECT_ROOT = Path("/workspace/siu_revision")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CODACustomDataset

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIR = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
    ANN_FILE = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json"
    MODEL_PATH = "/workspace/siu_revision/weights/combined_model.pth"
    OUTPUT_JSON = "/workspace/siu_revision/outputs/results/combined_model_predictions.json"
    
    os.makedirs("/workspace/siu_revision/outputs/results", exist_ok=True)
    
    print(f"Loading validation dataset from: {IMG_DIR}")
    # Dataset (CODA — test verisi)
    dataset = CODACustomDataset(root_dir=IMG_DIR, annotation_file=ANN_FILE)
    print(f"Dataset loaded: {len(dataset)} scenes.")
    
    # Model yükle
    print(f"Loading model weights from: {MODEL_PATH}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 36)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Inference
    print("Running inference on CODA scenes...")
    results = []
    for i in tqdm(range(len(dataset))):
        img, target = dataset[i]
        img_id = target["image_id"].item()
        
        # Skip if dummy image (missing file)
        if img.sum() < 0.1:
            continue
            
        with torch.no_grad():
            preds = model([img.to(DEVICE)])[0]
        
        for box, score, label in zip(preds["boxes"].cpu(), preds["scores"].cpu(), preds["labels"].cpu()):
            if score.item() >= 0.05:  # COCOeval requires low threshold (0.05)
                x1, y1, x2, y2 = box.numpy()
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(label.item()),
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(score.item()),
                })
    
    # Save predictions
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)
    print(f"Predictions saved to {OUTPUT_JSON}")
    
    # COCOeval
    print("Running COCOeval...")
    coco_gt = COCO(ANN_FILE)
    if len(results) > 0:
        coco_dt = coco_gt.loadRes(OUTPUT_JSON)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            "mAP@[0.5:0.95]": round(coco_eval.stats[0] * 100, 2),
            "mAP@0.50": round(coco_eval.stats[1] * 100, 2),
            "mAP@0.75": round(coco_eval.stats[2] * 100, 2),
            "AR@100": round(coco_eval.stats[8] * 100, 2),
            "AP_small": round(coco_eval.stats[3] * 100, 2),
            "AP_medium": round(coco_eval.stats[4] * 100, 2),
            "AP_large": round(coco_eval.stats[5] * 100, 2),
        }
    else:
        print("Warning: No detections generated.")
        metrics = {
            "mAP@[0.5:0.95]": 0.0,
            "mAP@0.50": 0.0,
            "mAP@0.75": 0.0,
            "AR@100": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }
    
    metrics_out_path = "/workspace/siu_revision/outputs/results/combined_model_metrics.json"
    with open(metrics_out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_out_path}")
    
    print("\n=== BİRLEŞİK MODEL SONUÇLARI (CODA 1500 sahne) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}%")
        
if __name__ == "__main__":
    evaluate()
