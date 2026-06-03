import json
import time
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

# ====================================================================
# 1. COCO-STYLE METRİK (mAP, AR — makaledeki mevcut metrikler)
# ====================================================================
def generate_coco_results(dataset, inference_fn, coco_gt_json, output_json):
    """
    Herhangi bir inference fonksiyonunun çıktılarını COCO format JSON'a yaz.
    inference_fn(image_tensor) -> (boxes, scores, labels)
    """
    coco_gt = COCO(coco_gt_json)
    results = []

    for i in tqdm(range(len(dataset)), desc="Generating COCO results"):
        img_tensor, target = dataset[i]
        img_id = target["image_id"].item()
        
        boxes, scores, labels = inference_fn(img_tensor)

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for box, score, label in zip(boxes_np, scores_np, labels_np):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(results, f)
    return output_json


def evaluate_coco_metrics(gt_json, pred_json):
    """COCO API ile mAP, AR hesapla."""
    coco_gt = COCO(gt_json)
    if not os.path.exists(pred_json) or os.path.getsize(pred_json) == 0:
        return {
            "mAP@[0.5:0.95]": 0.0, "mAP@0.50": 0.0, "mAP@0.75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
            "AR@1": 0.0, "AR@10": 0.0, "AR@100": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
        }
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP@[0.5:0.95]": coco_eval.stats[0],
        "mAP@0.50": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

import os

# ====================================================================
# 2. EK METRİKLER (Reviewer 1 isteği: Accuracy, F1, FPR, Precision-Recall)
# ====================================================================
def compute_detection_metrics(dataset, inference_fn, device,
                              iou_thresh=0.5, score_thresh=0.5):
    """
    Her görüntü için TP, FP, FN sayarak:
    - Precision, Recall, F1-Score
    - Accuracy (TP / (TP + FP + FN))
    - False Positive Rate (FP / (FP + TN)) — detection'da TN tanımsız
      olduğu için bunu FP / toplam_tespit olarak raporla
    - Sınıf bazlı breakdown
    Tüm 1500 sahne üzerinde çalıştır.
    """
    class_tp = {}
    class_fp = {}
    class_fn = {}
    all_scores = []  # PR eğrisi için
    all_matches = []  # PR eğrisi için

    for i in tqdm(range(len(dataset)), desc="Computing detection metrics"):
        img_tensor, target = dataset[i]
        boxes, scores, labels = inference_fn(img_tensor)

        pred_boxes = boxes.cpu()
        pred_scores = scores.cpu()
        pred_labels = labels.cpu()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Score filtresi
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

        # GT sınıflarını kaydet (FN hesabı için)
        gt_matched = [False] * len(gt_labels)

        # Her tahmin için en iyi GT eşleşmesi bul
        for pi in range(len(pred_boxes)):
            p_label = pred_labels[pi].item()
            p_score = pred_scores[pi].item()
            best_iou = 0
            best_gt_idx = -1

            for gi in range(len(gt_boxes)):
                if gt_labels[gi].item() != p_label:
                    continue
                if gt_matched[gi]:
                    continue
                iou = box_iou(
                    pred_boxes[pi].unsqueeze(0), gt_boxes[gi].unsqueeze(0)
                ).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                # True Positive
                gt_matched[best_gt_idx] = True
                class_tp[p_label] = class_tp.get(p_label, 0) + 1
                all_scores.append(p_score)
                all_matches.append(1)
            else:
                # False Positive
                class_fp[p_label] = class_fp.get(p_label, 0) + 1
                all_scores.append(p_score)
                all_matches.append(0)

        # Eşleşmeyen GT'ler = False Negative
        for gi in range(len(gt_labels)):
            if not gt_matched[gi]:
                lbl = gt_labels[gi].item()
                class_fn[lbl] = class_fn.get(lbl, 0) + 1

    # Genel metrikler
    all_classes = set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys()))
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = total_tp / max(total_tp + total_fp + total_fn, 1)
    fpr = total_fp / max(total_fp + total_tp, 1)  # FP oranı
    miss_rate = total_fn / max(total_fn + total_tp, 1)  # Kaçırma oranı

    # Sınıf bazlı
    per_class = {}
    for c in sorted(all_classes):
        tp = class_tp.get(c, 0)
        fp = class_fp.get(c, 0)
        fn = class_fn.get(c, 0)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-6)
        per_class[c] = {"TP": tp, "FP": fp, "FN": fn,
                        "Precision": p, "Recall": r, "F1": f}

    return {
        "overall": {
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": precision, "Recall": recall, "F1": f1,
            "Accuracy": accuracy, "FPR": fpr, "MissRate": miss_rate,
        },
        "per_class": per_class,
        "pr_curve_data": {
            "scores": [float(s) for s in all_scores],
            "matches": [int(m) for m in all_matches],
        },
    }


# ====================================================================
# 3. INFERENCE HIZI (Reviewer 3 isteği: FPS)
# ====================================================================
def measure_inference_speed(dataset, inference_fn, device, 
                            num_warmup=10, num_measure=100):
    """GPU warmup sonrası ortalama inference süresi ve FPS ölç."""
    times = []

    for i in range(min(num_warmup + num_measure, len(dataset))):
        img_tensor, _ = dataset[i]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = inference_fn(img_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if i >= num_warmup:
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    return {
        "avg_inference_ms": float(avg_time * 1000),
        "std_inference_ms": float(std_time * 1000),
        "fps": float(fps),
        "num_samples": len(times),
    }
