import os
import cv2
import json
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm

class CODACustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        if file_name.startswith("images/"):
            file_name = file_name.replace("images/", "")
        img_path = os.path.join(self.root_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, 512, 512), {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for ann in coco_annotation:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        target = {"image_id": torch.tensor([img_id])}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def load_specialist(weights_path, num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_generalist(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model

# ====================================================================
# FÜZYON STRATEJİLERİ (Inference-time threshold'suz, filtrelemeyi script yapacak)
# ====================================================================

def hybrid_late_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.01):
    with torch.no_grad():
        preds_spec = model_spec([img_tensor])[0]
        preds_gen = model_gen([img_tensor])[0]
    
    s_mask = preds_spec["scores"] >= score_thresh
    spec_boxes = preds_spec["boxes"][s_mask]
    spec_scores = preds_spec["scores"][s_mask]
    spec_labels = preds_spec["labels"][s_mask]
    
    gen_boxes, gen_scores, gen_labels = [], [], []
    for box, score, label in zip(preds_gen["boxes"], preds_gen["scores"], preds_gen["labels"]):
        lid = label.item()
        if lid in coco_to_coda_map and score.item() >= score_thresh:
            gen_boxes.append(box)
            gen_scores.append(score)
            gen_labels.append(coco_to_coda_map[lid])
    
    device = img_tensor.device
    if gen_boxes:
        gen_boxes = torch.stack(gen_boxes)
        gen_scores = torch.tensor([s.item() for s in gen_scores], device=device)
        gen_labels = torch.tensor(gen_labels, device=device)
    else:
        gen_boxes = torch.empty((0, 4), device=device)
        gen_scores = torch.empty(0, device=device)
        gen_labels = torch.empty(0, dtype=torch.int64, device=device)
    
    all_boxes = torch.cat([spec_boxes, gen_boxes])
    all_scores = torch.cat([spec_scores, gen_scores])
    all_labels = torch.cat([spec_labels, gen_labels])
    
    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_labels
    
    keep = torchvision.ops.nms(all_boxes, all_scores, nms_thresh)
    return all_boxes[keep], all_scores[keep], all_labels[keep]


def hybrid_separate_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.01):
    with torch.no_grad():
        preds_spec = model_spec([img_tensor])[0]
        preds_gen = model_gen([img_tensor])[0]
    
    device = img_tensor.device
    
    s_mask = preds_spec["scores"] >= score_thresh
    s_boxes = preds_spec["boxes"][s_mask]
    s_scores = preds_spec["scores"][s_mask]
    s_labels = preds_spec["labels"][s_mask]
    if len(s_boxes) > 0:
        keep_s = torchvision.ops.nms(s_boxes, s_scores, nms_thresh)
        s_boxes, s_scores, s_labels = s_boxes[keep_s], s_scores[keep_s], s_labels[keep_s]
    
    g_boxes, g_scores, g_labels = [], [], []
    for box, score, label in zip(preds_gen["boxes"], preds_gen["scores"], preds_gen["labels"]):
        lid = label.item()
        if lid in coco_to_coda_map and score.item() >= score_thresh:
            g_boxes.append(box)
            g_scores.append(score)
            g_labels.append(coco_to_coda_map[lid])
    
    if g_boxes:
        g_boxes = torch.stack(g_boxes)
        g_scores = torch.tensor([s.item() for s in g_scores], device=device)
        g_labels = torch.tensor(g_labels, device=device)
        keep_g = torchvision.ops.nms(g_boxes, g_scores, nms_thresh)
        g_boxes, g_scores, g_labels = g_boxes[keep_g], g_scores[keep_g], g_labels[keep_g]
    else:
        g_boxes = torch.empty((0, 4), device=device)
        g_scores = torch.empty(0, device=device)
        g_labels = torch.empty(0, dtype=torch.int64, device=device)
    
    return (torch.cat([s_boxes, g_boxes]),
            torch.cat([s_scores, g_scores]),
            torch.cat([s_labels, g_labels]))


def specialist_only(img_tensor, model_spec, score_thresh=0.01):
    with torch.no_grad():
        preds = model_spec([img_tensor])[0]
    mask = preds["scores"] >= score_thresh
    return preds["boxes"][mask], preds["scores"][mask], preds["labels"][mask]


def generalist_only(img_tensor, model_gen, coco_to_coda_map, score_thresh=0.01):
    device = img_tensor.device
    with torch.no_grad():
        preds = model_gen([img_tensor])[0]
    boxes, scores, labels = [], [], []
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        lid = label.item()
        if lid in coco_to_coda_map and score.item() >= score_thresh:
            boxes.append(box)
            scores.append(score)
            labels.append(coco_to_coda_map[lid])
    if boxes:
        return (torch.stack(boxes),
                torch.tensor([s.item() for s in scores], device=device),
                torch.tensor(labels, device=device))
    return (torch.empty((0, 4), device=device),
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.int64, device=device))

# ====================================================================
# METRİK HESAPLAMA MANTIĞI
# ====================================================================

def compute_metrics(dataset, inference_fn, device, iou_thresh=0.5, score_thresh=0.5):
    class_tp = {}
    class_fp = {}
    class_fn = {}
    
    all_scores = []
    all_matches = []

    for i in tqdm(range(len(dataset)), desc="Computing metrics"):
        img_tensor, target = dataset[i]
        boxes, scores, labels = inference_fn(img_tensor.to(device))

        pred_boxes = boxes.cpu()
        pred_scores = scores.cpu()
        pred_labels = labels.cpu()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Score filter
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

        gt_matched = [False] * len(gt_labels)

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
                gt_matched[best_gt_idx] = True
                class_tp[p_label] = class_tp.get(p_label, 0) + 1
                all_scores.append(p_score)
                all_matches.append(1)
            else:
                class_fp[p_label] = class_fp.get(p_label, 0) + 1
                all_scores.append(p_score)
                all_matches.append(0)

        for gi in range(len(gt_labels)):
            if not gt_matched[gi]:
                lbl = gt_labels[gi].item()
                class_fn[lbl] = class_fn.get(lbl, 0) + 1

    all_classes = set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys()))
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = total_tp / max(total_tp + total_fp + total_fn, 1)
    fpr = total_fp / max(total_fp + total_tp, 1)
    miss_rate = total_fn / max(total_fn + total_tp, 1)

    per_class = {}
    CODA_CLASSES = {
        1: "Pedestrian", 2: "Cyclist", 3: "Car", 4: "Truck", 6: "Bus",
        17: "Dog", 18: "Barrier", 23: "Cone", 26: "Traffic Sign",
        27: "Debris", 30: "Concrete Block"
    }

    for c in sorted(all_classes):
        tp = class_tp.get(c, 0)
        fp = class_fp.get(c, 0)
        fn = class_fn.get(c, 0)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-6)
        cname = CODA_CLASSES.get(c, f"ID:{c}")
        per_class[cname] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(p * 100, 2),
            "Recall": round(r * 100, 2),
            "F1": round(f * 100, 2)
        }

    return {
        "overall": {
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1": round(f1 * 100, 2),
            "Accuracy": round(accuracy * 100, 2),
            "FPR": round(fpr * 100, 2),
            "MissRate": round(miss_rate * 100, 2)
        },
        "per_class": per_class,
        "pr_curve_data": {
            "scores": [float(s) for s in all_scores],
            "matches": [int(m) for m in all_matches]
        }
    }

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIR = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
    TEST_ANN = "/workspace/siu_revision/data/coda_split/test_annotations.json"
    RESULTS_DIR = "/workspace/siu_revision/outputs/results"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    COCO_TO_CODA_MAP = {3: 3, 8: 3, 6: 3, 1: 1}
    
    test_dataset = CODACustomDataset(root_dir=IMG_DIR, annotation_file=TEST_ANN)
    
    # Modelleri yükle
    model_spec = load_specialist("/workspace/siu_revision/weights/specialist_retrained.pth", 36, DEVICE)
    model_gen = load_generalist(DEVICE)
    
    configs = {
        "specialist_only": lambda img: specialist_only(img, model_spec),
        "generalist_only": lambda img: generalist_only(img, model_gen, COCO_TO_CODA_MAP),
        "hybrid_late_nms": lambda img: hybrid_late_nms(img, model_spec, model_gen, COCO_TO_CODA_MAP),
        "hybrid_separate_nms": lambda img: hybrid_separate_nms(img, model_spec, model_gen, COCO_TO_CODA_MAP)
    }
    
    extra_metrics = {}
    
    for name, fn in configs.items():
        print(f"\nEvaluating extra metrics for: {name}")
        res = compute_metrics(test_dataset, fn, DEVICE, iou_thresh=0.5, score_thresh=0.5)
        extra_metrics[name] = res
        
        o = res["overall"]
        print(f"  Precision: {o['Precision']}% | Recall: {o['Recall']}% | F1: {o['F1']}% | FPR: {o['FPR']}%")

    out_path = f"{RESULTS_DIR}/extra_metrics.json"
    with open(out_path, "w") as f:
        json.dump(extra_metrics, f, indent=4)
    print(f"\nExtra metrics saved to: {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("KLASİK DETEKSİYON METRİKLERİ ÖZETİ (IoU=0.5, score_thresh=0.5)")
    print(f"{'='*80}")
    header = f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'FPR':>10} {'Miss Rate':>10}"
    print(header)
    print("-" * len(header))
    for name, data in extra_metrics.items():
        o = data["overall"]
        print(f"{name:<25} {o['Precision']:>9.2f}% {o['Recall']:>9.2f}% {o['F1']:>9.2f}% {o['FPR']:>9.2f}% {o['MissRate']:>9.2f}%")

if __name__ == "__main__":
    main()
