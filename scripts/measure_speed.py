import os
import cv2
import json
import time
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
# FÜZYON STRATEJİLERİ
# ====================================================================

def hybrid_late_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.5):
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


def hybrid_separate_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.5):
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


def specialist_only(img_tensor, model_spec, score_thresh=0.5):
    with torch.no_grad():
        preds = model_spec([img_tensor])[0]
    mask = preds["scores"] >= score_thresh
    return preds["boxes"][mask], preds["scores"][mask], preds["labels"][mask]


def generalist_only(img_tensor, model_gen, coco_to_coda_map, score_thresh=0.5):
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
# SPEED PROFILER
# ====================================================================

def measure_speed(dataset, inference_fn, device, num_warmup=10, num_measure=100):
    times = []

    for i in range(min(num_warmup + num_measure, len(dataset))):
        img_tensor, _ = dataset[i]
        img_tensor = img_tensor.to(device)

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
        "avg_latency_ms": float(avg_time * 1000),
        "std_latency_ms": float(std_time * 1000),
        "fps": float(fps)
    }

def main():
    print("=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

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
    
    speed_results = {}
    
    for name, fn in configs.items():
        print(f"\nBenchmarking: {name}")
        res = measure_speed(test_dataset, fn, DEVICE, num_warmup=10, num_measure=100)
        speed_results[name] = res
        print(f"  Latency: {res['avg_latency_ms']:.2f} ms ± {res['std_latency_ms']:.2f} ms | FPS: {res['fps']:.2f}")

    out_path = f"{RESULTS_DIR}/inference_speed.json"
    with open(out_path, "w") as f:
        json.dump(speed_results, f, indent=4)
    print(f"\nSpeed benchmark saved to: {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("ÇIKARIM HIZI VE GECİKME ÖZETİ")
    print(f"{'='*60}")
    header = f"{'Model':<25} {'Ortalama Gecikme (ms)':>22} {'Hız (FPS)':>10}"
    print(header)
    print("-" * len(header))
    for name, data in speed_results.items():
        print(f"{name:<25} {data['avg_latency_ms']:>18.2f} ms {data['fps']:>9.2f}")

if __name__ == "__main__":
    main()
