import os
import cv2
import json
import torch
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
# FÜZYON STRATEJİLERİ
# ====================================================================

def hybrid_late_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.05):
    """İki modelin ham çıktılarını birleştir, sonra TEK NMS uygula."""
    with torch.no_grad():
        preds_spec = model_spec([img_tensor])[0]
        preds_gen = model_gen([img_tensor])[0]
    
    # Specialist çıktıları (score filtresi)
    s_mask = preds_spec["scores"] >= score_thresh
    spec_boxes = preds_spec["boxes"][s_mask]
    spec_scores = preds_spec["scores"][s_mask]
    spec_labels = preds_spec["labels"][s_mask]
    
    # Generalist çıktılarını filtrele ve CODA ID'lerine map et
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
    
    # Birleştir
    all_boxes = torch.cat([spec_boxes, gen_boxes])
    all_scores = torch.cat([spec_scores, gen_scores])
    all_labels = torch.cat([spec_labels, gen_labels])
    
    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_labels
    
    # TEK NMS
    keep = torchvision.ops.nms(all_boxes, all_scores, nms_thresh)
    return all_boxes[keep], all_scores[keep], all_labels[keep]


def hybrid_separate_nms(img_tensor, model_spec, model_gen, coco_to_coda_map, nms_thresh=0.5, score_thresh=0.05):
    """Her modele AYRI NMS uygula, sonra sonuçları birleştir."""
    with torch.no_grad():
        preds_spec = model_spec([img_tensor])[0]
        preds_gen = model_gen([img_tensor])[0]
    
    device = img_tensor.device
    
    # Specialist: filtrele + NMS
    s_mask = preds_spec["scores"] >= score_thresh
    s_boxes = preds_spec["boxes"][s_mask]
    s_scores = preds_spec["scores"][s_mask]
    s_labels = preds_spec["labels"][s_mask]
    if len(s_boxes) > 0:
        keep_s = torchvision.ops.nms(s_boxes, s_scores, nms_thresh)
        s_boxes, s_scores, s_labels = s_boxes[keep_s], s_scores[keep_s], s_labels[keep_s]
    
    # Generalist: filtrele + map + NMS
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
    
    # Basit birleştirme (ikinci NMS YOK)
    return (torch.cat([s_boxes, g_boxes]),
            torch.cat([s_scores, g_scores]),
            torch.cat([s_labels, g_labels]))


def specialist_only(img_tensor, model_spec, score_thresh=0.05):
    with torch.no_grad():
        preds = model_spec([img_tensor])[0]
    mask = preds["scores"] >= score_thresh
    return preds["boxes"][mask], preds["scores"][mask], preds["labels"][mask]


def generalist_only(img_tensor, model_gen, coco_to_coda_map, score_thresh=0.05):
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
# RESULT GENERATION
# ====================================================================

def generate_coco_results(dataset, inference_fn, gt_json_path, output_json, device):
    """Herhangi bir inference fonksiyonunun çıktılarını COCO format JSON'a yaz."""
    coco_gt = COCO(gt_json_path)
    results = []
    
    for i in tqdm(range(len(dataset)), desc=f"Generating {os.path.basename(output_json)}"):
        img_tensor, target = dataset[i]
        img_id = target["image_id"].item()
        img_tensor = img_tensor.to(device)
        
        boxes, scores, labels = inference_fn(img_tensor)
        
        for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score),
            })
    
    with open(output_json, "w") as f:
        json.dump(results, f)
    print(f"Kaydedildi: {output_json} ({len(results)} tespit)")
    return output_json

def evaluate_all():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIR = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
    TEST_ANN = "/workspace/siu_revision/data/coda_split/test_annotations.json"
    RESULTS_DIR = "/workspace/siu_revision/outputs/results"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    COCO_TO_CODA_MAP = {3: 3, 8: 3, 6: 3, 1: 1}
    NMS_THRESH = 0.5
    
    # Dataset
    test_dataset = CODACustomDataset(root_dir=IMG_DIR, annotation_file=TEST_ANN)
    print(f"Test seti: {len(test_dataset)} görüntü")
    
    # Modelleri yükle
    model_spec = load_specialist("/workspace/siu_revision/weights/specialist_retrained.pth", 36, DEVICE)
    model_gen = load_generalist(DEVICE)
    
    # 4 konfigürasyon
    configs = {
        "specialist_only": lambda img: specialist_only(img, model_spec),
        "generalist_only": lambda img: generalist_only(img, model_gen, COCO_TO_CODA_MAP),
        "hybrid_late_nms": lambda img: hybrid_late_nms(img, model_spec, model_gen, COCO_TO_CODA_MAP, NMS_THRESH),
        "hybrid_separate_nms": lambda img: hybrid_separate_nms(img, model_spec, model_gen, COCO_TO_CODA_MAP, NMS_THRESH),
    }
    
    all_results = {}
    
    for name, fn in configs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        # COCO format JSON üret
        json_path = f"{RESULTS_DIR}/{name}_results.json"
        generate_coco_results(test_dataset, fn, TEST_ANN, json_path, DEVICE)
        
        # pycocotools ile değerlendir
        coco_gt = COCO(TEST_ANN)
        coco_dt = coco_gt.loadRes(json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            "mAP@[0.5:0.95]": round(coco_eval.stats[0] * 100, 2),
            "mAP@0.50": round(coco_eval.stats[1] * 100, 2),
            "mAP@0.75": round(coco_eval.stats[2] * 100, 2),
            "AP_small": round(coco_eval.stats[3] * 100, 2),
            "AP_medium": round(coco_eval.stats[4] * 100, 2),
            "AP_large": round(coco_eval.stats[5] * 100, 2),
            "AR@1": round(coco_eval.stats[6] * 100, 2),
            "AR@10": round(coco_eval.stats[7] * 100, 2),
            "AR@100": round(coco_eval.stats[8] * 100, 2),
            "AR_small": round(coco_eval.stats[9] * 100, 2),
            "AR_medium": round(coco_eval.stats[10] * 100, 2),
            "AR_large": round(coco_eval.stats[11] * 100, 2),
        }
        all_results[name] = metrics
        print(f"\n{name} sonuçları:")
        for k, v in metrics.items():
            print(f"  {k}: {v}%")
    
    # Tüm sonuçları kaydet
    summary_path = f"{RESULTS_DIR}/fair_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nTüm özet kaydedildi: {summary_path}")
    
    # Karşılaştırma tablosu yazdır
    print(f"\n{'='*80}")
    print("KARŞILAŞTIRMALI SONUÇLAR (Tüm modeller aynı 300 test sahnesi)")
    print(f"{'='*80}")
    header = f"{'Model':<25} {'mAP':>8} {'mAP50':>8} {'AR@100':>8} {'AP_s':>8} {'AP_m':>8} {'AP_l':>8}"
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        print(f"{name:<25} {m['mAP@[0.5:0.95]']:>7.1f}% {m['mAP@0.50']:>7.1f}% {m['AR@100']:>7.1f}% {m['AP_small']:>7.1f}% {m['AP_medium']:>7.1f}% {m['AP_large']:>7.1f}%")
    
    return all_results

if __name__ == "__main__":
    evaluate_all()
