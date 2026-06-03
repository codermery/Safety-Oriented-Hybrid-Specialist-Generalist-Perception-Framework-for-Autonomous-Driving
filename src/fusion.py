import torch
import torchvision

def fusion_late_nms(model_specialist, model_generalist, image_tensor, 
                    coco_to_coda_map, nms_thresh=0.5, score_thresh=0.05):
    """
    Strateji A — Makalede Önerilen (Late Fusion):
    İki modelin ham çıktılarını birleştir, sonra tek bir NMS uygula.
    """
    device = image_tensor.device
    with torch.no_grad():
        preds_spec = model_specialist([image_tensor])[0]
        preds_gen = model_generalist([image_tensor])[0]

    # Genelci çıktılarından sadece ilgili sınıfları al ve CODA ID'lerine map et
    gen_boxes, gen_scores, gen_labels = [], [], []
    for box, score, label in zip(preds_gen["boxes"], preds_gen["scores"], preds_gen["labels"]):
        lid = label.item()
        if lid in coco_to_coda_map and score.item() >= score_thresh:
            gen_boxes.append(box)
            gen_scores.append(score)
            gen_labels.append(coco_to_coda_map[lid])

    if gen_boxes:
        gen_boxes = torch.stack(gen_boxes)
        gen_scores = torch.tensor([s.item() for s in gen_scores], device=device)
        gen_labels = torch.tensor(gen_labels, device=device)
    else:
        gen_boxes = torch.empty((0, 4), device=device)
        gen_scores = torch.empty(0, device=device)
        gen_labels = torch.empty(0, dtype=torch.int64, device=device)

    # Uzman çıktılarını filtrele
    spec_mask = preds_spec["scores"] >= score_thresh
    spec_boxes = preds_spec["boxes"][spec_mask]
    spec_scores = preds_spec["scores"][spec_mask]
    spec_labels = preds_spec["labels"][spec_mask]

    # Birleştir
    all_boxes = torch.cat([spec_boxes, gen_boxes])
    all_scores = torch.cat([spec_scores, gen_scores])
    all_labels = torch.cat([spec_labels, gen_labels])

    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_labels

    # Tek NMS
    keep = torchvision.ops.nms(all_boxes, all_scores, nms_thresh)
    return all_boxes[keep], all_scores[keep], all_labels[keep]


def fusion_separate_nms(model_specialist, model_generalist, image_tensor,
                        coco_to_coda_map, nms_thresh=0.5, score_thresh=0.05):
    """
    Strateji B — Alternatif (Reviewer 2 İsteği):
    Her modele ayrı NMS uygula, sonra sonuçları basitçe birleştir.
    """
    device = image_tensor.device
    with torch.no_grad():
        preds_spec = model_specialist([image_tensor])[0]
        preds_gen = model_generalist([image_tensor])[0]

    # Uzman: kendi NMS'i
    spec_mask = preds_spec["scores"] >= score_thresh
    spec_boxes = preds_spec["boxes"][spec_mask]
    spec_scores = preds_spec["scores"][spec_mask]
    spec_labels = preds_spec["labels"][spec_mask]
    if len(spec_boxes) > 0:
        keep_s = torchvision.ops.nms(spec_boxes, spec_scores, nms_thresh)
        spec_boxes = spec_boxes[keep_s]
        spec_scores = spec_scores[keep_s]
        spec_labels = spec_labels[keep_s]

    # Genelci: filtrele, map et, kendi NMS'i
    gen_boxes, gen_scores, gen_labels = [], [], []
    for box, score, label in zip(preds_gen["boxes"], preds_gen["scores"], preds_gen["labels"]):
        lid = label.item()
        if lid in coco_to_coda_map and score.item() >= score_thresh:
            gen_boxes.append(box)
            gen_scores.append(score)
            gen_labels.append(coco_to_coda_map[lid])

    if gen_boxes:
        gen_boxes = torch.stack(gen_boxes)
        gen_scores = torch.tensor([s.item() for s in gen_scores], device=device)
        gen_labels = torch.tensor(gen_labels, device=device)
        keep_g = torchvision.ops.nms(gen_boxes, gen_scores, nms_thresh)
        gen_boxes = gen_boxes[keep_g]
        gen_scores = gen_scores[keep_g]
        gen_labels = gen_labels[keep_g]
    else:
        gen_boxes = torch.empty((0, 4), device=device)
        gen_scores = torch.empty(0, device=device)
        gen_labels = torch.empty(0, dtype=torch.int64, device=device)

    # Basit birleştirme (ikinci NMS yok)
    all_boxes = torch.cat([spec_boxes, gen_boxes])
    all_scores = torch.cat([spec_scores, gen_scores])
    all_labels = torch.cat([spec_labels, gen_labels])

    return all_boxes, all_scores, all_labels


def fusion_specialist_only(model_specialist, image_tensor, score_thresh=0.05):
    """Sadece uzman model (Baseline karşılaştırma için)."""
    with torch.no_grad():
        preds = model_specialist([image_tensor])[0]
    mask = preds["scores"] >= score_thresh
    return preds["boxes"][mask], preds["scores"][mask], preds["labels"][mask]


def fusion_generalist_only(model_generalist, image_tensor, 
                           coco_to_coda_map, score_thresh=0.05):
    """Sadece genelci model (Baseline karşılaştırma için)."""
    device = image_tensor.device
    with torch.no_grad():
        preds = model_generalist([image_tensor])[0]

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
