import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_specialist(weights_path, num_classes, device):
    """CODA üzerinde fine-tune edilmiş uzman modeli yükle."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_generalist(device):
    """COCO pretrained genelci modeli yükle."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model

def get_trainable_model(num_classes):
    """Eğitim için model oluştur (birleşik eğitim baseline'ı için)."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
