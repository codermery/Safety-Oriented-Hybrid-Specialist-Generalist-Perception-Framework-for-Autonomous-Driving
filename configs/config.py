import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "coda_dataset" / "CODA" / "base-val-1500"
IMG_DIR = DATA_DIR / "images"
ANN_FILE = DATA_DIR / "corner_case.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model
NUM_CLASSES_CODA = 36  # 35 sınıf + 1 arka plan
SPECIALIST_WEIGHTS = WEIGHTS_DIR / "coda_advanced_model.pth"

# Inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NMS_IOU_THRESH = 0.5
SCORE_THRESH_INFERENCE = 0.5
SCORE_THRESH_EVAL = 0.05  # COCOeval düşük skor ister

# COCO → CODA sınıf eşleştirme
COCO_TO_CODA_MAP = {
    3: 3,   # Car → Car
    8: 3,   # Truck → Car
    6: 3,   # Bus → Car
    1: 1,   # Person → Pedestrian
}

# Tüm CODA sınıfları
CODA_CLASSES = {
    1: "Pedestrian", 2: "Cyclist", 3: "Car", 4: "Truck", 6: "Bus",
    17: "Dog", 18: "Barrier", 23: "Cone", 26: "Traffic Sign",
    27: "Debris", 30: "Concrete Block"
}

# Eğitim parametreleri (birleşik eğitim baseline için de)
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 4
TRAIN_LR = 0.005
TRAIN_MOMENTUM = 0.9
TRAIN_WEIGHT_DECAY = 0.0005
TRAIN_LR_MILESTONES = [12, 16]
TRAIN_LR_GAMMA = 0.1
