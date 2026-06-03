import os
import cv2
import torch
import random
import sys
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

class CombinedDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.root_dir, img_info["file_name"])
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
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
        
        target = {"image_id": torch.tensor([img_id])}
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        
        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 36  # 35 CODA classes + 1 background
    IMG_DIR = "/workspace/siu_revision/data/combined_dataset/images"
    ANN_FILE = "/workspace/siu_revision/data/combined_dataset/combined_annotations.json"
    SAVE_PATH = "/workspace/siu_revision/weights/combined_model.pth"
    
    os.makedirs("/workspace/siu_revision/weights", exist_ok=True)
    
    print(f"Loading dataset from annotations: {ANN_FILE}")
    dataset = CombinedDataset(root_dir=IMG_DIR, annotation_file=ANN_FILE)
    # Using num_workers=2 to prevent resource exhaustion, batch_size=4 as specified
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    print(f"Birleşik veri seti yükleme tamamlandı: {len(dataset)} görüntü")
    
    print("Loading pretrained Faster R-CNN ResNet-50 FPN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 16], gamma=0.1)
    
    print("Starting training combined model...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        count = 0
        
        # We wrap with tqdm to see the training progress clearly
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/20")
        for images, targets in pbar:
            aug_images = []
            valid_targets = []
            for img, tgt in zip(images, targets):
                # Skip dummy images (no boxes and zero pixel values)
                if tgt["boxes"].shape[0] == 0 and img.sum() < 0.1:
                    continue
                
                # Apply 50% probability brightness/contrast data augmentation
                if random.random() > 0.5:
                    factor = random.uniform(0.6, 1.4)
                    img = TF.adjust_brightness(img, factor)
                    img = TF.adjust_contrast(img, factor)
                aug_images.append(img.to(DEVICE))
                valid_targets.append({k: v.to(DEVICE) for k, v in tgt.items()})
            
            if not aug_images:
                continue
            
            loss_dict = model(aug_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if torch.isnan(losses):
                continue
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            count += 1
            
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}/20 completed. Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model successfully saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train()
