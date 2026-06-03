import os
import cv2
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
import torchvision
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

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train():
    print("=" * 60)
    print("TRAINING CODA SPECIALIST FROM SCRATCH (All 1500 scenes)")
    print("=" * 60)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 36
    IMG_DIR = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
    ANN_FILE = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json"
    
    dataset = CODACustomDataset(root_dir=IMG_DIR, annotation_file=ANN_FILE)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 16], gamma=0.1)
    
    loss_history = []
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        count = 0
        
        for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/20"):
            # Data augmentation + filter
            aug_images = []
            valid_targets = []
            for img, tgt in zip(images, targets):
                if tgt["boxes"].shape[0] > 0 or img.sum() > 0.1:
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
        
        lr_scheduler.step()
        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/20 | Loss: {avg_loss:.4f}")
    
    # Save weights
    weights_dir = "/workspace/siu_revision/weights"
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, "coda_fresh_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    # Save Loss plot
    figures_dir = "/workspace/siu_revision/outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, 21), loss_history, marker='o', color='red', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fresh specialist training loss curve (1500 images)')
    plt.grid(True)
    plt.legend()
    loss_plot_path = os.path.join(figures_dir, "fresh_train_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to: {loss_plot_path}")
    
    return loss_history

if __name__ == "__main__":
    train()
