"""
BİRLEŞİK EĞİTİM BASELINE (Reviewer 2 İsteği)

COCO'dan ilgili sınıfların alt kümesi (500 adet temsilci resim) + CODA veri setini birleştir.
Tek bir Faster R-CNN modeli eğit.
Bu, "neden geç füzyon gerekli, neden birleşik eğitim yetmez?" sorusuna cevap verir.
"""
import os
import sys
import json
import zipfile
import urllib.request
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.dataset import CODACustomDataset, collate_fn
from src.models import get_trainable_model
from src.evaluation import evaluate_coco_metrics, generate_coco_results
from src.fusion import fusion_specialist_only

# Download helper
def download_url(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"File already exists: {dest}")

def main():
    print("=" * 60)
    print("BİRLEŞİK EĞİTİM BASELINE DENEYİ")
    print("=" * 60)

    # Paths for COCO subset
    COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    COCO_ZIP_PATH = PROJECT_ROOT / "data" / "coco_annotations.zip"
    COCO_EXTRACT_DIR = PROJECT_ROOT / "data"
    
    os.makedirs(PROJECT_ROOT / "data", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "results", exist_ok=True)

    # 1. Download COCO annotations
    download_url(COCO_ANN_URL, COCO_ZIP_PATH)
    
    # Extract
    coco_json_path = COCO_EXTRACT_DIR / "annotations" / "instances_train2017.json"
    if not os.path.exists(coco_json_path):
        print("Extracting COCO annotations...")
        with zipfile.ZipFile(COCO_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extract("annotations/instances_train2017.json", COCO_EXTRACT_DIR)
    else:
        print("COCO annotations already extracted.")

    # 2. Filter COCO annotations & download subset images
    print("Filtering COCO annotations for Person, Car, Bus, Truck...")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Categories: 1=person, 3=car, 6=bus, 8=truck
    target_cats = {1, 3, 6, 8}
    
    # Find all images containing target categories
    img_ids_with_cats = set()
    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in target_cats:
            img_id = ann["image_id"]
            img_ids_with_cats.add(img_id)
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

    # Map COCO images info
    coco_images = {img["id"]: img for img in coco_data["images"] if img["id"] in img_ids_with_cats}
    
    # Select subset of 500 images for combined training
    selected_img_ids = list(coco_images.keys())[:500]
    print(f"Selected {len(selected_img_ids)} COCO images for joint training baseline.")

    # Download COCO images
    coco_img_dir = PROJECT_ROOT / "data" / "coco_subset"
    os.makedirs(coco_img_dir, exist_ok=True)
    
    print("Downloading COCO subset images...")
    for img_id in tqdm(selected_img_ids):
        img_info = coco_images[img_id]
        file_name = img_info["file_name"]
        dest_path = coco_img_dir / file_name
        if not os.path.exists(dest_path):
            img_url = f"http://images.cocodataset.org/train2017/{file_name}"
            try:
                urllib.request.urlretrieve(img_url, dest_path)
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

    # 3. Create combined annotation JSON in CODA layout
    print("Combining CODA & COCO annotations...")
    with open(ANN_FILE, "r") as f:
        coda_data = json.load(f)

    # CODA class maps: 1=Pedestrian, 3=Car
    # COCO maps: 1 -> 1, 3,6,8 -> 3
    coco_to_coda = {1: 1, 3: 3, 6: 3, 8: 3}
    
    combined_images = []
    combined_annotations = []
    ann_id_counter = 1000000  # Start high to avoid collision

    # Add all CODA data
    for img in coda_data["images"]:
        # Ensure path points to the CODA image path
        img_copy = img.copy()
        # CODA custom dataset expected local file paths relative to IMG_DIR
        combined_images.append(img_copy)

    for ann in coda_data["annotations"]:
        combined_annotations.append(ann)

    # Add selected COCO subset data
    for img_id in selected_img_ids:
        img_info = coco_images[img_id]
        # In CODA Custom Dataset reader, we need to match path
        # Let's symlink or copy COCO subset images to the same folder
        # or handle it by placing COCO images in the directory
        file_name = img_info["file_name"]
        src_path = coco_img_dir / file_name
        dest_link = IMG_DIR / file_name
        if not os.path.exists(dest_link) and os.path.exists(src_path):
            # Create a hard link or copy
            if sys.platform == "win32":
                import shutil
                shutil.copy(src_path, dest_link)
            else:
                os.link(src_path, dest_link)
        
        # Add to image list
        combined_images.append({
            "id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
            "file_name": file_name
        })

        # Add annotations
        for ann in img_id_to_anns[img_id]:
            cat_id = ann["category_id"]
            coda_cat_id = coco_to_coda[cat_id]
            
            combined_annotations.append({
                "id": ann_id_counter,
                "image_id": img_id,
                "category_id": coda_cat_id,
                "segmentation": [],
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": 0
            })
            ann_id_counter += 1

    combined_coco_format = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": coda_data["categories"]
    }

    combined_ann_file = PROJECT_ROOT / "data" / "combined_annotations.json"
    with open(combined_ann_file, "w") as f:
        json.dump(combined_coco_format, f)
    print(f"Combined annotations saved to: {combined_ann_file}")

    # 4. Train Faster R-CNN on combined dataset
    print("\n--- Training Faster R-CNN on Combined Dataset ---")
    combined_dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(combined_ann_file))
    train_loader = DataLoader(combined_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = get_trainable_model(NUM_CLASSES_CODA)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=TRAIN_LR, momentum=TRAIN_MOMENTUM, weight_decay=TRAIN_WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=TRAIN_LR_MILESTONES, gamma=TRAIN_LR_GAMMA)

    # Let's run 5 epochs for quick baseline, or 20 epochs as config states if time allows.
    # To keep execution within limits, we will do 5 epochs for the baseline.
    epochs = 5
    print(f"Training for {epochs} epochs on combined dataset...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            valid_images = [img.to(DEVICE) for img in images]
            valid_targets = [{k: v.to(DEVICE) for k, v in tgt.items()} for tgt in targets]
            
            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
        lr_scheduler.step()
        print(f"Epoch {epoch+1} Completed. Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save weights
    combined_weights_path = WEIGHTS_DIR / "combined_trained_model.pth"
    torch.save(model.state_dict(), combined_weights_path)
    print(f"Saved combined model weights to: {combined_weights_path}")

    # 5. Evaluate on pure CODA dataset (original base-val-1500 validation annotation)
    print("\n--- Evaluating Combined Model on Pure CODA Dataset ---")
    model.eval()
    coda_dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(ANN_FILE))
    
    fn_combined = lambda img: fusion_specialist_only(model, img.to(DEVICE), score_thresh=SCORE_THRESH_EVAL)
    
    combined_results_json = OUTPUT_DIR / "results" / "combined_coco_predictions.json"
    generate_coco_results(coda_dataset, fn_combined, str(ANN_FILE), str(combined_results_json))
    metrics = evaluate_coco_metrics(str(ANN_FILE), str(combined_results_json))

    # Save metrics
    metrics_path = OUTPUT_DIR / "results" / "combined_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Combined Model Performance on CODA ---")
    print(f"mAP@0.50: {metrics['mAP@0.50']*100:.2f}%")
    print(f"AR@100: {metrics['AR@100']*100:.2f}%")

if __name__ == "__main__":
    main()
