"""
COCO train2017'den trafik sınıflarını (Person, Car, Truck, Bus, Bicycle, Motorcycle)
çıkar ve CODA veri setiyle birleştir.

COCO sınıf ID'leri:
- 1: person
- 2: bicycle  
- 3: car
- 4: motorcycle
- 6: bus
- 8: truck

CODA'nın kendi 34 sınıfı var (1-34 arası ID'ler).

Birleşik veri setinde:
- COCO sınıflarını CODA ID'lerine MAP ET:
  - COCO person(1) → Combined person(1) — CODA'daki Pedestrian(1) ile aynı
  - COCO car(3) → Combined car(3) — CODA'daki Car(3) ile aynı
  - COCO truck(8) → Combined truck(4) — CODA'daki Truck(4) ile aynı
  - COCO bus(6) → Combined bus(6) — CODA'daki Bus(6) ile aynı
  - COCO bicycle(2) → Combined cyclist(2) — CODA'daki Cyclist(2) ile aynı
  - COCO motorcycle(4) → Combined cyclist(2) — CODA'daki Cyclist(2) ile aynı
- CODA annotation'ları olduğu gibi kalsın

Çıktı: combined_annotations.json (COCO format)
- images listesi: COCO trafik görüntüleri + CODA görüntüleri
- annotations listesi: COCO trafik annotation'ları (map edilmiş) + CODA annotation'ları
- categories: CODA'nın orijinal 34 kategori listesi

COCO'dan her sınıftan MAX 500 görüntü al (toplam ~2000-3000 COCO görüntü).
Böylece CODA'nın 1500 sahnesi ile dengeli bir birleşim oluşur.
"""
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

random.seed(42)

# Remote server paths
COCO_SUBSET_DIR = "/workspace/siu_revision/data/coco_subset"
COCO_ANN_ZIP = f"{COCO_SUBSET_DIR}/annotations_trainval2017.zip"
COCO_ANN = f"{COCO_SUBSET_DIR}/annotations/instances_train2017.json"
COCO_IMG_DIR = f"{COCO_SUBSET_DIR}/train2017"
CODA_ANN = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json"
CODA_IMG_DIR = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
OUTPUT_DIR = "/workspace/siu_revision/data/combined_dataset"
OUTPUT_IMG_DIR = f"{OUTPUT_DIR}/images"
OUTPUT_ANN = f"{OUTPUT_DIR}/combined_annotations.json"

COCO_TO_CODA_CLASS_MAP = {
    1: 1,    # person → Pedestrian
    3: 3,    # car → Car
    8: 4,    # truck → Truck
    6: 6,    # bus → Bus
    2: 2,    # bicycle → Cyclist
    4: 2,    # motorcycle → Cyclist
}

MAX_IMAGES_PER_CLASS = 500

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Already exists: {dest}")

def download_image(file_name, dest_dir):
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(dest_path):
        return True
    
    url = f"http://images.cocodataset.org/train2017/{file_name}"
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        # Retry once
        try:
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception:
            print(f"Failed to download image {file_name}: {e}")
            return False

def main():
    os.makedirs(COCO_SUBSET_DIR, exist_ok=True)
    os.makedirs(COCO_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    # 1. Download COCO annotations zip if not present
    if not os.path.exists(COCO_ANN):
        print("COCO annotations indiriliyor...")
        download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", COCO_ANN_ZIP)
        print("Extracting COCO annotations...")
        with zipfile.ZipFile(COCO_ANN_ZIP, 'r') as zip_ref:
            zip_ref.extract("annotations/instances_train2017.json", COCO_SUBSET_DIR)
    else:
        print("COCO annotations zaten mevcut.")

    # 2. COCO annotation'larını oku
    print("COCO annotations okunuyor...")
    with open(COCO_ANN) as f:
        coco = json.load(f)
    
    # 3. İlgili COCO sınıflarının annotation'larını filtrele
    target_cat_ids = set(COCO_TO_CODA_CLASS_MAP.keys())
    
    # Her sınıf için görüntüleri topla
    class_to_images = defaultdict(set)
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            class_to_images[ann["category_id"]].add(ann["image_id"])
            img_to_anns[ann["image_id"]].append(ann)
    
    # Her sınıftan max N görüntü seç
    selected_image_ids = set()
    for cat_id, img_ids in class_to_images.items():
        img_list = list(img_ids)
        random.shuffle(img_list)
        selected = img_list[:MAX_IMAGES_PER_CLASS]
        selected_image_ids.update(selected)
        print(f"  COCO sınıf {cat_id}: {len(img_ids)} görüntüden {len(selected)} seçildi")
    
    print(f"  Toplam seçilen COCO görüntü: {len(selected_image_ids)}")
    
    # 4. Seçilen COCO görüntülerinin bilgilerini al
    coco_img_map = {img["id"]: img for img in coco["images"]}
    
    # Download selected COCO images concurrently
    print("Seçilen COCO görüntüleri indiriliyor (Multi-threaded)...")
    selected_files = [coco_img_map[img_id]["file_name"] for img_id in selected_image_ids if img_id in coco_img_map]
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_image, fname, COCO_IMG_DIR): fname for fname in selected_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="COCO Downloads"):
            if future.result():
                success_count += 1
                
    print(f"  Görüntü indirme tamamlandı: {success_count}/{len(selected_files)} başarılı.")
    
    # 5. CODA annotation'larını oku
    print("CODA annotations okunuyor...")
    with open(CODA_ANN) as f:
        coda = json.load(f)
    
    # 6. Birleşik annotation oluştur
    combined_images = []
    combined_annotations = []
    new_ann_id = 1
    
    # ID offset: COCO image ID'leri CODA ile çakışmasın
    coco_id_offset = 1000000
    
    # COCO görüntülerini ekle
    print("COCO görüntüleri birleşik veri setine ekleniyor...")
    copied_count = 0
    for img_id in selected_image_ids:
        if img_id not in coco_img_map:
            continue
        img_info = coco_img_map[img_id]
        src_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
        
        if not os.path.exists(src_path):
            continue
        
        new_img_id = img_id + coco_id_offset
        new_file_name = f"coco_{img_info['file_name']}"
        dst_path = os.path.join(OUTPUT_IMG_DIR, new_file_name)
        
        # Symlink ile kopyala
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)
        
        combined_images.append({
            "id": new_img_id,
            "file_name": new_file_name,
            "width": img_info["width"],
            "height": img_info["height"]
        })
        
        # Bu görüntüdeki ilgili annotation'ları ekle (sınıf map ile)
        for ann in img_to_anns[img_id]:
            if ann["category_id"] in COCO_TO_CODA_CLASS_MAP:
                new_ann = {
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": COCO_TO_CODA_CLASS_MAP[ann["category_id"]],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0)
                }
                combined_annotations.append(new_ann)
                new_ann_id += 1
        
        copied_count += 1
    
    print(f"  {copied_count} COCO görüntü eklendi")
    
    # CODA görüntülerini ekle (symlink)
    print("CODA görüntüleri ekleniyor...")
    for img_info in coda["images"]:
        file_name = img_info["file_name"]
        if file_name.startswith("images/"):
            file_name = file_name.replace("images/", "")
        
        src_path = os.path.join(CODA_IMG_DIR, file_name)
        new_file_name = f"coda_{file_name}"
        dst_path = os.path.join(OUTPUT_IMG_DIR, new_file_name)
        
        # Eger CODA resmi mevcutsa ekle, yoksa bos zero-tensor olarak skip/dummy olacak sekilde handle et
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
            
            combined_images.append({
                "id": img_info["id"],
                "file_name": new_file_name,
                "width": img_info["width"],
                "height": img_info["height"]
            })
            
            # Annotation'lari da sadece resim mevcutsa ekliyoruz (veya dummy oldugunda da bos target handled)
            for ann in coda["annotations"]:
                if ann["image_id"] == img_info["id"]:
                    new_ann = {
                        "id": new_ann_id,
                        "image_id": ann["image_id"],
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann.get("iscrowd", 0)
                    }
                    combined_annotations.append(new_ann)
                    new_ann_id += 1
        else:
            # Missing image (like kitti/nuscenes)
            combined_images.append({
                "id": img_info["id"],
                "file_name": new_file_name,  # will lead to None in imread, handled by CombinedDataset
                "width": img_info["width"],
                "height": img_info["height"]
            })
            
            for ann in coda["annotations"]:
                if ann["image_id"] == img_info["id"]:
                    new_ann = {
                        "id": new_ann_id,
                        "image_id": ann["image_id"],
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann.get("iscrowd", 0)
                    }
                    combined_annotations.append(new_ann)
                    new_ann_id += 1
    
    print(f"  {len(coda['images'])} CODA görüntü eklendi")
    
    # Categories: CODA'nın orijinal listesini kullan
    combined = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": coda["categories"]
    }
    
    with open(OUTPUT_ANN, "w") as f:
        json.dump(combined, f)
    
    print(f"\nBirleşik veri seti oluşturuldu:")
    print(f"  Toplam görüntü: {len(combined_images)}")
    print(f"  Toplam annotation: {len(combined_annotations)}")
    print(f"  Annotation dosyası: {OUTPUT_ANN}")

if __name__ == "__main__":
    main()
