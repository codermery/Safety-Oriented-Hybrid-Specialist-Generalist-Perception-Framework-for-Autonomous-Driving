import json
import random
from pathlib import Path
from collections import Counter

def split_coda_dataset(ann_file, output_dir, train_ratio=0.8, seed=42):
    """CODA veri setini train/test olarak böl."""
    with open(ann_file) as f:
        coco = json.load(f)
    
    # Image ID'leri karıştır
    image_ids = [img["id"] for img in coco["images"]]
    random.seed(seed)
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    test_ids = set(image_ids[split_idx:])
    
    # Images ve annotations'ı böl
    train_images = [img for img in coco["images"] if img["id"] in train_ids]
    test_images = [img for img in coco["images"] if img["id"] in test_ids]
    train_anns = [ann for ann in coco["annotations"] if ann["image_id"] in train_ids]
    test_anns = [ann for ann in coco["annotations"] if ann["image_id"] in test_ids]
    
    # Kaydet
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, imgs, anns in [("train", train_images, train_anns), ("test", test_images, test_anns)]:
        split_coco = {
            "images": imgs,
            "annotations": anns,
            "categories": coco["categories"]
        }
        out_path = output_dir / f"{name}_annotations.json"
        with open(out_path, "w") as f:
            json.dump(split_coco, f)
        
        # İstatistikler
        cat_counts = Counter(ann["category_id"] for ann in anns)
        print(f"\n{name.upper()}: {len(imgs)} görüntü, {len(anns)} annotation")
        for cat_id, count in sorted(cat_counts.items()):
            cat_name = next((c["name"] for c in coco["categories"] if c["id"] == cat_id), f"ID:{cat_id}")
            print(f"  {cat_name} (ID:{cat_id}): {count}")
    
    # Test ID'lerini kaydet (sonra lazım olacak)
    with open(output_dir / "test_image_ids.json", "w") as f:
        json.dump(sorted(test_ids), f)
    
    print(f"\nDosyalar kaydedildi: {output_dir}")

if __name__ == "__main__":
    split_coda_dataset(
        ann_file="/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json",
        output_dir="/workspace/siu_revision/data/coda_split"
    )
