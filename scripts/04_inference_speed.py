"""
INFERENCE SPEED (Reviewer 3 İsteği)

GPU üzerinde her model için FPS ve gecikme (ms) değerlerini ölç.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from src.dataset import CODACustomDataset
from src.models import load_specialist, load_generalist
from src.fusion import (
    fusion_late_nms, 
    fusion_separate_nms, 
    fusion_specialist_only, 
    fusion_generalist_only
)
from src.evaluation import measure_inference_speed

def main():
    print("=" * 60)
    print("GERÇEK ZAMANLI HIZ ANALİZİ (FPS & LATENCY)")
    print("=" * 60)

    # 1. Veri seti yükle
    dataset = CODACustomDataset(root_dir=str(IMG_DIR), annotation_file=str(ANN_FILE))
    print(f"Veri seti: {len(dataset)} sahne")

    # 2. Modelleri yükle
    model_spec = load_specialist(str(SPECIALIST_WEIGHTS), NUM_CLASSES_CODA, DEVICE)
    model_gen = load_generalist(DEVICE)

    # 3. Model inference tanımlamaları (Hız ölçümü için 0.5 threshold yeterli)
    models_to_test = {
        "specialist": {
            "fn": lambda img: fusion_specialist_only(model_spec, img.to(DEVICE), score_thresh=SCORE_THRESH_INFERENCE),
            "name": "Tekil Uzman (Baseline)"
        },
        "generalist": {
            "fn": lambda img: fusion_generalist_only(model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, score_thresh=SCORE_THRESH_INFERENCE),
            "name": "Tekil Genelci"
        },
        "hybrid_late_nms": {
            "fn": lambda img: fusion_late_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_INFERENCE),
            "name": "Hibrit (Late Fusion - Önerilen)"
        },
        "hybrid_separate_nms": {
            "fn": lambda img: fusion_separate_nms(model_spec, model_gen, img.to(DEVICE), COCO_TO_CODA_MAP, NMS_IOU_THRESH, score_thresh=SCORE_THRESH_INFERENCE),
            "name": "Hibrit (Separate NMS - Alternatif)"
        }
    }

    speed_results = {}

    for key, model_info in models_to_test.items():
        print(f"\nMeasuring speed for: {model_info['name']}")
        # Warmup and profile on first 150 images
        res = measure_inference_speed(
            dataset, model_info["fn"], DEVICE, 
            num_warmup=15, num_measure=150
        )
        
        print(f"  Avg Latency: {res['avg_inference_ms']:.2f} ms ± {res['std_inference_ms']:.2f} ms")
        print(f"  FPS: {res['fps']:.2f}")
        
        speed_results[key] = {
            "model_name": model_info["name"],
            "avg_latency_ms": res["avg_inference_ms"],
            "std_latency_ms": res["std_inference_ms"],
            "fps": res["fps"]
        }

    # Save to JSON
    save_path = OUTPUT_DIR / "results" / "inference_speed.json"
    with open(save_path, "w") as f:
        json.dump(speed_results, f, indent=4)
    print(f"\nInference speed results saved to {save_path}")

if __name__ == "__main__":
    main()
