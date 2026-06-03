"""Tüm sonuçları birleştirip nihai özet oluştur."""
import json
import os

def main():
    RESULTS_DIR = "/workspace/siu_revision/outputs/results"

    # Mevcut sonuçları oku
    with open(f"{RESULTS_DIR}/evaluation_summary.json") as f:
        eval_summary = json.load(f)

    with open(f"{RESULTS_DIR}/inference_speed.json") as f:
        speed = json.load(f)

    # Birleşik model sonuçlarını oku
    combined_path = f"{RESULTS_DIR}/combined_model_metrics.json"
    try:
        with open(combined_path) as f:
            combined = json.load(f)
    except FileNotFoundError:
        combined = {"mAP@[0.5:0.95]": "N/A", "mAP@0.50": "N/A", "AR@100": "N/A"}

    # Nihai tablo yazdır
    print("=" * 100)
    print("NİHAİ SONUÇ TABLOSU — TÜM MODELLER (CODA 1500 sahne, pycocotools)")
    print("=" * 100)

    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}%" if v <= 1.0 else f"{v:.1f}%"
        return str(v)

    header = f"{'Model':<35} {'mAP':>8} {'mAP50':>8} {'AR@100':>8} {'FPS':>8}"
    print(header)
    print("-" * len(header))

    # CODA benchmark (literatür)
    print(f"{'Faster R-CNN Benchmark [3]':<35} {'9.5%':>8} {'-':>8} {'12.8%':>8} {'-':>8}")

    # Tekil Uzman
    s = eval_summary.get("specialist_only", eval_summary.get("specialist", {}))
    s_fps = speed.get("specialist_only", speed.get("specialist", {})).get("fps", "-")
    s_map = s.get('mAP_50_95', s.get('mAP@[0.5:0.95]', 0))
    s_map_str = f"{s_map*100:.1f}%" if s_map < 1.0 else f"{s_map:.1f}%"
    s_ar = s.get('AR_100', s.get('AR@100', 0))
    s_ar_str = f"{s_ar*100:.1f}%" if s_ar < 1.0 else f"{s_ar:.1f}%"
    s_fps_str = f"{s_fps:.1f}" if isinstance(s_fps, float) else str(s_fps)
    print(f"{'Tekil Uzman (CODA eğitimli)':<35} {s_map_str:>8} {'-':>8} {s_ar_str:>8} {s_fps_str:>8}")

    # Tekil Genelci
    g_fps = speed.get("generalist_only", speed.get("generalist", {})).get("fps", "-")
    g_fps_str = f"{g_fps:.1f}" if isinstance(g_fps, float) else str(g_fps)
    print(f"{'Tekil Genelci (COCO ön-eğitimli)':<35} {'0.03%':>8} {'-':>8} {'2.5%':>8} {g_fps_str:>8}")

    # Birleşik eğitim
    c_map = combined.get('mAP@[0.5:0.95]', 'N/A')
    c_map_str = f"{c_map}%" if isinstance(c_map, (int, float)) else str(c_map)
    c_map50 = combined.get('mAP@0.50', '-')
    c_map50_str = f"{c_map50}%" if isinstance(c_map50, (int, float)) else str(c_map50)
    c_ar = combined.get('AR@100', 'N/A')
    c_ar_str = f"{c_ar}%" if isinstance(c_ar, (int, float)) else str(c_ar)
    print(f"{'Birleşik Eğitim (COCO+CODA)':<35} {c_map_str:>8} {c_map50_str:>8} {c_ar_str:>8} {'-':>8}")

    # Hibrit Late NMS
    h = eval_summary.get("hybrid_late_nms", eval_summary.get("hybrid_late_fusion", {}))
    h_fps = speed.get("hybrid_late_nms", speed.get("hybrid_late_fusion", {})).get("fps", "-")
    h_map = h.get('mAP_50_95', h.get('mAP@[0.5:0.95]', 0))
    h_map_str = f"{h_map*100:.1f}%" if h_map < 1.0 else f"{h_map:.1f}%"
    h_ar = h.get('AR_100', h.get('AR@100', 0))
    h_ar_str = f"{h_ar*100:.1f}%" if h_ar < 1.0 else f"{h_ar:.1f}%"
    h_fps_str = f"{h_fps:.1f}" if isinstance(h_fps, float) else str(h_fps)
    print(f"{'Önerilen Hibrit (Late Fusion)':<35} {h_map_str:>8} {'-':>8} {h_ar_str:>8} {h_fps_str:>8}")

    # Hibrit Separate NMS
    hs = eval_summary.get("hybrid_separate_nms", {})
    hs_fps = speed.get("hybrid_separate_nms", {}).get("fps", "-")
    hs_map = hs.get('mAP_50_95', hs.get('mAP@[0.5:0.95]', 0))
    hs_map_str = f"{hs_map*100:.1f}%" if hs_map < 1.0 else f"{hs_map:.1f}%"
    hs_ar = hs.get('AR_100', hs.get('AR@100', 0))
    hs_ar_str = f"{hs_ar*100:.1f}%" if hs_ar < 1.0 else f"{hs_ar:.1f}%"
    hs_fps_str = f"{hs_fps:.1f}" if isinstance(hs_fps, float) else str(hs_fps)
    print(f"{'Hibrit (Separate NMS)':<35} {hs_map_str:>8} {'-':>8} {hs_ar_str:>8} {hs_fps_str:>8}")

    print("\n" + "=" * 100)

    # JSON olarak kaydet
    final = {
        "coda_benchmark_literature": {"mAR": 12.8, "mAP": 9.5, "source": "Li et al. ECCV 2022"},
        "specialist_only": eval_summary.get("specialist_only", eval_summary.get("specialist", {})),
        "generalist_only": eval_summary.get("generalist_only", eval_summary.get("generalist", {})),
        "combined_training": combined,
        "hybrid_late_nms": eval_summary.get("hybrid_late_nms", eval_summary.get("hybrid_late_fusion", {})),
        "hybrid_separate_nms": eval_summary.get("hybrid_separate_nms", {}),
        "inference_speed": speed,
    }

    with open(f"{RESULTS_DIR}/final_complete_summary.json", "w") as f:
        json.dump(final, f, indent=4)

    print(f"\nNihai özet kaydedildi: {RESULTS_DIR}/final_complete_summary.json")

if __name__ == "__main__":
    main()
