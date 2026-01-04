import os
import re
import cv2
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
from mediapipe.python.solutions import hands as mp_hands



#User Setthings (EDIT THESE)
DATASET_DIR = r"D:\BdSL\dataset_224"      #change to your dataset root
RESULTS_DIR = r"D:\BdSL\results\mediapipe_failure_audit"   #your output folder 
CLASS_MEANING_PATH = r"D:\BdSL\results\class_meaning.md"   # your mapping file
BENGALI_FONT_PATH = r"C:\Users\steve\Downloads\Noto_Sans_Bengali\NotoSansBengali-VariableFont_wdth,wght.ttf" # Force Bengali font (change path to your own TTF)
N_CLASSES = 49

# MediaPipe settings (match your pipeline choices)
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5
STATIC_IMAGE_MODE = True
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 1

MAX_ITEMS = None          
PRINT_EVERY = 500

SAVE_FAILURE_EXAMPLES_SUBDIR = None   
MAX_FAILURE_EXAMPLES = 200

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")



def _set_bengali_font(font_path: str = BENGALI_FONT_PATH) -> FontProperties:
 
    if not font_path:
        return None
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Bengali font not found at: {font_path}")

    fm.fontManager.addfont(font_path)
    font_prop = FontProperties(fname=font_path)

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return font_prop


def _read_class_meaning(path: str):
  
    mapping_pairs = []
    if not path or (not os.path.exists(path)):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            left, right = [x.strip() for x in line.split("=", 1)]
            if re.fullmatch(r"\d+", left):
                mapping_pairs.append((int(left), right))

    mapping = {}
    dups = {}
    for k, v in mapping_pairs:
        if k in mapping:
            dups.setdefault(k, [mapping[k]]).append(v)
        mapping[k] = v

    for k in dups.keys():
        if (k - 1) not in mapping:
            occurrences = [v for kk, v in mapping_pairs if kk == k]
            if len(occurrences) >= 2:
                mapping[k - 1] = occurrences[0]
                mapping[k] = occurrences[1]

    return mapping


def build_class_labels():
  
    class_map = _read_class_meaning(CLASS_MEANING_PATH)
    labels_list = [class_map.get(i, str(i)) for i in range(N_CLASSES)]
    return class_map, labels_list



def iter_image_paths(root_dir, splits=("train", "test")):
  
    items = []
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue

        
        for cls in sorted(os.listdir(split_dir), key=lambda x: int(x) if x.isdigit() else x):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(IMG_EXTS):
                    items.append((split, cls, os.path.join(cls_dir, fn)))

    return items



def plot_skip_rate(per_class, out_path, title_suffix="", class_labels=None):
  
    if not per_class:
        return

    class_ids = sorted(list(per_class.keys()), key=lambda x: int(x) if str(x).isdigit() else x)

    used = np.array([per_class[c]["used"] for c in class_ids], dtype=float)
    skipped = np.array([per_class[c]["skipped"] for c in class_ids], dtype=float)
    total = used + skipped
    skip_rate = np.divide(skipped, np.maximum(total, 1.0))

    tick_labels = []
    for c in class_ids:
        if str(c).isdigit() and class_labels is not None:
            tick_labels.append(class_labels.get(int(c), str(c)))
        else:
            tick_labels.append(str(c))

    plt.figure(figsize=(14, 5))
    plt.plot(range(len(class_ids)), skip_rate)

    plt.xticks(range(len(class_ids)), tick_labels, rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("Skip rate (MediaPipe failed / total)")
    plt.xlabel("Class (BdSL label)")
    title = "MediaPipe skip rate per class"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def audit_mediapipe(
    dataset_dir,
    results_dir,
    min_det_conf=MIN_DET_CONF,
    min_track_conf=MIN_TRACK_CONF,
    static_image_mode=STATIC_IMAGE_MODE,
    max_num_hands=MAX_NUM_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    max_items=MAX_ITEMS,
    print_every=PRINT_EVERY,
    save_failure_examples_subdir=SAVE_FAILURE_EXAMPLES_SUBDIR,
    max_failure_examples=MAX_FAILURE_EXAMPLES,
):
    os.makedirs(results_dir, exist_ok=True)
    out_fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(out_fig_dir, exist_ok=True)

    
    class_map = {}
    try:
        _set_bengali_font(BENGALI_FONT_PATH)
        class_map, _ = build_class_labels()
    except Exception as e:
        print(f"[WARN] Bengali font / class mapping not applied: {e}")
        class_map = _read_class_meaning(CLASS_MEANING_PATH) if os.path.exists(CLASS_MEANING_PATH) else {}

    items = iter_image_paths(dataset_dir, splits=("train", "test"))
    if not items:
        raise RuntimeError(
            f"No images found under {dataset_dir}. Expected train/test/<class_id>/*.jpg"
        )

    if max_items is not None:
        items = items[: int(max_items)]

    
    per_split_class = {
        "train": defaultdict(lambda: {"used": 0, "skipped": 0}),
        "test": defaultdict(lambda: {"used": 0, "skipped": 0}),
    }
    per_split_reason = {
        "train": Counter(),
        "test": Counter(),
    }

    detail_rows = []

    
    save_fail_dir = None
    if save_failure_examples_subdir:
        save_fail_dir = os.path.join(results_dir, save_failure_examples_subdir)
        os.makedirs(save_fail_dir, exist_ok=True)

    saved_failures = 0

    t0 = time.time()
    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    ) as hands:

        for idx, (split, cls, path) in enumerate(items, start=1):
            if print_every and (idx % print_every == 0):
                dt = time.time() - t0
                print(f"[{idx}/{len(items)}] elapsed {dt:.1f}s")

            reason = None
            ok = False

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                reason = "read_fail"
                ok = False
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                try:
                    res = hands.process(img_rgb)
                    if res.multi_hand_landmarks and len(res.multi_hand_landmarks) > 0:
                        ok = True
                    else:
                        ok = False
                        reason = "no_hand_detected"
                except Exception:
                    ok = False
                    reason = "mediapipe_exception"

            if ok:
                per_split_class[split][cls]["used"] += 1
            else:
                per_split_class[split][cls]["skipped"] += 1
                per_split_reason[split][reason or "unknown"] += 1

                if save_fail_dir and saved_failures < int(max_failure_examples):
                    cls_int = int(cls) if str(cls).isdigit() else None
                    cls_label = class_map.get(cls_int, str(cls)) if cls_int is not None else str(cls)
                    base = os.path.basename(path)
                    out_name = f"{split}_c{cls}_{cls_label}_{reason or 'unknown'}_{base}"
                    out_path = os.path.join(save_fail_dir, out_name)

                    
                    try:
                        if img_bgr is not None:
                            cv2.imwrite(out_path, img_bgr)
                            saved_failures += 1
                    except Exception:
                        pass

            cls_int = int(cls) if str(cls).isdigit() else None
            cls_label = class_map.get(cls_int, str(cls)) if cls_int is not None else str(cls)

            detail_rows.append({
                "split": split,
                "class_id": cls,
                "class_label": cls_label,
                "path": path,
                "ok": int(ok),
                "reason": reason or "",
            })

    
    detail_csv = os.path.join(results_dir, "mediapipe_audit_details.csv")
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")

    
    summary_rows = []
    for split in ("train", "test"):
        for cls in sorted(per_split_class[split].keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            used = per_split_class[split][cls]["used"]
            skipped = per_split_class[split][cls]["skipped"]
            total = used + skipped

            cls_int = int(cls) if str(cls).isdigit() else None
            cls_label = class_map.get(cls_int, str(cls)) if cls_int is not None else str(cls)

            summary_rows.append({
                "split": split,
                "class_id": cls,
                "class_label": cls_label,
                "total": total,
                "used": used,
                "skipped": skipped,
                "skip_rate": (skipped / total) if total else 0.0,
            })

    summary_csv = os.path.join(results_dir, "mediapipe_skip_summary_by_class.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    
    reason_rows = []
    for split in ("train", "test"):
        for reason, cnt in per_split_reason[split].most_common():
            reason_rows.append({"split": split, "reason": reason, "count": cnt})
    reason_csv = os.path.join(results_dir, "mediapipe_skip_reasons.csv")
    pd.DataFrame(reason_rows).to_csv(reason_csv, index=False, encoding="utf-8-sig")

    plot_skip_rate(
        per_split_class["train"],
        os.path.join(out_fig_dir, "mediapipe_skip_rate_per_class_train.png"),
        title_suffix="(train)",
        class_labels=class_map if class_map else None,
    )
    plot_skip_rate(
        per_split_class["test"],
        os.path.join(out_fig_dir, "mediapipe_skip_rate_per_class_test.png"),
        title_suffix="(test)",
        class_labels=class_map if class_map else None,
    )

  
    def _split_totals(split):
        used = sum(per_split_class[split][c]["used"] for c in per_split_class[split])
        skipped = sum(per_split_class[split][c]["skipped"] for c in per_split_class[split])
        total = used + skipped
        return total, used, skipped, (skipped / total) if total else 0.0

    for split in ("train", "test"):
        total, used, skipped, rate = _split_totals(split)
        print(f"[{split}] total={total} used={used} skipped={skipped} skip_rate={rate:.4f}")
        if per_split_reason[split]:
            print(f"[{split}] top reasons:", per_split_reason[split].most_common(5))

    print("\nSaved:")
    print(" -", detail_csv)
    print(" -", summary_csv)
    print(" -", reason_csv)
    print(" -", os.path.join(out_fig_dir, "mediapipe_skip_rate_per_class_train.png"))
    print(" -", os.path.join(out_fig_dir, "mediapipe_skip_rate_per_class_test.png"))
    if save_fail_dir:
        print(" - failure examples in:", save_fail_dir)


if __name__ == "__main__":
    audit_mediapipe(
        dataset_dir=DATASET_DIR,
        results_dir=RESULTS_DIR,

        min_det_conf=MIN_DET_CONF,
        min_track_conf=MIN_TRACK_CONF,
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,

        max_items=MAX_ITEMS,
        print_every=PRINT_EVERY,

        save_failure_examples_subdir=SAVE_FAILURE_EXAMPLES_SUBDIR,
        max_failure_examples=MAX_FAILURE_EXAMPLES,
    )
