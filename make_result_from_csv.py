from __future__ import annotations
import os
import re
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
from matplotlib.ticker import MaxNLocator

#User settings (EDIT THESE)

RESULTS_ROOT = r"D:\BdSL\results"                 # contains: efficientnet_b0, vit_small, mobilenetv3, mediapipe_svm
CLASS_MEANING_PATH = r"D:\BdSL\results\class_meaning.md"  # your mapping file
OUT_DIR = os.path.join(RESULTS_ROOT, "figures")           # your output figures folder (change for different models)

# "auto" picks the model with best accuracy from metrics.csv
# or set explicitly: "efficientnet_b0" / "vit_small" / "mobilenetv3" / "mediapipe_svm"
BEST_MODEL_OVERRIDE = "auto"

# Force Bengali font (change path to your own TTF)
BENGALI_FONT_PATH = r"C:\Users\steve\Downloads\Noto_Sans_Bengali\NotoSansBengali-VariableFont_wdth,wght.ttf"


# Internals
MODEL_FOLDERS = {
    "efficientnet_b0": "EfficientNet-B0",
    "vit_small": "ViT-small",
    "mobilenetv3": "MobileNetV3",
    "mediapipe_svm": "MediaPipe + SVM",
}

def _set_bengali_font(font_path: str = BENGALI_FONT_PATH) -> FontProperties:
 
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Bengali font not found at: {font_path}")

    fm.fontManager.addfont(font_path)

    font_prop = FontProperties(fname=font_path)

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    return font_prop

def _read_class_meaning(path: str) -> Dict[int, str]:
 
    seen: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            left, right = [x.strip() for x in line.split("=", 1)]
            if re.fullmatch(r"\d+", left):
                seen.append((int(left), right))

    mapping: Dict[int, str] = {}
    dups: Dict[int, List[str]] = {}
    for k, v in seen:
        if k in mapping:
            dups.setdefault(k, [mapping[k]]).append(v)
        mapping[k] = v

    for k, _vals in dups.items():
        if (k - 1) not in mapping:
            occurrences = [v for kk, v in seen if kk == k]
            if len(occurrences) >= 2:
                mapping[k - 1] = occurrences[0]
                mapping[k] = occurrences[1]

    return mapping

def _load_metrics(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) != 1:
        df = df.tail(1)
    return df.iloc[0]

def _load_cm(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    cm = df.values
    if cm.dtype == object:
        cm = pd.read_csv(path, header=None).values
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got {cm.shape} from {path}")
    return cm

def _normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cm / np.maximum(row_sums, 1)

def _per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1)
    diag = np.diag(cm)
    return np.divide(diag, np.maximum(row_sums, 1), dtype=float)

def _pick_best_model(metrics_by_key: Dict[str, pd.Series]) -> str:
    if BEST_MODEL_OVERRIDE != "auto":
        return BEST_MODEL_OVERRIDE
    best_key, best_acc = None, -1.0
    for k, s in metrics_by_key.items():
        acc = float(s.get("accuracy", np.nan))
        if not math.isnan(acc) and acc > best_acc:
            best_acc = acc
            best_key = k
    return best_key or "mobilenetv3"

def _save_fig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix_normalized(cm: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    cmn = _normalize_rows(cm)
    plt.figure(figsize=(16, 14))
    im = plt.imshow(cmn, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title, fontsize=16, pad=12)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    n = cm.shape[0]
    ticks = np.arange(n)
    plt.xticks(ticks, labels, rotation=90, fontsize=7)
    plt.yticks(ticks, labels, fontsize=7)

    plt.gca().set_xticks(ticks - 0.5, minor=True)
    plt.gca().set_yticks(ticks - 0.5, minor=True)
    plt.grid(which="minor", linestyle="-", linewidth=0.2)
    plt.tick_params(which="minor", bottom=False, left=False)

    _save_fig(out_path)

def plot_per_class_accuracy(acc: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    idx = np.arange(len(acc))
    plt.figure(figsize=(18, 5))
    plt.bar(idx, acc)
    plt.gca().yaxis.set_major_locator(MaxNLocator(6))
    plt.ylim(0, 1.0)
    plt.title(title, fontsize=16, pad=10)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.xticks(idx, labels, rotation=90, fontsize=8)
    plt.tight_layout()
    _save_fig(out_path)

def plot_worst10(acc: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    order = np.argsort(acc)[:10]
    vals = acc[order]
    labs = [labels[i] for i in order]

    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(vals)), vals)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.0)
    plt.title(title, fontsize=16, pad=10)
    plt.xlabel("Accuracy")
    plt.yticks(np.arange(len(vals)), labs, fontsize=11)
    plt.tight_layout()
    _save_fig(out_path)

def plot_top10_confusions(cm: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    cmn = _normalize_rows(cm).copy()
    np.fill_diagonal(cmn, 0.0)

    pairs: List[Tuple[int, int, float]] = []
    n = cmn.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                v = float(cmn[i, j])
                if v > 0:
                    pairs.append((i, j, v))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:10] if len(pairs) >= 10 else pairs

    names = [f"{labels[i]} as {labels[j]}" for i, j, _ in top]
    vals = [v for _, _, v in top]

    plt.figure(figsize=(12, 6))
    plt.barh(np.arange(len(vals)), vals)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=16, pad=10)
    plt.xlabel("Confusion rate (row-normalized)")
    plt.yticks(np.arange(len(vals)), names, fontsize=11)
    plt.tight_layout()
    _save_fig(out_path)

def plot_unified_figure(cm: np.ndarray, labels: List[str], model_title: str, out_path: str) -> None:
    acc = _per_class_accuracy(cm)
    cmn = _normalize_rows(cm)

    cmn_off = cmn.copy()
    np.fill_diagonal(cmn_off, 0.0)
    pairs = []
    n = cmn_off.shape[0]
    for i in range(n):
        for j in range(n):
            v = float(cmn_off[i, j])
            if v > 0:
                pairs.append((i, j, v))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:10] if len(pairs) >= 10 else pairs
    names = [f"{labels[i]}as{labels[j]}" for i, j, _ in top]
    vals = [v for _, _, v in top]

    worst_idx = np.argsort(acc)[:10]
    worst_vals = acc[worst_idx]
    worst_labels = [labels[i] for i in worst_idx]

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0], height_ratios=[1.2, 0.8])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(cmn, interpolation="nearest")
    ax0.set_title(f"{model_title}: Normalized Confusion Matrix", fontsize=14, pad=10)
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ticks = np.arange(n)
    ax0.set_xticks(ticks)
    ax0.set_yticks(ticks)
    ax0.set_xticklabels(labels, rotation=90, fontsize=6)
    ax0.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(np.arange(len(acc)), acc)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Per-class Accuracy", fontsize=14, pad=10)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(np.arange(n))
    ax1.set_xticklabels(labels, rotation=90, fontsize=6)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.barh(np.arange(len(worst_vals)), worst_vals)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    ax2.set_title("Worst-10 Classes (Accuracy)", fontsize=14, pad=10)
    ax2.set_xlabel("Accuracy")
    ax2.set_yticks(np.arange(len(worst_vals)))
    ax2.set_yticklabels(worst_labels, fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.barh(np.arange(len(vals)), vals)
    ax3.invert_yaxis()
    ax3.set_title("Top-10 Confusions", fontsize=14, pad=10)
    ax3.set_xlabel("Confusion rate")
    ax3.set_yticks(np.arange(len(vals)))
    ax3.set_yticklabels(names, fontsize=10)

    fig.suptitle("BdSL-49 Evaluation Summary (Best Model)", fontsize=18, y=1.02)
    fig.tight_layout()
    _save_fig(out_path)

def plot_model_comparison_table(rows: List[Dict[str, object]], out_path: str) -> None:
    df = pd.DataFrame(rows)

    col_order = ["Model", "Accuracy", "Precision", "Recall", "F1_macro", "Avg_loss",
                 "Train_time_s", "Infer_time_s", "Notes"]
    for c in col_order:
        if c not in df.columns:
            df[c] = ""
    df = df[col_order]

    def fmt(x, nd=4):
        try:
            if x == "" or pd.isna(x):
                return ""
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    df_fmt = df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1_macro", "Avg_loss"]:
        df_fmt[c] = df_fmt[c].apply(lambda v: fmt(v, 4))
    for c in ["Train_time_s", "Infer_time_s"]:
        df_fmt[c] = df_fmt[c].apply(lambda v: fmt(v, 1))

    plt.figure(figsize=(14, 2 + 0.5 * len(df_fmt)))
    plt.axis("off")
    tbl = plt.table(cellText=df_fmt.values, colLabels=df_fmt.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.title("Model Comparison on BdSL-49 (dataset_224)", fontsize=16, pad=16)
    _save_fig(out_path)

def main() -> None:
    font_prop = _set_bengali_font(BENGALI_FONT_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)

    class_map = _read_class_meaning(CLASS_MEANING_PATH)
    n_classes = 49
    labels = [class_map.get(i, str(i)) for i in range(n_classes)]

    metrics_by_key: Dict[str, pd.Series] = {}
    cm_by_key: Dict[str, np.ndarray] = {}
    rows_for_table: List[Dict[str, object]] = []

    for folder_key, display_name in MODEL_FOLDERS.items():
        model_dir = os.path.join(RESULTS_ROOT, folder_key)
        metrics_path = os.path.join(model_dir, "metrics.csv")
        cm_path = os.path.join(model_dir, "confusion_matrix.csv")

        if not (os.path.exists(metrics_path) and os.path.exists(cm_path)):
            continue

        s = _load_metrics(metrics_path)
        cm = _load_cm(cm_path)

        metrics_by_key[folder_key] = s
        cm_by_key[folder_key] = cm

        precision = s.get("precision_macro", s.get("precision", ""))
        recall = s.get("recall_macro", s.get("recall", ""))
        f1 = s.get("f1_macro", s.get("f1", ""))

        infer_time = s.get("test_inference_time_sec", s.get("inference_time_sec", s.get("inference_time_s", "")))

        row = {
            "Model": display_name,
            "Accuracy": s.get("accuracy", ""),
            "Precision": precision,
            "Recall": recall,
            "F1_macro": f1,
            "Avg_loss": s.get("avg_loss", ""), 
            "Train_time_s": s.get("train_time_sec", s.get("train_time_s", "")),
            "Infer_time_s": infer_time,
            "Notes": "",
        }

        if folder_key == "mediapipe_svm":
            stats_path = os.path.join(model_dir, "extraction_stats.csv")
            if os.path.exists(stats_path):
                st = pd.read_csv(stats_path).iloc[0]
                row["Notes"] = (
                    f"Train used {int(st.get('train_used',0))}, skipped {int(st.get('train_skipped',0))}; "
                    f"Test used {int(st.get('test_used',0))}, skipped {int(st.get('test_skipped',0))}"
                )
            else:
                row["Notes"] = "Landmark detection can skip images → not fully comparable"

        rows_for_table.append(row)

    if not metrics_by_key:
        raise RuntimeError(
            "No results found.\n"
            "Expected e.g. D:\\BdSL\\results\\mobilenetv3\\metrics.csv and confusion_matrix.csv\n"
            "Update RESULTS_ROOT if needed."
        )

    best_key = _pick_best_model(metrics_by_key)
    best_name = MODEL_FOLDERS.get(best_key, best_key)
    best_cm = cm_by_key[best_key]

    plot_confusion_matrix_normalized(
        best_cm, labels,
        title=f"Normalized Confusion Matrix — {best_name} (BdSL-49)",
        out_path=os.path.join(OUT_DIR, "confusion_matrix_normalized.png"),
    )
    acc = _per_class_accuracy(best_cm)
    plot_per_class_accuracy(
        acc, labels,
        title=f"Per-class Accuracy — {best_name} (BdSL-49)",
        out_path=os.path.join(OUT_DIR, "per_class_accuracy.png"),
    )
    plot_worst10(
        acc, labels,
        title=f"Worst-10 Class Accuracy — {best_name} (BdSL-49)",
        out_path=os.path.join(OUT_DIR, "worst10_class_accuracy.png"),
    )
    plot_top10_confusions(
        best_cm, labels,
        title=f"Top-10 Confusions — {best_name} (BdSL-49)",
        out_path=os.path.join(OUT_DIR, "top10_confusions.png"),
    )
    plot_unified_figure(
        best_cm, labels,
        model_title=best_name,
        out_path=os.path.join(OUT_DIR, "unified_figure.png"),
    )

    plot_model_comparison_table(
        rows_for_table,
        out_path=os.path.join(OUT_DIR, "model_comparison_table.png"),
    )
    pd.DataFrame(rows_for_table).to_csv(os.path.join(OUT_DIR, "model_comparison_table.csv"), index=False)

    print(f"Saved figures to: {OUT_DIR}")
    print(f"Best model picked: {best_key} ({best_name})")

if __name__ == "__main__":
    main()
