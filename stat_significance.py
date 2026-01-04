import os
import math
import numpy as np
import pandas as pd

# User Settings (EDIT THESE)
RESULTS_ROOT = r"D:\BdSL\results"   # contains subfolders below
OUT_DIR = os.path.join(RESULTS_ROOT, "significance")

# folders inside RESULTS_ROOT that contain y_true.npy / y_pred.npy
MODELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "vit_small": "ViT-small",
    "mobilenetv3": "MobileNetV3",
    "mediapipe_svm": "MediaPipe + SVM",
}

# expected filenames (in each model folder)
Y_TRUE_NAME = "y_true.npy"
Y_PRED_NAME = "y_pred.npy"

USED_INDICES_TEST_NAME = "used_indices_test.npy"   # indices into the FULL test ordering

# Sentinel used for "no prediction" (for MediaPipe failures)
INVALID_PRED = -1

# Bootstrap settings
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42


# Helpers


def _as_1d_int(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1).astype(np.int64)

def accuracy(y_true, y_pred):
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")

def per_class_accuracy(y_true, y_pred, num_classes=None):
   
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)

    if num_classes is None:
        k = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    else:
        k = int(num_classes)

    accs = np.full((k,), np.nan, dtype=np.float64)
    for c in range(k):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            continue
        accs[c] = float(np.mean(y_pred[idx] == c))
    return accs

def mcnemar_exact(y_true, y_a, y_b):
  
    y_true = _as_1d_int(y_true)
    y_a = _as_1d_int(y_a)
    y_b = _as_1d_int(y_b)

    a_correct = (y_a == y_true)
    b_correct = (y_b == y_true)

    b = int(np.sum(a_correct & (~b_correct)))
    c = int(np.sum((~a_correct) & b_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0

    k = min(b, c)

    denom = 2 ** n
    cum = 0.0
    for i in range(0, k + 1):
        cum += math.comb(n, i) / denom
    p = min(1.0, 2.0 * cum)
    return b, c, p

def paired_bootstrap_acc_diff(y_true, y_a, y_b, B=5000, seed=42):
    
    rng = np.random.default_rng(seed)
    y_true = _as_1d_int(y_true)
    y_a = _as_1d_int(y_a)
    y_b = _as_1d_int(y_b)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    idx = np.arange(n)
    diffs = np.empty((B,), dtype=np.float64)
    for b in range(B):
        bs = rng.choice(idx, size=n, replace=True)
        diffs[b] = np.mean(y_a[bs] == y_true[bs]) - np.mean(y_b[bs] == y_true[bs])

    point = float(np.mean(y_a == y_true) - np.mean(y_b == y_true))
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return point, lo, hi

def paired_ttest_over_classes(y_true, y_a, y_b, num_classes=49):
  
    acc_a = per_class_accuracy(y_true, y_a, num_classes=num_classes)
    acc_b = per_class_accuracy(y_true, y_b, num_classes=num_classes)

    mask = ~np.isnan(acc_a) & ~np.isnan(acc_b)
    da = acc_a[mask]
    db = acc_b[mask]
    if da.size < 2:
        return 1.0, int(da.size)

    d = da - db
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return 1.0, int(da.size)

    t = mean / (sd / math.sqrt(d.size))
    try:
        from scipy import stats  
        p = float(2 * stats.t.sf(abs(t), df=d.size - 1))
    except Exception:
        p = float(2 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2)))))
    return p, int(da.size)

def holm_correction(pvals):
 
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        adj = min(1.0, max(adj, prev))
        adjusted[i] = adj
        prev = adjusted[i]
    return adjusted


def load_model_arrays(model_folder):
   
    y_true_path = os.path.join(model_folder, Y_TRUE_NAME)
    y_pred_path = os.path.join(model_folder, Y_PRED_NAME)
    if not (os.path.isfile(y_true_path) and os.path.isfile(y_pred_path)):
        return None

    y_true = _as_1d_int(np.load(y_true_path, allow_pickle=False))
    y_pred = _as_1d_int(np.load(y_pred_path, allow_pickle=False))

    used_idx_path = os.path.join(model_folder, USED_INDICES_TEST_NAME)
    used_indices = None
    if os.path.isfile(used_idx_path):
        used_indices = _as_1d_int(np.load(used_idx_path, allow_pickle=False))

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "used_indices": used_indices
    }


def build_paired_arrays(A, B, y_true_ref):
  
    if len(A["y_true"]) == len(B["y_true"]) and np.array_equal(A["y_true"], B["y_true"]):
        y_true_s = A["y_true"]
        y_a_s = A["y_pred"]
        y_b_s = B["y_pred"]
        mask = (y_a_s != INVALID_PRED) & (y_b_s != INVALID_PRED)
        return y_true_s[mask], y_a_s[mask], y_b_s[mask]

    idxA = A.get("used_indices", None)
    idxB = B.get("used_indices", None)

   
    if idxA is not None or idxB is not None:
       
        if idxA is not None and idxB is not None:
            setA = set(map(int, idxA))
            setB = set(map(int, idxB))
            common = sorted(setA.intersection(setB))
            if len(common) == 0:
                return None

            posA = {int(full_i): pos for pos, full_i in enumerate(idxA)}
            posB = {int(full_i): pos for pos, full_i in enumerate(idxB)}

            y_true_s = y_true_ref[np.array(common, dtype=np.int64)]
            y_a_s = A["y_pred"][np.array([posA[i] for i in common], dtype=np.int64)]
            y_b_s = B["y_pred"][np.array([posB[i] for i in common], dtype=np.int64)]

            mask = (y_a_s != INVALID_PRED) & (y_b_s != INVALID_PRED)
            return y_true_s[mask], y_a_s[mask], y_b_s[mask]

        if idxA is not None and idxB is None:
            idx = idxA.astype(np.int64)
            if len(B["y_true"]) != len(y_true_ref) or not np.array_equal(B["y_true"], y_true_ref):
                return None
            y_true_s = y_true_ref[idx]
            y_a_s = A["y_pred"]
            y_b_s = B["y_pred"][idx]

            mask = (y_a_s != INVALID_PRED) & (y_b_s != INVALID_PRED)
            return y_true_s[mask], y_a_s[mask], y_b_s[mask]

        if idxA is None and idxB is not None:
            idx = idxB.astype(np.int64)
            if len(A["y_true"]) != len(y_true_ref) or not np.array_equal(A["y_true"], y_true_ref):
                return None
            y_true_s = y_true_ref[idx]
            y_a_s = A["y_pred"][idx]
            y_b_s = B["y_pred"]

            mask = (y_a_s != INVALID_PRED) & (y_b_s != INVALID_PRED)
            return y_true_s[mask], y_a_s[mask], y_b_s[mask]

        return None

    can_pair = (
        len(A["y_true"]) == len(y_true_ref) and
        len(B["y_true"]) == len(y_true_ref) and
        np.array_equal(A["y_true"], y_true_ref) and
        np.array_equal(B["y_true"], y_true_ref)
    )
    if not can_pair:
        return None

    y_true_s = y_true_ref
    y_a_s = A["y_pred"]
    y_b_s = B["y_pred"]
    mask = (y_a_s != INVALID_PRED) & (y_b_s != INVALID_PRED)
    return y_true_s[mask], y_a_s[mask], y_b_s[mask]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    loaded = {}
    for folder, name in MODELS.items():
        model_dir = os.path.join(RESULTS_ROOT, folder)
        blob = load_model_arrays(model_dir)
        if blob is None:
            print(f"[MISS] {folder}: missing {Y_TRUE_NAME}/{Y_PRED_NAME} in {model_dir}")
            continue

        y_true = blob["y_true"]
        y_pred = blob["y_pred"]
        used_indices = blob["used_indices"]

        loaded[folder] = {
            "display": name,
            "dir": model_dir,
            "y_true": y_true,
            "y_pred": y_pred,
            "used_indices": used_indices,
            "n": len(y_true),
        }

        if used_indices is not None:
            print(f"[OK] {folder}: subset={len(y_true)} with {USED_INDICES_TEST_NAME}")
        else:
            used = int(np.sum(y_pred != INVALID_PRED)) if np.any(y_pred == INVALID_PRED) else len(y_pred)
            print(f"[OK] {folder}: {len(y_true)} samples, used={used}")

    keys = list(loaded.keys())
    rows = []
    mcnemar_pvals_for_holm = []
    holm_row_indices = []

    ref_key = None
    for k in ["efficientnet_b0", "vit_small", "mobilenetv3"]:
        if k in loaded and loaded[k]["used_indices"] is None:
            ref_key = k
            break
    if ref_key is None:
        ref_key = keys[0] if keys else None

    if ref_key is None:
        print("No models loaded. Exiting.")
        return

    y_true_ref = loaded[ref_key]["y_true"]

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            A = loaded[a]
            B = loaded[b]

            paired = build_paired_arrays(A, B, y_true_ref)
            if paired is None:
                print(f"[SKIP] {a} vs {b} (cannot build paired set)")
                continue

            y_true_s, y_a_s, y_b_s = paired
            n_pair = int(len(y_true_s))
            if n_pair == 0:
                print(f"[SKIP] {a} vs {b} (no paired samples after filtering)")
                continue

            acc_a = accuracy(y_true_s, y_a_s)
            acc_b = accuracy(y_true_s, y_b_s)
            diff, ci_lo, ci_hi = paired_bootstrap_acc_diff(
                y_true_s, y_a_s, y_b_s, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED
            )
            b01, c10, p_mcn = mcnemar_exact(y_true_s, y_a_s, y_b_s)
            p_t, k_used = paired_ttest_over_classes(y_true_s, y_a_s, y_b_s, num_classes=49)

            rows.append({
                "model_a": a,
                "model_b": b,
                "name_a": A["display"],
                "name_b": B["display"],
                "N_paired": n_pair,
                "acc_a": acc_a,
                "acc_b": acc_b,
                "acc_diff_a_minus_b": diff,
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
                "mcnemar_b(A correct,B wrong)": b01,
                "mcnemar_c(A wrong,B correct)": c10,
                "mcnemar_p": p_mcn,
                "ttest_p_per_class_acc": p_t,
                "ttest_classes_used": k_used,
            })

            mcnemar_pvals_for_holm.append(p_mcn)
            holm_row_indices.append(len(rows) - 1)

    if mcnemar_pvals_for_holm:
        adj = holm_correction(mcnemar_pvals_for_holm)
        for idx, p_adj in zip(holm_row_indices, adj):
            rows[idx]["mcnemar_p_holm"] = float(p_adj)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "pairwise_significance.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== DONE ===")
    print(f"Saved: {out_csv}")
    print("\nNote:")
    print("- Deep vs Deep uses full test set (N=2940) if y_true matches.")
    print("- Deep vs MediaPipe uses used_indices_test.npy to slice Deep into MediaPipe subset.")
    print("- Holm correction applied on McNemar p-values")


if __name__ == "__main__":
    main()
