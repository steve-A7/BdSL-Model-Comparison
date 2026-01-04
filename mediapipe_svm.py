import os
import time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from mediapipe.python.solutions import hands as mp_hands
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# User Settings (EDIT THESE)
DATASET_DIR = r"D:\BdSL\dataset_224"
RESULTS_DIR = r"D:\BdSL\results\mediapipe_svm"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 49
SEED = 42


def extract_landmarks_bgr(img_bgr, hands_model):
    
   
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_model.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    
    hand = results.multi_hand_landmarks[0]
    feat = []
    for lm in hand.landmark:
        feat.extend([lm.x, lm.y, lm.z])
    return np.array(feat, dtype=np.float32)

def load_split(split):
 
    X, y = [], []
    used = 0
    skipped = 0

    used_indices = []
    used_mask = []

    full_index = 0  

    split_dir = os.path.join(DATASET_DIR, split)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands:

        for cls in range(NUM_CLASSES):
            cls_dir = os.path.join(split_dir, str(cls))
            if not os.path.isdir(cls_dir):
                continue

            files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png"))]
            files.sort()

            for fname in tqdm(files, desc=f"{split} class {cls}"):
                fpath = os.path.join(cls_dir, fname)

                img = cv2.imread(fpath)
                if img is None:
                    skipped += 1
                    used_mask.append(False)
                    full_index += 1
                    continue

                feat = extract_landmarks_bgr(img, hands)
                if feat is None:
                    skipped += 1
                    used_mask.append(False)
                    full_index += 1
                    continue

                
                X.append(feat)
                y.append(cls)

                used_indices.append(full_index)
                used_mask.append(True)

                used += 1
                full_index += 1

    return (
        np.array(X),
        np.array(y),
        used,
        skipped,
        np.array(used_indices, dtype=np.int64),
        np.array(used_mask, dtype=np.bool_)
    )


#Load data (feature extraction)
t0 = time.time()
X_train, y_train, used_train, skipped_train, used_indices_train, used_mask_train = load_split("train")
X_test,  y_test,  used_test,  skipped_test,  used_indices_test,  used_mask_test  = load_split("test")
feature_time_sec = time.time() - t0

print("\n=== MediaPipe extraction summary ===")
print(f"Train used: {used_train}, skipped: {skipped_train}")
print(f"Test  used: {used_test}, skipped: {skipped_test}")
print(f"Feature extraction time (sec): {feature_time_sec:.1f}")

# Save extraction stats
pd.DataFrame([{
    "train_used": used_train,
    "train_skipped": skipped_train,
    "test_used": used_test,
    "test_skipped": skipped_test,
    "feature_extraction_time_sec": feature_time_sec
}]).to_csv(os.path.join(RESULTS_DIR, "extraction_stats.csv"), index=False)

#Train SVM
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=SEED))
])

t1 = time.time()
svm.fit(X_train, y_train)
train_time_sec = time.time() - t1

#Evaluate
t2 = time.time()
y_pred = svm.predict(X_test)
infer_time_sec = time.time() - t2

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== MediaPipe + SVM Test Metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"SVM train time (sec): {train_time_sec:.1f}")
print(f"SVM inference time on test (sec): {infer_time_sec:.1f}")

#Save results
pd.DataFrame({
    "model": ["mediapipe_hands + SVM(rbf)"],
    "train_used": [used_train],
    "test_used": [used_test],
    "feature_extraction_time_sec": [feature_time_sec],
    "train_time_sec": [train_time_sec],
    "test_inference_time_sec": [infer_time_sec],
    "accuracy": [acc],
    "precision_macro": [prec],
    "recall_macro": [rec],
    "f1_macro": [f1],
}).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"), index=False)

np.save(os.path.join(RESULTS_DIR, "y_true.npy"), y_test)
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), y_pred)

np.save(os.path.join(RESULTS_DIR, "used_indices_train.npy"), used_indices_train)
np.save(os.path.join(RESULTS_DIR, "used_mask_train.npy"), used_mask_train)

np.save(os.path.join(RESULTS_DIR, "used_indices_test.npy"), used_indices_test)
np.save(os.path.join(RESULTS_DIR, "used_mask_test.npy"), used_mask_test)

print("\n=== Alignment check ===")
print(f"Full train size (mask length): {len(used_mask_train)}")
print(f"Full test  size (mask length): {len(used_mask_test)}")
print(f"Train used_indices: {len(used_indices_train)} == used_train: {used_train}")
print(f"Test  used_indices: {len(used_indices_test)} == used_test : {used_test}")

