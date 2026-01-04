import os

DATASET_DIR = r"D:\BdSL\dataset_224"

def count_split(split):
    total = 0
    per_class = {}
    for cls in sorted(os.listdir(os.path.join(DATASET_DIR, split))):
        cls_dir = os.path.join(DATASET_DIR, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])
        per_class[int(cls)] = n
        total += n
    return total, per_class

train_total, train_per_class = count_split("train")
test_total, test_per_class = count_split("test")

print(f"Training images: {train_total}")
print(f"Testing images : {test_total}")
print(f"Total images   : {train_total + test_total}")
print(f"Train %        : {100 * train_total / (train_total + test_total):.2f}%")
print(f"Test %         : {100 * test_total / (train_total + test_total):.2f}%")
