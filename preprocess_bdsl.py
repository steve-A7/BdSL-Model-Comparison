import os
import cv2
from tqdm import tqdm

SRC_DIR = r"D:\BdSL"
OUT_DIR = r"D:\BdSL\dataset_224"
IMG_SIZE = 224

sources = [
    ("Recognition_1", range(0, 23)),
    ("Recognition_2", range(23, 49))
]

for split in ["train", "test"]:
    for cls in range(49):
        os.makedirs(os.path.join(OUT_DIR, split, str(cls)), exist_ok=True)

for folder, class_range in sources:
    for split in ["train", "test"]:
        for cls in class_range:
            src_cls_dir = os.path.join(SRC_DIR, folder, split, str(cls))
            dst_cls_dir = os.path.join(OUT_DIR, split, str(cls))

            if not os.path.exists(src_cls_dir):
                continue

            images = os.listdir(src_cls_dir)

            for img_name in tqdm(images, desc=f"{folder}/{split}/{cls}"):
                src_path = os.path.join(src_cls_dir, img_name)
                img = cv2.imread(src_path)

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(
                    os.path.join(dst_cls_dir, img_name),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
