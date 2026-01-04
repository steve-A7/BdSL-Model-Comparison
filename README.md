# BdSL-Model-Comparison

This repository contains the code, scripts, and generated figures for the research paper:

**“Comparative Analysis between Deep Learning and Machine Learning Models for Bangla Sign Language (BdSL) Alphabet Recognition”**

It includes:

- Training + evaluation scripts for "EfficientNet-B0", "MobileNetV3-Large", "ViT-Small", and "MediaPipe Hands + SVM"
- Scripts to generate all "tables/figures" used in the paper except figure 6.
- Statistical significance testing (McNemar + bootstrap CI + paired t-test)
- MediaPipe failure/skip-rate auditing (limitations analysis)

1. Environment Setup (Anaconda, Python 3.10)

Install Anaconda
Download: https://www.anaconda.com/download

Create and activate environment:

conda create -n bdsl49 python=3.10 -y
conda activate bdsl49

Install dependencies:

pip install torch torchvision torchaudio
pip install timm scikit-learn pandas numpy matplotlib seaborn
pip install opencv-python albumentations tqdm scipy
pip install tensorboard wandb
pip install mediapipe==0.10.14

Note: The original experiments were run on Windows (CPU).
If you use GPU or a different OS, adjust configs inside each model script accordingly.

2. Dataset Download and Expected Structure

Download BdSL-49 dataset (Mendeley) : https://data.mendeley.com/datasets/k5yk4j8z8s/6

Expected structure (use Recognition folders)
After extracting, ensure the recognition folder format looks like:

Recognition_1/
train/
0/
0_37.png
...
1/
...
test/
0/
0_37.png
...
1/
...
You also have to format Recognition_2/ with the same structure.

3. Preprocessing (Create dataset_224)

Run the preprocessing script to merge recognition folders and resize images to 224×224:

python preprocess_bdsl.py

Edit paths inside the script to point to your extracted dataset location.

The output should be:

dataset_224/
train/0 ... 48
test/0 ... 48

4. Dataset Statistics (Optional)

Count images per split/class:

python count_images.py

5. Train + Test Models (Generate y_true/y_pred + reports)

Train and test each model script one by one:

EfficientNet-B0

python efficientnet_b0.py

MobileNetV3-Large

python mobilenetv3.py

ViT-Small

python vit_small.py

MediaPipe + SVM

python train_mediapipe_svm.py

Each model saves outputs under:

results/<model_name>/
y_true.npy
y_pred.npy
metrics.csv
confusion_matrix.csv
For MediaPipe evaluation on paired subset, it also saves:

results/mediapipe_svm/

used_indices_test.npy

6. Generate Tables and Figures (Paper Results)
   Generate result figures from CSV outputs

This script generates:

confusion matrices
per-class accuracy plots
top confusions / worst classes
summary tables/plots used in the Results section

python make_result_from_csv.py

Follow instructions inside the script to set:

1. input CSV paths

2. Bengali font path (for proper Bangla labels) (Bengali font recommended: Noto Sans Bengali, extract the zip and copy the path with .ttf)

3. output directories

7) Statistical Significance Tables (McNemar + Bootstrap + t-test)

This generates the significance table used in the paper:

McNemar exact test p-values (with Holm correction)
Paired bootstrap 95% CI for accuracy differences
Paired t-test over per-class accuracy (optional / appendix)

python stat_significance.py

output: results/significance/pairwise_significance.csv

8. MediaPipe Limitations Analysis (Skip-rate + failure reasons)

This script audits MediaPipe failures and generates:

per-class skip-rate plots (train/test)
failure reason counts
detailed per-image audit CSV

python mediapipe_failure_audit.py

Outputs:
results/mediapipe_failure_audit/
mediapipe_skip_summary_by_class.csv
mediapipe_skip_reasons.csv
mediapipe_audit_details.csv
figures/
mediapipe_skip_rate_per_class_train.png
mediapipe_skip_rate_per_class_test.png

9. Running Any Script

General usage:

python script_name.py

Notes:

The dataset is pre-cropped in recognition folders; this work focuses on static image alphabet recognition (not continuous video recognition).
For exact reproduction, keep:
identical train/test splits
fixed input size (224×224)
same MediaPipe version: 0.10.14

Citation
If you use this repository, please cite the corresponding paper.
