# Solar Panel Dust Detection with Explainable AI (XAI)

> Hybrid **EfficientNet-B2 + SVM** pipeline for clean-vs-dirty solar-panel
> classification, with five explainability methods (Grad-CAM, Score-CAM,
> Integrated Gradients, SHAP, LIME), zero-shot bounding boxes (Grounding DINO),
> and a live dashboard.

| Item | Value |
|------|-------|
| Task | Binary image classification (clean / dirty) |
| Best model | EfficientNet-B2 (frozen) + RBF-SVM head |
| Test accuracy | **90.29%** |
| AUC-ROC | **0.9596** |
| 5-fold CV | **86.38%** |
| External val. | 98.24% / 99.74% (two independent sets) |
| Dataset | 3,787 images (3,406 train+val / 381 test) |
| XAI methods | Grad-CAM, Score-CAM, Integrated Gradients, SHAP, LIME |
| Object detection | Grounding DINO-tiny (zero-shot dust localization) |

---

## Quick Links

| Resource | Link |
|----------|------|
| **Live App (Streamlit)** | [cuhm9yammazovzcpxuaj3t.streamlit.app](https://cuhm9yammazovzcpxuaj3t.streamlit.app) |
| **GitHub Repo** | [adityashirsatrao007/solar-panel-dust-detection-vgg-svm](https://github.com/adityashirsatrao007/solar-panel-dust-detection-vgg-svm) |
| **HF Model (SVM + scaler + meta)** | [adityashirsatrao007/solar-panel-dust-xai](https://huggingface.co/adityashirsatrao007/solar-panel-dust-xai) |
| **HF Dataset (merged 3,787 imgs)** | [adityashirsatrao007/solar-dust-data-combined](https://huggingface.co/datasets/adityashirsatrao007/solar-dust-data-combined) |
| **HF Dataset (3 raw sources)** | [adityashirsatrao007/solar-dust-datasets-raw](https://huggingface.co/datasets/adityashirsatrao007/solar-dust-datasets-raw) |
| **Flask Dashboard** | `http://localhost:5001` (local, systemd-managed) |
| **Paper (Overleaf)** | `~/Desktop/solar_dust_paper_overleaf.zip` |

---

## 1. Overview

Dust accumulation on photovoltaic (PV) panels reduces power output and accelerates
degradation. This project detects dust from a single RGB image using transfer
learning: a frozen **EfficientNet-B2** backbone extracts 1408-d GAP features, and a
linear **SVM** head classifies them. The same features feed five XAI techniques so
predictions are human-interpretable — essential for field deployment and trust.

The Streamlit app adds **Grounding DINO** zero-shot object detection to draw
bounding boxes around individual dirty spots (bird droppings, mud stains, debris).

Two training phases are supported:
- **Phase 1 (frozen):** extract features → train SVM head (default, fast).
- **Phase 2 (fine-tune):** unfreeze last 30% of the backbone → re-extract → retrain SVM.

A comparison across three backbones (EfficientNet-B2, MobileNetV3-Large, ResNet50)
is reproducible via the `--backbone` flag (see §6).

---

## 2. Features

- Frozen transfer-learning feature extractor (ImageNet weights)
- SVM head with 5-fold CV grid search (C / gamma)
- Five XAI explainers for localization & channel importance
- **Zero-shot bounding boxes** via Grounding DINO (detects dust/dirt/debris without training)
- Live Streamlit dashboard with classification + Grad-CAM heatmap + bounding boxes
- Flask REST API (`/analyze`, `/explain`, `/health`)
- Confusion matrix, ROC, confidence, and loss/accuracy figures
- Backbone ablation table (EfficientNet-B2 / MobileNetV3 / ResNet50)
- Model versioning (`Models/model_versions.json`)

---

## 3. Dataset

| Split | Clean | Dirty | Total |
|-------|-------|-------|-------|
| Train | 1,750 | 1,279 | 3,029 |
| Val | 218 | 159 | 377 |
| Test | 220 | 161 | 381 |
| **All** | 2,188 | 1,599 | **3,787** |

- Source: a **merged multi-source corpus** assembled from three publicly available
  solar-panel dust datasets — a Kaggle PV-dust set, the clean/dirty classes of a
  6-class faulty-panel set, and a binary Dusty/Clean set — then split 80/10/10.
- **Download:** [HF Dataset (merged)](https://huggingface.co/datasets/adityashirsatrao007/solar-dust-data-combined) or [HF Dataset (raw sources)](https://huggingface.co/datasets/adityashirsatrao007/solar-dust-datasets-raw)
- Layout: `data_combined/{train,val,test}/{clean,dirty}/*.jpg`
- Images resized to 224x224 for the backbone.

---

## 4. Installation

```bash
git clone https://github.com/adityashirsatrao007/solar-panel-dust-detection-vgg-svm.git
cd solar-panel-dust-detection-vgg-svm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> TensorFlow 2.19 + GPU (NVIDIA GTX 1650 Ti, 4 GB) was used for training.
> The dashboard runs on CPU or GPU.

---

## 5. Training (Phase 1 — frozen + SVM)

```bash
python train_solar_dust.py --data data_combined --train-head
```

Optional Phase 2 fine-tuning (slower, needs GPU):

```bash
python train_solar_dust.py --data data_combined --finetune --head-epochs 10
```

Artifacts are written to `Models/`:
`svm_classifier.pkl`, `scaler.pkl`, `pipeline_meta.json`, `model_versions.json`,
`classification_report.txt`, plus feature caches (`cache_*.npz`).

---

## 6. Backbone Comparison (reproducible)

The pipeline supports three backbones. Comparison runs never overwrite the
production model — they are isolated under `Models/compare/<backbone>/` and
accumulated in `Models/comparison_results.csv`.

```bash
for bb in efficientnetb2 mobilenetv3 resnet50; do
  python train_solar_dust.py --data data_combined --train-head --backbone $bb
done
```

### Results (3,787 images, frozen backbone + SVM)

| Backbone | Feat-dim | Params | Accuracy | AUC | 5-fold CV |
|----------|---------:|-------:|---------:|----:|----------:|
| **EfficientNet-B2** | 1408 | 7.77 M | **90.29%** | 0.9596 | 86.38% |
| **MobileNetV3-Large** | 960 | 3.00 M | 88.71% | 0.9502 | 88.05% |
| **ResNet50** | 2048 | 23.59 M | **90.29%** | 0.9655 | 89.55% |

### External validation (independent datasets)
- 2,562-image binary Dusty/Clean set: **98.24%** accuracy, 98.0% dirty recall
- 383-image clean/dirty subset: **99.74%** accuracy, 100% dirty recall

---

## 7. Running the App

### Streamlit (recommended — includes bounding boxes)

```bash
streamlit run streamlit_app/app.py
```

Features:
- Classification (EfficientNet-B2 + SVM)
- Grad-CAM heatmap overlay
- **Grounding DINO bounding boxes** around dusty regions

### Flask Dashboard (REST API)

```bash
# Development:
python app.py

# Production (systemd + Gunicorn, port 5001):
sudo systemctl restart solar-dust-detection
```

Endpoints:
- `GET  /` — dashboard UI
- `GET  /health` — health check (returns model status)
- `POST /analyze` — multipart upload `file=@img.jpg` → JSON `{dustiness, confidence, processing_time}`
- `POST /explain` — full XAI explanation (Grad-CAM, SHAP, IG overlays as base64)

---

## 8. Results & Figures

All figures are generated by `train_solar_dust.py` (metrics/ROC/confusion/confidence)
and `scripts/make_xai_figures.py` (Grad-CAM / SHAP), and live in `figures/`.

### 8.1 Performance metrics
![Performance Metrics](figures/fig5_metrics.png)

### 8.2 Confusion matrix (test set, EfficientNet-B2)
![Confusion Matrix](figures/fig6_confusion_matrix.png)

### 8.3 ROC / AUC
![ROC Curve](figures/fig3_roc_auc.png)

### 8.4 Prediction confidence distribution
![Confidence Distribution](figures/fig7_confidence.png)

### 8.5 Validation loss & accuracy (fine-tuning)
![Loss/Accuracy](figures/fig8_loss_accuracy.png)

### 8.6 Grad-CAM localization
![Grad-CAM](figures/fig9_gradcam.png)

### 8.7 SHAP channel importance
![SHAP](figures/fig10_shap.png)

---

## 9. Explainable AI

`explanations.py` implements five methods, all operating on the frozen backbone's
GAP features + SVM:

1. **Grad-CAM** — gradient-weighted class activation map (localization).
2. **Score-CAM** — forward-pass-weighted activation map (parameter-free).
3. **Integrated Gradients** — axiomatic attribution along an input baseline.
4. **SHAP** — kernel-SHAP feature (channel) importance.
5. **LIME** — local surrogate linear explanation.

Additionally, the Streamlit app uses **Grounding DINO** for zero-shot bounding boxes
around dirty regions — no training required.

Regenerate XAI figures:

```bash
python scripts/make_xai_figures.py
```

---

## 10. Project Structure

```
solar-panel-dust-xai/
├── app.py                      # Flask dashboard (port 5001)
├── explanations.py             # 5 XAI methods
├── train_solar_dust.py         # training + ablation + figures + CSV
├── gunicorn.conf.py            # production WSGI config
├── requirements.txt            # Flask + TF + XAI deps
├── runtime.txt                 # Python 3.12 (Streamlit Cloud)
├── Models/                     # svm_classifier.pkl, scaler.pkl, meta, caches
│   └── compare/<backbone>/     # isolated backbone-comparison artifacts
├── figures/                    # all generated plots (see §8)
├── streamlit_app/
│   ├── app.py                  # Streamlit app (classification + Grad-CAM + DINO boxes)
│   └── requirements.txt        # TF + PyTorch + Streamlit deps
├── scripts/
│   ├── make_xai_figures.py     # fig9 / fig10
│   ├── prepare_data.py         # dataset split
│   ├── make_combined_dataset.py # multi-source merge
│   ├── external_eval.py        # external generalization test
│   └── finetune.py             # backbone fine-tuning
├── templates/  static/         # Flask dashboard UI
└── tests/                      # pytest suite (9 tests)
```

---

## 11. Testing

```bash
python -m pytest tests/ -q
```

9 unit tests cover feature extraction, SVM training, XAI methods, and the
`/analyze` endpoint.

---

## 12. Model Deployment

### HuggingFace Hub

The trained model artifacts are published to HuggingFace for easy access:

```python
from huggingface_hub import hf_hub_download
import joblib

svm_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "svm_classifier.pkl")
scaler_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "scaler.pkl")
svm = joblib.load(svm_p)
scaler = joblib.load(scaler_p)
```

### Datasets

```python
from datasets import load_dataset

# Merged dataset (3,787 images)
ds = load_dataset("adityashirsatrao007/solar-dust-data-combined")

# Raw source datasets
ds_raw = load_dataset("adityashirsatrao007/solar-dust-datasets-raw")
```

---

## 13. Limitations & Future Work

- Per-source datasets are modest; the merged 3,787-image multi-source corpus mitigates single-source overfitting.
- Labels are binary (clean/dirty); multi-class dust-severity is future work.
- Fine-tuning the last 30% of the backbone did not beat the frozen representation on this corpus.
- Future: edge deployment (TensorFlow Lite / ONNX) for on-panel inference.
- Future: train a dedicated YOLO/DETR model for tighter bounding boxes (current DINO boxes are zero-shot, not domain-specific).

---

## 14. License

Code: MIT. Dataset: respect the Kaggle source license.

---

## 15. Author

**Aditya Shirsatrao** — [GitHub](https://github.com/adityashirsatrao007) · [LinkedIn](https://www.linkedin.com/in/adityashirsatrao/) · [Portfolio](https://bento.me/adityashirsatrao007)
