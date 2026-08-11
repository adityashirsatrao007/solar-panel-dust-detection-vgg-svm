# Dataset

This folder receives the prepared training/test dataset (see `.gitignore` - images are
**not** committed).

## Layout expected by `train_solar_dust.py`

```
data/
  train/clean/*.jpg
  train/dirty/*.jpg
  test/clean/*.jpg
  test/dirty/*.jpg
```

## Prepare it (one command)

    pip install kagglehub
    python scripts/prepare_data.py

If you have an already-downloaded archive locally, structure it instead:

    python scripts/prepare_data.py --source /path/to/clean_and_dirty_folder

## Public sources

- Kaggle - Solar Photovoltaics Panel for Dust Detection:
  https://www.kaggle.com/datasets/safwanshamsir99/solar-photovoltaics-panell-for-dust-dectection
- Kaggle - Solar Panel Images Clean and Faulty:
  https://www.kaggle.com/datasets/pythonafroz/solar-panel-images
- Kaggle - Solar Panel Dust Detection:
  https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection

## Train

    python train_solar_dust.py --data data --train-head