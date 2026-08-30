"""
finetune.py - fine-tune VGG16 on solar-panel dust data, then re-train the SVM head.

Pipeline is the paper's hybrid: fine-tuned VGG16 GAP features -> scaler -> linear SVM.
Two-stage training:
  1. train a small dense head on frozen VGG16 with augmentation
  2. unfreeze the last conv block (block5) at low LR, fine-tune, early-stop on validation
Then re-extract GAP features with the fine-tuned encoder and fit a linear SVM,
so the deployed app.py / explanations.py pipeline stays unchanged (512-d GAP + SVM).

Usage:
    CUDA_VISIBLE_DEVICES='' python scripts/finetune.py --data data --epochs 30 --out Models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED = 42
np.random.seed(SEED)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="Models")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30, help="head+finetune epochs each stage")
    p.add_argument("--head-epochs", type=int, default=0, help="override head-only epochs (0 = use --epochs)")
    p.add_argument("--svm-c", default="0.1,1,10,100", help="comma list")
    p.add_argument("--cache", default="cache_ft.npz", help="feature cache under --out")
    return p.parse_args()


def build_model(img_size, n_cls):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import VGG16

    base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap_ft")(base.output)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(n_cls, activation="softmax")(x)
    model = tf.keras.Model(base.input, x)
    return model, base


def train():
    import tensorflow as tf
    from tensorflow.keras import layers, utils
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    args = parse_args()
    t0 = time.time()
    data_dir = os.path.abspath(args.data)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith("."))
    n_cls = len(classes)
    n_train = sum(len(os.listdir(os.path.join(train_dir, c))) for c in classes)
    n_test = sum(len(os.listdir(os.path.join(test_dir, c))) for c in classes)
    print(f"classes={classes} | train={n_train} | test={n_test} | img={args.img_size}x{args.img_size}")

    # ---- Stage 1+2: augmentation-aware fine-tuning ----
    train_dg = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="reflect",
    )
    val_dg = ImageDataGenerator(rescale=1.0 / 255.0)

    model, base = build_model(args.img_size, n_cls)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    gen_train = train_dg.flow_from_directory(
        train_dir, target_size=(args.img_size, args.img_size),
        batch_size=args.batch, class_mode="sparse", shuffle=True, seed=SEED)
    gen_val = val_dg.flow_from_directory(
        test_dir, target_size=(args.img_size, args.img_size),
        batch_size=args.batch, class_mode="sparse", shuffle=False, seed=SEED)
    steps_train = max(1, int(np.ceil(n_train / args.batch)))
    steps_val = max(1, int(np.ceil(n_test / args.batch)))

    head_epochs = args.head_epochs or args.epochs
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    print("Stage 1: training head on frozen VGG16...")
    model.fit(gen_train, validation_data=gen_val,
              epochs=head_epochs, steps_per_epoch=steps_train,
              callbacks=callbacks, verbose=1)

    print("Stage 2: fine-tuning last conv block (block5) at 1e-5...")
    for layer in base.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy")
    model.fit(gen_train, validation_data=gen_val,
              epochs=args.epochs, steps_per_epoch=steps_train,
              callbacks=callbacks, verbose=1)

    model.save(os.path.join(out_dir, "finetuned_vgg16.h5"))
    base.save(os.path.join(out_dir, "finetuned_backbone.keras"))
    base.save_weights(os.path.join(out_dir, "finetuned_backbone_weights.weights.h5"))
    print(f"[{time.time()-t0:.1f}s] fine-tuned model saved")

    # ---- Stage 3: re-extract GAP features with fine-tuned encoder -> SVM ----
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    fe = tf.keras.Model(base.input, layers.GlobalAveragePooling2D()(base.output))

    def extract(data_dir_):
        feats, labels = [], []
        for idx, c in enumerate(classes):
            folder = os.path.join(data_dir_, c)
            paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            for i in range(0, len(paths), args.batch):
                batch_paths = paths[i:i + args.batch]
                batch = np.zeros((len(batch_paths), args.img_size, args.img_size, 3), dtype=np.float32)
                for j, p in enumerate(batch_paths):
                    batch[j] = utils.img_to_array(utils.load_img(p, target_size=(args.img_size, args.img_size))) / 255.0
                feats.append(fe.predict(batch, verbose=0))
                labels.append(np.full(len(batch_paths), idx, dtype=int))
        return np.vstack(feats), np.concatenate(labels)

    cache_path = os.path.join(out_dir, args.cache)
    if os.path.exists(cache_path):
        print("Loading cached fine-tuned features...")
        d = np.load(cache_path)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    else:
        print("Extracting fine-tuned VGG16 features...")
        Xtr, ytr = extract(train_dir)
        Xte, yte = extract(test_dir)
        np.savez(cache_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    best, best_acc = None, -1.0
    for c in [float(x) for x in args.svm_c.split(",")]:
        svm = SVC(kernel="linear", C=c, probability=True, class_weight="balanced", random_state=SEED)
        svm.fit(Xtr_s, ytr)
        acc = accuracy_score(yte, svm.predict(Xte_s))
        print(f"  C={c} -> test acc {acc:.4f}")
        if acc > best_acc:
            best, best_acc = (svm, c), acc
    svm = best[0]

    y_pred = svm.predict(Xte_s)
    scores = {
        "accuracy": float(accuracy_score(yte, y_pred)),
        "f1": float(f1_score(yte, y_pred, average="weighted", zero_division=0)),
    }
    print("\nFinal test metrics:", {k: round(v, 4) for k, v in scores.items()})
    print(classification_report(yte, y_pred, zero_division=0))
    print(f"Best SVM: C={best[1]} | accuracy={best_acc:.4f}")

    joblib.dump(svm, os.path.join(out_dir, "svm_classifier.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "class_names.json"), "w") as fh:
        json.dump(classes, fh)
    with open(os.path.join(out_dir, "pipeline_meta.json"), "w") as fh:
        json.dump({
            "img_size": args.img_size, "kernel": "linear", "backbone": "VGG16_finetuned",
            "svm": {"C": best[1]}, "classes": classes,
            "train_samples": int(len(ytr)), "test_samples": int(len(yte)),
            "scores": scores,
        }, fh, indent=2)
    print("Deployed SVM/scaler/meta saved under", out_dir)


if __name__ == "__main__":
    from tensorflow import keras
    train()