#!/usr/bin/env python3
"""Preprocess the Kaggle datasets into flat f32 binaries that Vearo can load.

Writes to <data-dir>/preprocessed/:
  tabular_X_train.bin  tabular_y_train.bin  tabular_X_val.bin
  tabular_y_val.bin    tabular_X_test.bin   tabular_test_ids.txt
  image_X_train.bin    image_y_train.bin    image_X_val.bin
  image_y_val.bin      image_X_test.bin

Images are written channels-first (C, H, W), which is what the CNN expects.
Writing them H, W, C and reshaping in Rust scrambles the colour channels across
spatial positions and the model cannot learn - that bug cost real accuracy once.

Usage:
    python3 scripts/preprocess.py                    # 32x32 (default)
    python3 scripts/preprocess.py --size 64          # higher resolution
    python3 scripts/preprocess.py --data-dir /path   # non-default location
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42


def preprocess_tabular(data_dir, out_dir):
    print("Processing item price (tabular regression)...")
    train_df = pd.read_csv(os.path.join(data_dir, "item_price/train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "item_price/test.csv"))

    # Track missingness explicitly before filling nulls
    train_df["X2_missing"] = train_df["X2"].isnull().astype(float)
    test_df["X2_missing"] = test_df["X2"].isnull().astype(float)
    train_df["X9_missing"] = train_df["X9"].isnull().astype(float)
    test_df["X9_missing"] = test_df["X9"].isnull().astype(float)

    # 1. Fill missing weights X2 based on product ID X1 group mean across both datasets
    combined_temp = pd.concat([train_df.drop(columns=["Y"], errors="ignore"), test_df], axis=0)
    weight_map = combined_temp.groupby("X1")["X2"].mean()
    
    train_df["X2"] = train_df["X2"].fillna(train_df["X1"].map(weight_map))
    test_df["X2"] = test_df["X2"].fillna(test_df["X1"].map(weight_map))
    
    # Fill remaining missing weights with global train mean
    global_mean_weight = train_df["X2"].mean()
    train_df["X2"] = train_df["X2"].fillna(global_mean_weight)
    test_df["X2"] = test_df["X2"].fillna(global_mean_weight)

    # 2. Impute missing outlet sizes X9 logically based on outlet type X11 and location X10
    def impute_outlet_size(row):
        if pd.isna(row["X9"]):
            if row["X11"] == "Grocery Store":
                return "Small"
            elif row["X11"] in ["Supermarket Type2", "Supermarket Type3"]:
                return "Medium"
            elif row["X11"] == "Supermarket Type1" and row["X10"] == "Tier 2":
                return "Small"
        return row["X9"]

    train_df["X9"] = train_df.apply(impute_outlet_size, axis=1)
    test_df["X9"] = test_df.apply(impute_outlet_size, axis=1)
    train_df["X9"] = train_df["X9"].fillna("Missing")
    test_df["X9"] = test_df["X9"].fillna("Missing")

    # 3. Map inconsistent categories in X3 (Fat Content)
    x3_map = {
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "reg": "Regular",
        "Low Fat": "Low Fat",
        "Regular": "Regular"
    }
    train_df["X3"] = train_df["X3"].map(x3_map)
    test_df["X3"] = test_df["X3"].map(x3_map)

    # 4. NC (Non-Consumable) items should not have food fat content labels
    train_df.loc[train_df["X1"].str.startswith("NC"), "X3"] = "Non-Edible"
    test_df.loc[test_df["X1"].str.startswith("NC"), "X3"] = "Non-Edible"

    # 5. Replace 0.0 values in X4 (Product Visibility) with mean visibility of that product type (X5)
    mean_vis = train_df[train_df["X4"] > 0].groupby("X5")["X4"].mean()
    for df in [train_df, test_df]:
        df.loc[df["X4"] == 0.0, "X4"] = df.loc[df["X4"] == 0.0, "X5"].map(mean_vis)

    y = train_df["Y"].values.astype(np.float32)
    x_train_raw = train_df.drop(columns=["Y"])
    x_test_raw = test_df.copy()
    test_ids = test_df["X1"].values

    combined = pd.concat([x_train_raw, x_test_raw], axis=0, ignore_index=True)

    # 6. Extract Category Prefix from X1
    combined["X1_prefix"] = combined["X1"].str[:2]

    # 7. Convert establishment year X8 to Outlet Age (relative to data collections in 2013)
    combined["Outlet_Age"] = 2013 - combined["X8"]
    combined = combined.drop(columns=["X8"])

    # 8. Add Visibility to Mean Ratio (relative to average visibility in that outlet X7)
    outlet_mean_vis = combined.groupby("X7")["X4"].transform("mean")
    combined["Visibility_Mean_Ratio"] = combined["X4"] / (outlet_mean_vis + 1e-8)

    categorical = ["X3", "X5", "X7", "X9", "X10", "X11", "X1_prefix"]
    encoded = pd.get_dummies(combined, columns=categorical, drop_first=False)
    encoded = encoded.drop(columns=["X1"])

    x_train_full = encoded.iloc[: len(train_df)].values.astype(np.float32)
    x_test_full = encoded.iloc[len(train_df) :].values.astype(np.float32)

    mean = x_train_full.mean(axis=0)
    std = x_train_full.std(axis=0)
    std[std == 0] = 1.0
    x_train_full = (x_train_full - mean) / std
    x_test_full = (x_test_full - mean) / std

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train_full))
    split = int(0.8 * len(x_train_full))
    tr, va = idx[:split], idx[split:]

    x_train_full[tr].tofile(os.path.join(out_dir, "tabular_X_train.bin"))
    y[tr].tofile(os.path.join(out_dir, "tabular_y_train.bin"))
    x_train_full[va].tofile(os.path.join(out_dir, "tabular_X_val.bin"))
    y[va].tofile(os.path.join(out_dir, "tabular_y_val.bin"))
    x_test_full.tofile(os.path.join(out_dir, "tabular_X_test.bin"))

    with open(os.path.join(out_dir, "tabular_test_ids.txt"), "w") as f:
        for item_id in test_ids:
            f.write(f"{item_id}\n")

    print(f"  features={x_train_full.shape[1]} train={len(tr)} val={len(va)}")


def load_image(path, size):
    """Load one image as a flat channels-first (C, H, W) f32 array in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1)).flatten()  # HWC -> CHW


def preprocess_images(data_dir, out_dir, size):
    print(f"Processing scene style (image classification) at {size}x{size}...")
    root = os.path.join(
        data_dir, "scene_style/StyleClassificationIndoors/StyleClassificationIndoors"
    )
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    class_mapping = {}
    with open(os.path.join(root, "class_mapping.txt")) as f:
        for line in f:
            if line.strip():
                name, idx = line.strip().split(":")
                class_mapping[name.strip()] = int(idx.strip())

    images, labels = [], []
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    images.append(load_image(os.path.join(class_dir, fname), size))
                    labels.append(float(class_idx))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skipping {fname}: {exc}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(images))
    split = int(0.8 * len(images))
    tr, va = idx[:split], idx[split:]

    images[tr].tofile(os.path.join(out_dir, "image_X_train.bin"))
    labels[tr].tofile(os.path.join(out_dir, "image_y_train.bin"))
    images[va].tofile(os.path.join(out_dir, "image_X_val.bin"))
    labels[va].tofile(os.path.join(out_dir, "image_y_val.bin"))

    sample_sub = pd.read_csv(os.path.join(data_dir, "scene_style/sample_submission.csv"))
    test_images = []
    for name in sample_sub["ImageName"].values:
        path = os.path.join(test_dir, name)
        try:
            test_images.append(load_image(path, size))
        except Exception as exc:  # noqa: BLE001
            print(f"  missing test image {name} ({exc}), writing zeros")
            test_images.append(np.zeros(size * size * 3, dtype=np.float32))

    np.array(test_images, dtype=np.float32).tofile(
        os.path.join(out_dir, "image_X_test.bin")
    )

    print(f"  classes={len(class_mapping)} train={len(tr)} val={len(va)} test={len(test_images)}")
    print(f"  NOTE: input is {size}x{size}x3; the CNN's flatten dim must match.")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(repo_root, "data", "kaggle"))
    ap.add_argument("--size", type=int, default=32, help="image side length")
    ap.add_argument("--skip-tabular", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_tabular:
        preprocess_tabular(args.data_dir, out_dir)
    if not args.skip_images:
        preprocess_images(args.data_dir, out_dir, args.size)

    print(f"\nDone. Binaries in {out_dir}")
    print("Point the tests at it with:  export VEARO_DATA_DIR=" + args.data_dir)


if __name__ == "__main__":
    main()
