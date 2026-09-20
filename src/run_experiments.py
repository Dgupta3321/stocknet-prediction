"""
Entry point for the multimodal stock movement experiments.

Builds (or loads) the aligned dataset, then trains and evaluates four
configurations -- {BiLSTM, BiGRU} x {technical only, technical + sentiment} --
against a majority-class baseline, and writes the results to results.md.

Usage, from the repository root:

    python src/run_experiments.py                  # build dataset, then train
    python src/run_experiments.py --skip-build     # reuse data/aligned_dataset.csv
    python src/run_experiments.py --epochs 30 --seed 1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_merger import build_aligned_multimodal_dataset          # noqa: E402
from data_preprocessing import SENTIMENT_FEATURES, TECH_FEATURES  # noqa: E402
from models import StockMovementPredictor                         # noqa: E402
from train import train_model                                     # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICE_DIR = os.path.join(REPO_ROOT, "data", "price", "raw")
TWEET_DIR = os.path.join(REPO_ROOT, "data", "tweet", "preprocessed")
ALIGNED_CSV = os.path.join(REPO_ROOT, "data", "aligned_dataset.csv")
RESULTS_MD = os.path.join(REPO_ROOT, "results.md")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(skip_build):
    """Return the aligned multimodal dataframe, building it if necessary."""
    if skip_build or os.path.exists(ALIGNED_CSV):
        if not os.path.exists(ALIGNED_CSV):
            sys.exit(
                f"--skip-build was passed but {ALIGNED_CSV} does not exist.\n"
                "Run without --skip-build once to create it."
            )
        print(f"Loading cached dataset from {ALIGNED_CSV}")
        return pd.read_csv(ALIGNED_CSV, parse_dates=["Date"])

    if not os.path.isdir(PRICE_DIR) or not os.path.isdir(TWEET_DIR):
        sys.exit(
            "Could not find the StockNet data.\n"
            f"  expected prices at: {PRICE_DIR}\n"
            f"  expected tweets at: {TWEET_DIR}\n\n"
            "Download it from https://github.com/yumoxu/stocknet-dataset "
            "and place it under data/ as shown in the README."
        )

    os.makedirs(os.path.dirname(ALIGNED_CSV), exist_ok=True)
    return build_aligned_multimodal_dataset(PRICE_DIR, TWEET_DIR, output_path=ALIGNED_CSV)


def clean_non_finite(df):
    """
    Drop rows whose features are missing or infinite.

    Several technical features are ratios with a denominator that can legitimately
    be zero -- Volume_Change is a percentage change on volume, Volume_Ratio divides
    by a rolling volume mean, and High_Low_Ratio divides by the day's low. On a
    zero-volume day those evaluate to +/-inf rather than NaN, so the dropna() in
    data_merger.py does not catch them and StandardScaler rejects them.
    """
    feature_cols = TECH_FEATURES + SENTIMENT_FEATURES
    before = len(df)

    values = df[feature_cols].to_numpy(dtype=np.float64)
    infinite = np.isinf(values)
    if infinite.any():
        affected = [c for c, hit in zip(feature_cols, infinite.any(axis=0)) if hit]
        print(f"Infinite values in: {', '.join(affected)}")
        df = df.copy()
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=feature_cols + ["Target"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with missing or infinite features "
              f"({dropped / before:.2%} of the data); {len(df):,} remain")
    return df


def chronological_split(df, val_fraction=0.2):
    """
    Split by calendar date, not by row.

    Every ticker is cut at the same date, so the validation set is strictly
    the future relative to the training set. Splitting rows at random would
    let the model see a stock's Thursday while predicting its Wednesday.
    """
    dates = np.sort(df["Date"].unique())
    cutoff = dates[int(len(dates) * (1 - val_fraction))]
    train = df[df["Date"] < cutoff]
    val = df[df["Date"] >= cutoff]
    return train, val, pd.Timestamp(cutoff)


def build_sequences(df, feature_cols, sequence_length):
    """
    Build sliding windows one ticker at a time.

    The aligned dataset is a vertical stack of per-ticker frames, so slicing it
    as one long array would produce windows that straddle two companies. Each
    ticker is windowed separately and the results concatenated afterwards.
    """
    xs, ys = [], []
    for _, group in df.groupby("Ticker", sort=False):
        group = group.sort_values("Date")
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group["Target"].to_numpy(dtype=np.float32)
        for i in range(len(group) - sequence_length):
            xs.append(features[i:i + sequence_length])
            ys.append(targets[i + sequence_length])
    if not xs:
        sys.exit("No sequences were built -- check sequence_length against rows per ticker.")
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def make_loaders(train_df, val_df, feature_cols, sequence_length, batch_size):
    """Scale on the training rows only, then window each split independently."""
    scaler = StandardScaler().fit(train_df[feature_cols].to_numpy(dtype=np.float64))

    scaled = {}
    for name, frame in (("train", train_df), ("val", val_df)):
        frame = frame.copy()
        frame[feature_cols] = scaler.transform(frame[feature_cols].to_numpy(dtype=np.float64))
        scaled[name] = frame

    x_train, y_train = build_sequences(scaled["train"], feature_cols, sequence_length)
    x_val, y_val = build_sequences(scaled["val"], feature_cols, sequence_length)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        y_train,
        y_val,
    )


def majority_baseline(y_train, y_val):
    """Always predict whichever class is more common in training."""
    majority = 1.0 if y_train.mean() >= 0.5 else 0.0
    preds = np.full_like(y_val, majority)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_val, preds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="reuse data/aligned_dataset.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = load_dataset(args.skip_build)
    df = clean_non_finite(df)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    train_df, val_df, cutoff = chronological_split(df)
    print(
        f"\nRows: {len(df):,}  |  tickers: {df['Ticker'].nunique()}  |  "
        f"dates: {df['Date'].min().date()} to {df['Date'].max().date()}"
    )
    print(f"Chronological cutoff: {cutoff.date()}  "
          f"(train {len(train_df):,} rows, validation {len(val_df):,} rows)")

    configurations = [
        ("Technical only", TECH_FEATURES),
        ("Technical + sentiment", TECH_FEATURES + SENTIMENT_FEATURES),
    ]

    rows = []
    baseline_recorded = False

    for feature_label, feature_cols in configurations:
        train_loader, val_loader, y_train, y_val = make_loaders(
            train_df, val_df, feature_cols, args.sequence_length, args.batch_size
        )

        if not baseline_recorded:
            base = majority_baseline(y_train, y_val)
            rows.append(("Majority-class baseline", "-", base))
            print(
                f"\nMajority-class baseline  |  Acc {base['accuracy']:.4f}  "
                f"F1 {base['f1']:.4f}  MCC {base['mcc']:.4f}"
            )
            print(f"Validation class balance: {y_val.mean():.4f} up-days")
            baseline_recorded = True

        for model_type in ("LSTM", "GRU"):
            print(f"\n=== Bi{model_type} + attention | {feature_label} ===")
            set_seed(args.seed)
            model = StockMovementPredictor(input_dim=len(feature_cols), model_type=model_type)
            metrics = train_model(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, device=device,
            )
            rows.append((f"Bi{model_type} + attention", feature_label, metrics))

    write_results(rows, cutoff, args)


def write_results(rows, cutoff, args):
    header = (
        "| Model | Features | Accuracy | MCC | F1 |\n"
        "|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {model} | {features} | {m['accuracy']:.4f} | {m['mcc']:.4f} | {m['f1']:.4f} |\n"
        for model, features, m in rows
    )
    footer = (
        f"\nChronological split at {cutoff.date()} | sequence length "
        f"{args.sequence_length} | {args.epochs} epochs | seed {args.seed}\n"
    )
    table = header + body + footer

    with open(RESULTS_MD, "w") as handle:
        handle.write("# Results\n\n" + table)

    print("\n" + table)
    print(f"Written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
