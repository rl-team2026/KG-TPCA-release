from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DDSM CC/MLO pair CSV into train/val by side_id.")
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.pair_csv)
    if "side_id" not in df or "label" not in df:
        raise ValueError("Pair CSV must include side_id and label columns.")
    groups = df.groupby("side_id", as_index=False)["label"].max()
    train_groups, val_groups = train_test_split(
        groups,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=groups["label"] if groups["label"].nunique() == 2 else None,
    )
    train_ids = set(train_groups["side_id"].astype(str))
    train_df = df[df["side_id"].astype(str).isin(train_ids)].copy()
    val_df = df[~df["side_id"].astype(str).isin(train_ids)].copy()
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)
    print(f"Wrote train={len(train_df)} to {args.train_output}")
    print(f"Wrote val={len(val_df)} to {args.val_output}")


if __name__ == "__main__":
    main()

