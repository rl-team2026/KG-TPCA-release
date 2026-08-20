from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-cols", nargs="+", default=["patient_id"])
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from cdf_vlm.io import read_table, require_columns, write_table

    df = read_table(args.manifest)
    require_columns(df, [*args.group_cols, args.label_col], name="manifest")
    df = df.copy()
    df["_group_id"] = df[args.group_cols].astype(str).agg("__".join, axis=1)

    groups = (
        df.groupby("_group_id", as_index=False)[args.label_col]
        .max()
        .rename(columns={args.label_col: "_group_label"})
    )

    train_val, test = train_test_split(
        groups,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=groups["_group_label"] if groups["_group_label"].nunique() == 2 else None,
    )
    val_fraction_of_train_val = args.val_size / max(1.0 - args.test_size, 1e-8)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=args.seed,
        stratify=train_val["_group_label"] if train_val["_group_label"].nunique() == 2 else None,
    )

    split_map = {gid: "train" for gid in train["_group_id"]}
    split_map.update({gid: "val" for gid in val["_group_id"]})
    split_map.update({gid: "test" for gid in test["_group_id"]})
    df["split"] = df["_group_id"].map(split_map)
    df = df.drop(columns=["_group_id"])

    write_table(df, args.output)
    summary = pd.crosstab(df["split"], df[args.label_col], margins=True)
    print(summary)
    print(f"Wrote split manifest to {args.output}")


if __name__ == "__main__":
    main()
