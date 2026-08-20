from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ROC, PR, and calibration plots.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prob-col", default="prob_calibrated")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    df = df[df[args.split_col].astype(str) == args.split].copy()
    y_true = df[args.label_col].to_numpy()
    y_prob = df[args.prob_col].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=200)
    plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
