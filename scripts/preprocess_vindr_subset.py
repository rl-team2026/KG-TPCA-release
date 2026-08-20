#!/usr/bin/env python3
"""Preprocess the downloaded VinDr subset into paired-view PNGs + manifest.

Protocol mirrors INbreast preprocessing:
  - 16-bit raw DICOM -> 2-98 percentile stretch -> uint8 RGB PNG at 1520x912
  - pairs: same (study, laterality) with CC + MLO; labels from subset_manifest
  - descriptions carry density + BI-RADS findings (same keyword channel as DDSM)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dicom_to_png(dcm_path: Path, out_path: Path, size: tuple[int, int]) -> None:
    import pydicom

    ds = pydicom.dcmread(str(dcm_path))
    raw = ds.pixel_array.astype(np.float32)
    pad = ds.get("PixelPaddingValue")
    valid = raw != pad if pad is not None else np.ones(raw.shape, dtype=bool)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        arr = raw.max() - raw
    else:
        arr = raw
    lo, hi = np.percentile(arr[valid], 2), np.percentile(arr[valid], 98)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    out[~valid] = 0
    img = Image.fromarray(out, mode="L").resize((size[1], size[0]), Image.LANCZOS)
    img.convert("RGB").save(out_path)


def main() -> None:
    manifest = pd.read_csv("data/vindr/subset_manifest.csv")
    out_dir = Path("data/vindr_processed")
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    missing = 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="converting"):
        src = Path(r["local_path"])
        if not src.exists():
            missing += 1
            continue
        dst = images_out / f"{r['image_id']}.png"
        if not dst.exists():
            dicom_to_png(src, dst, (1520, 912))
    if missing:
        print(f"[WARN] {missing} DICOMs still missing (download incomplete?)")

    table = manifest[manifest["local_path"].apply(lambda p: (images_out / f"{Path(p).stem}.png").exists())].copy()

    pairs = []
    for (study, lat), grp in table.groupby(["study_id", "laterality"]):
        by_view = {row.view_position: row for _, row in grp.iterrows()}
        if "CC" not in by_view or "MLO" not in by_view:
            continue
        cc, mlo = by_view["CC"], by_view["MLO"]
        if cc["label"] != mlo["label"]:
            print(f"[WARN] label mismatch {study} {lat}: CC={cc['breast_birads']} MLO={mlo['breast_birads']}")
            continue
        lat_word = "left" if lat == "L" else "right"

        def desc(row, view: str) -> str:
            density = str(row["breast_density"]).replace("DENSITY ", "")
            base = f"breast density is {density}, {lat_word} breast, {view} view"
            findings = str(row["findings_text"])
            return f"{base}, {findings}" if findings and findings != "nan" else base

        pairs.append({
            "side_id": f"VINDR-{study}-{lat}",
            "cc_image_path": f"images/{cc['image_id']}.png",
            "cc_mask_path": "",
            "cc_description": desc(cc, "CC"),
            "mlo_image_path": f"images/{mlo['image_id']}.png",
            "mlo_mask_path": "",
            "mlo_description": desc(mlo, "MLO"),
            "label": int(cc["label"]),
        })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(out_dir / "breast_pairs.csv", index=False)
    print(f"\nPaired breasts: {len(pairs_df)} (malignant {pairs_df['label'].sum()}, benign {len(pairs_df) - pairs_df['label'].sum()})")
    print(f"Manifest: {out_dir / 'breast_pairs.csv'}")


if __name__ == "__main__":
    main()
