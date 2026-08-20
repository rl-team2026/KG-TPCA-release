"""Preprocess INbreast DICOMs into paired-view PNGs + manifest CSV.

Protocol (mirrors the CBIS-DDSM setup):
  - label: BI-RADS 4a/4b/4c/5/6 = malignant (1), BI-RADS 1/2 = benign (0), 3 dropped
  - pairs: same patient + laterality with both CC and MLO views
  - image: 16-bit raw values -> 2-98 percentile stretch -> uint8 -> RGB PNG
    at 1520x912 (kept at native aspect via resize)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xls", default="data/INbreast.xls")
    p.add_argument("--image-dir", default="data/ALL-IMGS")
    p.add_argument("--output-dir", default="data/inbreast_processed")
    p.add_argument("--size", type=int, nargs=2, default=[1520, 912])
    return p.parse_args()


def dicom_to_png(dcm_path: Path, out_path: Path, size: tuple[int, int]) -> None:
    import pydicom

    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((size[1], size[0]), Image.LANCZOS)
    img.convert("RGB").save(out_path)


def main() -> None:
    args = parse_args()
    xls_path = Path(args.xls)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xls_path, sheet_name="Sheet1")

    # The anonymized XLS stores numeric IDs in "File Name" (Excel float, e.g.
    # "20586960.0"); DICOM filenames carry the same ID as prefix:
    # {id}_{hash}_MG_{L/R}_{CC/ML}_ANON.dcm. Match by prefix and verify
    # laterality/view agreement.
    import re

    dcm_by_id = {}
    for p in sorted(img_dir.glob("*.dcm")):
        m = re.match(r"(\d+)_[a-f0-9]+_MG_([LR])_(CC|ML)_", p.name)
        if m:
            dcm_by_id[m.group(1)] = (p.name, m.group(2), "MLO" if m.group(3) == "ML" else m.group(3))

    birads_to_label = {
        "1": 0, "2": 0,
        "4a": 1, "4b": 1, "4c": 1, "5": 1, "6": 1,
    }
    rows = []
    for pos, (_, r) in tqdm(enumerate(df.iterrows()), total=len(df), desc="parsing xls"):
        birads = str(r["Bi-Rads"]).strip()
        label = birads_to_label.get(birads)
        if label is None:
            continue  # drop BI-RADS 3 and malformed values
        raw = str(r["File Name"]).strip()
        num = raw[:-2] if raw.endswith(".0") else raw
        lat = str(r["Laterality"]).strip()
        view = str(r["View"]).strip()
        match = dcm_by_id.get(num)
        if match is None:
            print(f"  [WARN] no DICOM for row (id={num}, {lat}, {view})", file=sys.stderr)
            continue
        fname, d_lat, d_view = match
        if d_lat != lat or d_view != view:
            print(f"  [WARN] lat/view mismatch for {fname}: xls=({lat},{view}) dcm=({d_lat},{d_view})", file=sys.stderr)
            continue
        hash_part = re.match(r"\d+_([a-f0-9]+)_MG_", fname).group(1)

        def has_flag(col: str) -> bool:
            val = r[col]
            return isinstance(val, str) and val.strip().upper() == "X"

        findings = []
        if has_flag("Mass "):
            findings.append("abnormality type mass")
        if has_flag("Micros"):
            findings.append("abnormality type calcification")
        if has_flag("Distortion"):
            findings.append("architectural distortion")

        rows.append(
            {
                "xls_pos": pos,
                "hash": hash_part,
                "laterality": lat,
                "view": view,
                "file_name": fname,
                "birads": birads,
                "label": label,
                "findings": ", ".join(findings),
            }
        )

    table = pd.DataFrame(rows)
    print(f"Usable images: {len(table)} (malignant {table['label'].sum()}, benign {len(table) - table['label'].sum()})")

    # Group images into patients. The filename hash is study-level, so a few
    # groups contain >4 views (two patients sharing a hash); split them with a
    # greedy rule: within a hash group sorted by XLS row order, start a new
    # patient whenever the current one already has this (laterality, view).
    patients = []
    for hash_part, grp in table.groupby("hash"):
        grp = grp.sort_values("xls_pos")
        cur, cur_keys = [], set()
        for _, r in grp.iterrows():
            key = (r["laterality"], r["view"])
            if key in cur_keys:
                patients.append(cur)
                cur, cur_keys = [], set()
            cur.append(r)
            cur_keys.add(key)
        patients.append(cur)

    pairs = []
    for pi, patient in enumerate(patients):
        by_lat = {}
        for r in patient:
            by_lat.setdefault(r["laterality"], {})[r["view"]] = r
        for lat, views in by_lat.items():
            if "CC" not in views or "MLO" not in views:
                continue
            cc_r, mlo_r = views["CC"], views["MLO"]
            if cc_r["label"] != mlo_r["label"]:
                print(
                    f"  [WARN] label mismatch patient {pi} {lat}: CC={cc_r['birads']} MLO={mlo_r['birads']}",
                    file=sys.stderr,
                )
                continue
            pairs.append({"hash": cc_r["hash"], "patient_idx": pi, "laterality": lat, "cc": cc_r, "mlo": mlo_r})
    print(f"Paired breasts: {len(pairs)}")

    manifest = []
    for p in tqdm(pairs, desc="converting"):
        side_id = f"INB-{p['patient_idx']}-{p['laterality']}"
        lat_word = "left" if p["laterality"] == "L" else "right"

        def build_desc(r, view: str) -> str:
            base = f"breast density is unknown, {lat_word} breast, {view} view"
            findings = r["findings"]
            return f"{base}, {findings}" if findings else base

        desc_cc = build_desc(p["cc"], "CC")
        desc_mlo = build_desc(p["mlo"], "MLO")

        cc_src = img_dir / p["cc"]["file_name"]
        mlo_src = img_dir / p["mlo"]["file_name"]
        cc_dst = images_out / f"{side_id}_CC.png"
        mlo_dst = images_out / f"{side_id}_MLO.png"
        dicom_to_png(cc_src, cc_dst, tuple(args.size))
        dicom_to_png(mlo_src, mlo_dst, tuple(args.size))

        manifest.append(
            {
                "side_id": side_id,
                "cc_image_path": str(cc_dst.relative_to(out_dir)),
                "cc_mask_path": "",
                "cc_description": desc_cc,
                "mlo_image_path": str(mlo_dst.relative_to(out_dir)),
                "mlo_mask_path": "",
                "mlo_description": desc_mlo,
                "label": int(p["cc"]["label"]),
            }
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_path = out_dir / "breast_pairs.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nPaired breasts: {len(manifest_df)} (malignant {manifest_df['label'].sum()}, benign {len(manifest_df) - manifest_df['label'].sum()})")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
