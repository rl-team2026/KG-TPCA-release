#!/usr/bin/env python3
"""Build the VinDr-Mammo test-subset manifest for external validation.

Subset (official VinDr-Mammo test split, benign=BI-RADS 1/2/3, malignant=4/5):
  - all official-test images with BI-RADS 4/5 (~198 images, 96 studies)
  - all official-test BI-RADS 3 images with CC/MLO views (~186 images, 91 studies)
  - 100 benign studies sampled from official-test BI-RADS 1/2 (seed 42),
    one laterality (L preferred) with CC+MLO (~200 images)

Outputs:
  data/vindr/subset_manifest.csv   (metadata + URLs + local paths)
  data/vindr/download_urls.txt     (for parallel wget)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_URL = "https://physionet.org/files/vindr-mammo/1.0.0/images/{study_id}/{image_id}.dicom"
LOCAL_DIR = Path(os.environ.get("VINDR_IMAGE_ROOT", "data/vindr_subset/images"))

FINDING_TO_TEXT = {
    "Mass": "abnormality type mass",
    "Suspicious Calcification": "abnormality type calcification",
    "Architectural Distortion": "architectural distortion",
    "Focal Asymmetry": "focal asymmetry",
    "Asymmetry": "asymmetry",
    "Global Asymmetry": "global asymmetry",
}


def main() -> None:
    breast = pd.read_csv("data/vindr/breast-level_annotations.csv")
    finding = pd.read_csv("data/vindr/finding_annotations.csv")
    test = breast[breast["split"] == "test"]

    # malignant: all BI-RADS 4/5
    mal = test[test["breast_birads"].isin(["BI-RADS 4", "BI-RADS 5"])].copy()
    mal["label"] = 1

    # benign: official definition BI-RADS 1/2/3; all BI-RADS 3 + 100 sampled 1/2 studies
    ben_pool = test[test["breast_birads"].isin(["BI-RADS 1", "BI-RADS 2", "BI-RADS 3"])].copy()
    ben_pool["label"] = 0
    ben3 = ben_pool[ben_pool["breast_birads"] == "BI-RADS 3"]
    ben12 = ben_pool[ben_pool["breast_birads"].isin(["BI-RADS 1", "BI-RADS 2"])]
    studies = ben12["study_id"].drop_duplicates().sample(100, random_state=42)
    rows = []
    for sid in studies:
        s = ben12[ben12["study_id"] == sid]
        for lat in ("L", "R"):
            sel = s[s["laterality"] == lat]
            if set(sel["view_position"]) >= {"CC", "MLO"}:
                rows.append(sel[sel["view_position"].isin(["CC", "MLO"])])
                break
    ben = pd.concat([*rows, ben3[ben3["view_position"].isin(["CC", "MLO"])]])

    subset = pd.concat([mal, ben]).drop_duplicates(subset="image_id")
    print(f"subset: {len(subset)} images "
          f"(malignant {len(subset[subset['label'] == 1])}, benign {len(subset[subset['label'] == 0])})")
    print(f"  studies: {subset['study_id'].nunique()}")

    # findings text per image (from finding_annotations; multiple findings per image)
    finding["findings_text"] = finding["finding_categories"].apply(
        lambda x: ", ".join(
            FINDING_TO_TEXT[f.strip().strip("'")]
            for f in x.strip("[]").split(",")
            if f.strip().strip("'") in FINDING_TO_TEXT
        )
    )
    fmap = finding.groupby("image_id")["findings_text"].apply(
        lambda texts: " | ".join(sorted(set(t for t in texts if t)))
    )
    subset["findings_text"] = subset["image_id"].map(fmap).fillna("")

    subset["url"] = subset.apply(
        lambda r: BASE_URL.format(study_id=r["study_id"], image_id=r["image_id"]), axis=1
    )
    subset["local_path"] = subset.apply(
        lambda r: str(LOCAL_DIR / r["study_id"] / f"{r['image_id']}.dicom"), axis=1
    )

    cols = ["image_id", "study_id", "laterality", "view_position", "breast_birads",
            "breast_density", "label", "findings_text", "url", "local_path"]
    subset[cols].to_csv("data/vindr/subset_manifest.csv", index=False)
    subset["url"].to_csv("data/vindr/download_urls.txt", index=False, header=False)
    print("Saved data/vindr/subset_manifest.csv and download_urls.txt")


if __name__ == "__main__":
    main()
