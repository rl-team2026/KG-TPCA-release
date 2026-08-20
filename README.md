# KG-TPCA: Knowledge-Graph-Enhanced Text Prompt Learning with Cross-View Consistency for Mammography Diagnosis

Official code for **KG-TPCA**, a BI-RADS-prior multimodal framework for benign–malignant classification of breast lesions in dual-view (CC/MLO) mammography.

KG-TPCA builds on a frozen [BiomedCLIP](https://aka.ms/biomedclip-paper) vision–language backbone and introduces:

- **Knowledge-graph (KG) enhanced prompt encoder**: 16 pathology keywords are embedded under BI-RADS prior constraints (malignancy-association prior loss + semantic relation loss), producing structured diagnostic prompt vectors;
- **Prompt attention gate**: text prompts condition visual features through channel-wise affine modulation, guiding the model toward lesion-relevant regions;
- **Cross-view consistency loss**: symmetric prediction agreement between the CC and MLO branches, emulating the radiologist's dual-view cross-validation practice.

All experiments use a small-sample protocol: the BiomedCLIP backbone is frozen and only the prompt/gate/classifier modules (≈290K parameters) are trained on CBIS-DDSM. External generalization is validated on INbreast and VinDr-Mammo **without any retraining**.

> **Disclaimer**: This is retrospective research code. It is not a clinical diagnostic device and must not be used for patient care.

## Key Results

### In-domain: CBIS-DDSM (official train/test split)

| Method | AUROC | AUPRC | F1 | Sens | Spec | ECE | Trainable params |
|---|---|---|---|---|---|---|---|
| ResNet50 (ImageNet, fine-tuned) | 0.7059 | 0.6100 | 0.6190 | 0.8426 | 0.3987 | 0.3375 | 23.5M |
| BiomedCLIP zero-shot (text+image) | 0.5742 | 0.4812 | 0.5674 | 0.9352 | 0.0696 | 0.0957 | 0 |
| BiomedCLIP + text prompt learning (16 keywords) | 0.7970 | 0.7170 | 0.6667 | 0.6111 | 0.8481 | 0.0967 | ~34K |
| **KG-TPCA (ours)** | **0.8404** | **0.7820** | **0.7156** | **0.7222** | **0.7975** | **0.0782** | **~290K** |
| Mammo-CLIP zero-shot (dual-view) | 0.5788 | 0.5168 | 0.1626 | 0.0926 | 0.9684 | 0.0927 | 0 |
| Mammo-CLIP linear probe (dual-view) | 0.7157 | 0.6286 | 0.5929 | 0.6204 | 0.6772 | 0.1622 | ~1K |
| Mammo-CLIP fine-tuned (frozen backbone) | 0.7792 | 0.7048 | 0.6857 | 0.7778 | 0.6646 | 0.1023 | ~2.0M |

### External validation (all models trained on CBIS-DDSM only, direct inference)

**INbreast** (191 CC/MLO paired breasts; BI-RADS 3 excluded, same protocol as the main experiments):

| Method | Training data | AUROC | AUPRC |
|---|---|---|---|
| Mammo-CLIP zero-shot (dual-view) | none (pretrained) | 0.5867 | 0.3179 |
| Mammo-CLIP linear probe (dual-view) | CBIS-DDSM | 0.4732 | 0.2528 |
| TPCA (no KG, ablation) | CBIS-DDSM | 0.6943 | 0.4210 |
| **KG-TPCA (ours)** | CBIS-DDSM | **0.8433** | **0.6497** |

**VinDr-Mammo, screening subset** (199 CC/MLO paired breasts: all BI-RADS 4/5 + 100 sampled BI-RADS 1/2 studies):

| Method | Training data | AUROC | AUPRC |
|---|---|---|---|
| Mammo-CLIP zero-shot (dual-view) | none (pretrained) | 0.5075 | 0.5843 |
| TPCA (no KG, ablation) | CBIS-DDSM | 0.6660 | 0.6760 |
| **KG-TPCA (ours)** | CBIS-DDSM | **0.6675** | **0.7899** |

**VinDr-Mammo, official definition** (292 CC/MLO paired breasts: BI-RADS 1/2/3 benign vs 4/5 malignant):

| Method | Training data | AUROC | AUPRC |
|---|---|---|---|
| Mammo-CLIP zero-shot (dual-view) | none (pretrained) | 0.4920 | 0.3966 |
| TPCA (no KG, ablation) | CBIS-DDSM | 0.6124 | 0.4706 |
| KG-TPCA (ours) | CBIS-DDSM | 0.5530 | 0.4072 |

## Repository Layout

```text
cdf_vlm/                  # core package: models, KG prompt encoder, datasets, metrics
cdf_vlm/cli/              # runnable entry points (training, evaluation, preprocessing)
scripts/                  # KG-TPCA training and analysis scripts
tests/smoke/              # local model-loading smoke tests
```

## Installation

```bash
conda create -n kgtpca python=3.10 -y
conda activate kgtpca
pip install -r requirements.txt
```

### Model weights

1. **BiomedCLIP** (required): download the OpenAI-format checkpoint from
   [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
   and place it at:

   ```text
   external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/
   ├── open_clip_config.json
   ├── open_clip_pytorch_model.bin
   ├── open_clip_tokenizer.json  (if distributed)
   └── ...
   ```

   Verify with:

   ```bash
   python tests/smoke/test_biomedclip.py
   ```

2. **Mammo-CLIP** (optional, only for the Mammo-CLIP comparison experiments): clone the official
   [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) repository to `external/Mammo-CLIP` and
   `external/Mammo-CLIP-src`, and download its pre-trained checkpoints
   (e.g. `external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar`).
   The Mammo-CLIP scripts fall back to a local backbone, so they run without network access once
   the weights are cached.

## Data Preparation

### CBIS-DDSM

Download the CBIS-DDSM DICOM images and the `calc/mass_case_description_train/test_set.csv` files
from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/), then run:

```bash
python -m cdf_vlm.cli.preprocess_ddsm \
    --root-dcm-folder /path/to/CBIS-DDSM \
    --train-csv /path/to/calc_case_description_train_set.csv /path/to/mass_case_description_train_set.csv \
    --test-csv /path/to/calc_case_description_test_set.csv /path/to/mass_case_description_test_set.csv \
    --output-dir data/processed/ddsm

python -m cdf_vlm.cli.make_splits
python -m cdf_vlm.cli.split_ddsm_pairs
```

### INbreast (external validation)

Request access to [INbreast](https://medicalresearch.inescporto.pt/breastresearch/index.html)
(`INbreast.xls` + `AllDICOMs`), place them under `data/`, then run:

```bash
python scripts/preprocess_inbreast.py \
    --xls data/INbreast.xls \
    --image-dir data/ALL-IMGS \
    --output-dir data/inbreast_processed
```

This produces `data/inbreast_processed/breast_pairs.csv` (CC/MLO paired breasts with report
descriptions and labels) and the 1520×912 PNG images.

### VinDr-Mammo (external validation)

VinDr-Mammo requires credentialed PhysioNet access
([https://physionet.org/content/vindr-mammo/1.0.0/](https://physionet.org/content/vindr-mammo/1.0.0/)).
Download `breast-level_annotations.csv`, `finding_annotations.csv`, and `metadata.csv` into
`data/vindr/`, then build the external-validation subset manifest:

```bash
python scripts/prepare_vindr_subset.py          # writes data/vindr/subset_manifest.csv + download_urls.txt
```

Download the listed DICOMs (parallel `wget -c` is recommended; PhysioNet is slow) into
`data/vindr_subset/images/<study_id>/<image_id>.dicom`
(override the location with the `VINDR_IMAGE_ROOT` environment variable), then convert:

```bash
python scripts/preprocess_vindr_subset.py       # writes data/vindr_processed/breast_pairs.csv + PNGs
```

To reproduce the screening-subset numbers (BI-RADS 1/2 vs 4/5, 199 pairs), filter
`breast_pairs.csv` to rows whose images are not BI-RADS 3, or simply use
`subset_manifest.csv` directly.

## Training (CBIS-DDSM)

### KG-TPCA (main model)

```bash
python scripts/run_kg_consistency.py
```

Trains the KG-enhanced prompt encoder + prompt gate + MLP classifier with the cross-view
consistency loss. Checkpoints and metrics are written to
`outputs/encoder_ablation/kg_enhanced_consistency/` (hyperparameters: lr 1e-4, batch 16,
20 epochs, kg-prior 0.01, kg-pairwise 0.005, consistency 0.05).

### TPCA baseline (no KG)

```bash
python -m cdf_vlm.cli.train_ddsm_joint_v2
```

### BiomedCLIP zero-shot baseline

```bash
python scripts/run_biomedclip_zeroshot.py
```

### Mammo-CLIP comparison protocols

```bash
python -m cdf_vlm.cli.mammoclip_zeroshot \
    --pair-csv data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv \
    --data-root data/processed/ddsm \
    --output-dir outputs/mammoclip_zeroshot

python -m cdf_vlm.cli.mammoclip_linear_probe \
    --train-pair-csv data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv \
    --val-pair-csv data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv \
    --test-pair-csv data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv \
    --data-root data/processed/ddsm \
    --output-dir outputs/mammoclip_linear_probe
```

## External Validation (no retraining)

```bash
# KG-TPCA on INbreast / VinDr
python -m cdf_vlm.cli.eval_kgtpca_external \
    --pair-csv data/inbreast_processed/breast_pairs.csv \
    --data-root data/inbreast_processed \
    --checkpoint outputs/encoder_ablation/kg_enhanced_consistency_v3/model.pt \
    --output-dir outputs/kgtpca_inbreast_eval

# TPCA (no KG) on INbreast / VinDr
python -m cdf_vlm.cli.eval_tpca_external \
    --pair-csv data/inbreast_processed/breast_pairs.csv \
    --data-root data/inbreast_processed \
    --checkpoint outputs/ddsm_ablation_v2/biomedclip_v2_with_consistency/model.pt \
    --output-dir outputs/tpca_inbreast_eval
```

Each run writes `metrics.json` (AUROC, AUPRC, F1, accuracy, sensitivity, specificity, ECE,
sensitivity@spec90) and `predictions.csv`. ROC/PR/calibration figures can be generated with
`python -m cdf_vlm.cli.plot_predictions --predictions <csv> --output-dir <dir>`.

## Analysis Scripts

| Script | Purpose |
|---|---|
| `scripts/45_keyword_shap_analysis.py` | Marginal contribution of each pathology keyword (paper Fig. 2/3) |
| `scripts/46_kg_encoder_ablation.py` | KG encoder component ablation (baseline / structured / kg-enhanced) |
| `scripts/regenerate_tsne_cn.py` | t-SNE visualization of the feature space (paper Fig. 7) |
| `scripts/redraw_keyword_figures.py` | Regenerate keyword contribution figures |

## Checkpoints

Trained checkpoints (KG-TPCA `kg_enhanced_consistency_v3`, TPCA `biomedclip_v2_with_consistency`)
are available from the authors upon request. They are not distributed in this repository.

## Citation

The manuscript is under review. If you use this code, please cite the repository, the underlying
models, and the datasets:

```bibtex
@misc{kgtpca2026,
  title  = {KG-TPCA: Knowledge-Graph-Enhanced Text Prompt Learning with Cross-View Consistency for Mammography Diagnosis},
  author = {Du, Tongchun and colleagues},
  year   = {2026},
  note   = {Code repository}
}

@article{zhang2024biomedclip,
  title   = {BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs},
  author  = {Zhang, Sheng and others},
  journal = {NEJM AI},
  year    = {2024}
}

@inproceedings{chen2024mammoclip,
  title     = {Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography},
  author    = {Chen, Shuhui and others},
  booktitle = {MICCAI},
  year      = {2024}
}

@article{moreira2012inbreast,
  title   = {INbreast: toward a full-field digital mammographic database},
  author  = {Moreira, In{\^e}s C and Amaral, Igor and Domingues, In{\^e}s and others},
  journal = {Academic Radiology},
  volume  = {19},
  number  = {2},
  pages   = {236--248},
  year    = {2012}
}

@misc{pham2022vindrmammo,
  title   = {VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography (version 1.0.0)},
  author  = {Pham, H. H. and Nguyen Trung, H. and Nguyen, H. Q.},
  publisher = {PhysioNet},
  year    = {2022},
  doi     = {10.13026/br2v-7517}
}
```

## License

MIT License. See [LICENSE](LICENSE). The third-party models (BiomedCLIP, Mammo-CLIP) and datasets
(CBIS-DDSM, INbreast, VinDr-Mammo) are subject to their own licenses and access agreements.
