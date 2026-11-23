# KG-TPCA: Knowledge Graph Constrained Text Prompt Adaptation for Mammography Benign-Malignant Classification

Official code for **KG-TPCA**, a BI-RADS-prior multimodal framework for benign–malignant classification of breast lesions in dual-view (CC/MLO) mammography.

KG-TPCA builds on a frozen [BiomedCLIP](https://aka.ms/biomedclip-paper) vision–language backbone and introduces:

- **Knowledge-graph (KG) enhanced prompt encoder**: 16 BI-RADS lesion keywords are embedded under two prior constraints — a malignancy-association prior loss ($L_{\text{prior}}$) and a semantic relation loss ($L_{\text{rel}}$) — producing structured, clinically interpretable prompt vectors;
- **Prompt attention gate**: text prompts condition visual features through residual channel-wise modulation (FiLM), guiding the model toward lesion-relevant regions;
- **Dual-view pipeline**: each CC/MLO image is encoded as both the original and the lesion mask-enhanced form, adaptively fused by a learnable gate, and averaged across views for the final prediction. The KG regularization is computed only during training and adds no inference overhead.

All experiments use a small-sample protocol: the BiomedCLIP backbone is frozen and only the prompt/gate/classifier modules (≈290K parameters) are trained on CBIS-DDSM. External generalization is validated on INbreast and VinDr-Mammo **without any retraining**.

> **Disclaimer**: This is retrospective research code. It is not a clinical diagnostic device and must not be used for patient care.

## Key Results

### In-domain: CBIS-DDSM (official train/test split, 266 test pairs)

| Method | AUROC | AUPRC | F1 | Sens | Spec | ECE | Trainable params |
|---|---|---|---|---|---|---|---|
| ResNet50 (ImageNet, fine-tuned) | 0.7059 | 0.6100 | 0.6190 | 0.8426 | 0.3987 | 0.3375 | 23.5M |
| BiomedCLIP zero-shot (text+image) | 0.5742 | 0.4812 | 0.5674 | 0.9352 | 0.0696 | 0.0957 | 0 |
| BiomedCLIP + text prompt learning (16 keywords) | 0.7970 | 0.7170 | 0.6667 | 0.6111 | 0.8481 | 0.0967 | ~34K |
| CoOp (context tokens + cosine scoring) | 0.6918 | 0.6211 | 0.5446 | 0.5093 | 0.7532 | 0.0754 | ~8K |
| CoOp-style context prompt (FiLM pipeline) | 0.7535 | 0.6863 | 0.6063 | 0.6204 | 0.7089 | 0.0873 | ~1.19M |
| CoCoOp-style visual-conditioned prompt | 0.7181 | 0.6451 | 0.5646 | 0.5463 | 0.7342 | 0.2368 | ~1.46M |
| Mammo-CLIP zero-shot (dual-view) | 0.5788 | 0.5168 | 0.1626 | 0.0926 | 0.9684 | 0.0927 | 0 |
| Mammo-CLIP linear probe (dual-view) | 0.7157 | 0.6286 | 0.5929 | 0.6204 | 0.6772 | 0.1622 | ~1K |
| Mammo-CLIP fine-tuned (frozen backbone) | 0.7792 | 0.7048 | 0.6857 | 0.7778 | 0.6646 | 0.1023 | ~2.0M |
| **KG-TPCA (ours)** | **0.8404** | **0.7820** | **0.7156** | **0.7222** | **0.7975** | **0.0782** | **~290K** |

Statistical comparison on the 266 test pairs (two-sided DeLong test; 95% CI by 2,000-iteration bootstrap):

| Method | AUROC | 95% CI | DeLong p vs KG-TPCA |
|---|---|---|---|
| **KG-TPCA (ours)** | **0.8404** | 0.7907–0.8850 | — |
| BiomedCLIP + text prompt (16 keywords) | 0.7970 | 0.7424–0.8477 | 0.008 |
| CoOp-style context prompt (FiLM pipeline) | 0.7535 | 0.6934–0.8114 | 0.002 |
| CoCoOp-style visual-conditioned prompt | 0.7181 | 0.6525–0.7812 | <0.001 |
| CoOp (context tokens + cosine scoring) | 0.6918 | 0.6258–0.7547 | <0.001 |
| Image-only (no text) | 0.7246 | 0.6642–0.7847 | <0.001 |

### Ablation: KG regularization components

| KG Component | AUROC | Δ |
|---|---|---|
| Baseline (no KG) | 0.7970 | — |
| $L_{\text{prior}}$ only | 0.8247 | +0.0277 |
| $L_{\text{rel}}$ only | 0.8135 | +0.0165 |
| **Full KG (ours)** | **0.8404** | **+0.0434** |

### Ablation: keyword aggregation mechanism (with/without KG)

| Aggregation | KG | AUROC |
|---|---|---|
| Linear weighted average (paper) | No / Yes | 0.7970 / **0.8404** |
| Attention pooling | No / Yes | 0.7699 / 0.7688 |
| Low-rank bilinear second order | No / Yes | 0.7784 / 0.7789 |

### Sensitivity: regularization coefficients (12-configuration grid)

Test AUROC stays within **0.8400–0.8408** (span 0.0008) over $\lambda_{\text{prior}} \in \{0.005, 0.01, 0.02, 0.05\}$ and $\lambda_{\text{pairwise}} \in \{0.0025, 0.005, 0.01\}$; the published configuration (0.01/0.005, **0.8404**) is a representative interior point, not an isolated optimum. The 16 prior scores come from the BI-RADS lexicon and published clinical statistics, not from tuning on the test data.

### Robustness (prior perturbation, multi-seed, input ablation)

- Prior perturbations (scores reassigned, ±0.1 noise, relation targets reassigned) change test AUROC by at most **0.0011** (0.8404–0.8415).
- Three seeds (42/123/456): KG model 0.8260 ± 0.013 vs no-KG 0.8072 ± 0.010 (per-seed gains +0.0434 / −0.001 / +0.0138); the KG variant is never significantly worse at any seed.
- Input ablation: text prompts contribute +0.0724 over image-only input (0.7246); the KG constraints add a further +0.0434 under identical inputs; probes without text reach only 0.7065–0.7115.

### External validation (all models trained on CBIS-DDSM only, direct inference)

**INbreast** (191 CC/MLO paired breasts; BI-RADS 3 excluded, same protocol as the main experiments):

| Method | Training data | AUROC | AUPRC |
|---|---|---|---|
| Mammo-CLIP zero-shot (dual-view) | none (pretrained) | 0.5867 | 0.3179 |
| Mammo-CLIP linear probe (dual-view) | CBIS-DDSM | 0.4732 | 0.2528 |
| TPCA (no KG, ablation) | CBIS-DDSM | 0.6943 | 0.4210 |
| **KG-TPCA (ours)** | CBIS-DDSM | **0.8433** | **0.6497** |

**VinDr-Mammo, screening subset** (199 CC/MLO paired breasts: all BI-RADS 4/5 + 100 sampled BI-RADS 1/2 studies, seed 42):

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
cdf_vlm/aggregations.py   # attention-pooling and low-rank bilinear aggregation encoders (ablation)
cdf_vlm/cli/              # runnable entry points (training, evaluation, preprocessing)
scripts/                  # KG-TPCA training, rebuttal, and statistical analysis scripts
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
descriptions and labels) and the PNG images.

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
python scripts/46_kg_encoder_ablation.py --encoders kg_enhanced
```

Trains the KG-enhanced prompt encoder + prompt gate + MLP classifier. Checkpoints and metrics
are written to `outputs/encoder_ablation/kg_enhanced/` (the published run: AUROC 0.8404;
hyperparameters: lr 1e-4, batch 16, 20 epochs max with early stopping, kg-prior 0.01,
kg-pairwise 0.005).

### Baselines and ablations

```bash
python scripts/46_kg_encoder_ablation.py --encoders baseline structured   # 16-keyword / structured-field encoders
python scripts/48_aggregation_ablation.py                                 # attention / bilinear aggregation, with/without KG
python scripts/49_coop_baseline.py                                        # CoOp / CoOp-FiLM / CoCoOp-style baselines
python scripts/50_prior_robustness.py                                     # prior perturbation, multi-seed repeats, image-only
python scripts/51_probe_masks.py                                          # masked linear probes (input ablation)
```

The lambda sensitivity grid and rebuttal suites are driven by:

```bash
bash scripts/run_lambda_sweep.sh             # 12-configuration coefficient grid
bash scripts/run_rebuttal_experiments.sh     # aggregation ablation + CoOp baselines
bash scripts/run_rebuttal_exp2.sh            # prior robustness + probe masks
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
# KG-TPCA on INbreast / VinDr (published checkpoint: outputs/encoder_ablation/kg_enhanced/model.pt)
python -m cdf_vlm.cli.eval_kgtpca_external \
    --pair-csv data/inbreast_processed/breast_pairs.csv \
    --data-root data/inbreast_processed \
    --checkpoint outputs/encoder_ablation/kg_enhanced/model.pt \
    --output-dir outputs/kgtpca_inbreast_eval

# TPCA (no KG) on INbreast / VinDr
python -m cdf_vlm.cli.eval_tpca_external \
    --pair-csv data/inbreast_processed/breast_pairs.csv \
    --data-root data/inbreast_processed \
    --checkpoint outputs/encoder_ablation/baseline/model.pt \
    --output-dir outputs/tpca_inbreast_eval
```

Each run writes `metrics.json` (AUROC, AUPRC, F1, accuracy, sensitivity, specificity, ECE,
sensitivity@spec90) and `predictions.csv`. ROC/PR/calibration figures can be generated with
`python -m cdf_vlm.cli.plot_predictions --predictions <csv> --output-dir <dir>`.

## Statistical Analysis

```bash
python scripts/52_test_predictions_delong.py
```

Collects paired test-set predictions from the published checkpoints (KG-TPCA, 16-keyword baseline,
CoOp family, image-only, multi-seed runs) and computes two-sided DeLong tests and 2,000-iteration
bootstrap 95% confidence intervals. Results are written to `outputs/statistics/`.

## Analysis Scripts

| Script | Purpose |
|---|---|
| `scripts/45_keyword_shap_analysis.py` | Marginal contribution of each pathology keyword (paper Fig. 2/3) |
| `scripts/46_kg_encoder_ablation.py` | KG encoder component ablation (baseline / structured / kg-enhanced) |
| `scripts/48_aggregation_ablation.py` | Keyword aggregation mechanism ablation (linear / attention / bilinear, ± KG) |
| `scripts/49_coop_baseline.py` | CoOp-family baselines on frozen BiomedCLIP |
| `scripts/50_prior_robustness.py` | Prior perturbation, multi-seed repeats, image-only ablation |
| `scripts/51_probe_masks.py` | Masked linear probes (input fairness ablation) |
| `scripts/52_test_predictions_delong.py` | DeLong tests and bootstrap confidence intervals |
| `scripts/regenerate_tsne_cn.py` | t-SNE visualization of the feature space |
| `scripts/redraw_keyword_figures.py` | Regenerate keyword contribution figures |
| `scripts/make_fig1_vector_en.py` | English vector flowchart (PDF + 300 dpi PNG) |

## Checkpoints

Trained checkpoints (KG-TPCA `outputs/encoder_ablation/kg_enhanced/model.pt`, AUROC 0.8404)
are available from the authors upon request. They are not distributed in this repository.

## Citation

The manuscript is under review (AAMAS 2027 submission). If you use this code, please cite the
repository, the underlying models, and the datasets:

```bibtex
@misc{kgtpca2026,
  title  = {KG-TPCA: Knowledge Graph Constrained Text Prompt Adaptation for Mammography Benign-Malignant Classification},
  author = {Du, Tongchun and Gao, Jiahao and Liu, Tianqing and Liu, Jie},
  year   = {2026},
  note   = {Code repository}
}

@article{zhang2024biomedclip,
  title   = {BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs},
  author  = {Zhang, Sheng and others},
  journal = {NEJM AI},
  year    = {2024}
}

@inproceedings{ghosh2024mammoclip,
  title     = {Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography},
  author    = {Ghosh, Soham and others},
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
