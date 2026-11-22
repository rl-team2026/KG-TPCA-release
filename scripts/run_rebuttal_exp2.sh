#!/usr/bin/env bash
# Stage 4 rebuttal experiments (Reviewer 2): prior robustness, multi-seed,
# no-text ablation, image+mask linear probe.
set -u
cd /home/rl/du/x_ray_postdoc

PY=/home/rl/anaconda3/envs/mammo-clip-env/bin/python
export PYTHONPATH=/home/rl/du/x_ray_postdoc

echo "[$(date +%H:%M:%S)] === STAGE 4: prior robustness + multi-seed + no-text ==="
$PY scripts/50_prior_robustness.py \
    --runs seed123_kg seed456_kg seed123_base seed456_base \
           shuffled_prior noisy_prior shuffled_rel no_text \
    --output-dir outputs/prior_robustness \
    > outputs/prior_robustness_driver.log 2>&1
echo "[$(date +%H:%M:%S)] === STAGE 4b: image+mask linear probe ==="
$PY scripts/51_probe_masks.py \
    --output-dir outputs/probe_masks \
    > outputs/probe_masks_driver.log 2>&1
echo "[$(date +%H:%M:%S)] ALL STAGE 4 DONE"
