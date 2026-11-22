#!/usr/bin/env bash
# Master driver for the AAMAS rebuttal experiments.
# Stage 1: lambda grid remainder (skips existing)
# Stage 2: aggregation ablation (attention/bilinear, with/without KG)
# Stage 3: CoOp-family baselines
set -u
cd /home/rl/du/x_ray_postdoc

PY=/home/rl/anaconda3/envs/mammo-clip-env/bin/python
export PYTHONPATH=/home/rl/du/x_ray_postdoc
SWEEP_LOG=outputs/lambda_sweep/master.log

echo "[$(date +%H:%M:%S)] === STAGE 1: lambda sweep ===" | tee -a "$SWEEP_LOG"

run_sweep_one() {
    lp=$1; lpp=$2
    outdir=outputs/lambda_sweep/prior_${lp}_pair_${lpp}
    metrics=$outdir/kg_enhanced/metrics.json
    if [ -f "$metrics" ]; then
        auroc=$($PY -c "import json;print(round(json.load(open('$metrics'))['auroc'],4))")
        ece=$($PY -c "import json;print(round(json.load(open('$metrics'))['ece'],4))")
        echo "skip lp=$lp lpp=$lpp (exists, AUROC=$auroc ECE=$ece)" | tee -a "$SWEEP_LOG"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start lp=$lp lpp=$lpp" | tee -a "$SWEEP_LOG"
    $PY scripts/46_kg_encoder_ablation.py \
        --encoders kg_enhanced \
        --kg-prior "$lp" --kg-pairwise "$lpp" \
        --output-dir "$outdir" >> "$outdir.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$metrics" ]; then
        auroc=$($PY -c "import json;print(round(json.load(open('$metrics'))['auroc'],4))")
        ece=$($PY -c "import json;print(round(json.load(open('$metrics'))['ece'],4))")
        echo "[$(date +%H:%M:%S)] done lp=$lp lpp=$lpp AUROC=$auroc ECE=$ece" | tee -a "$SWEEP_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAILED lp=$lp lpp=$lpp rc=$rc" | tee -a "$SWEEP_LOG"
    fi
}

for lp in 0.005 0.01 0.02 0.05; do
    for lpp in 0.0025 0.005 0.01; do
        if [ "$lp" = "0.01" ] && [ "$lpp" = "0.005" ]; then
            echo "skip lp=0.01 lpp=0.005 (paper reference: kg_enhanced 0.8404/0.0782)" | tee -a "$SWEEP_LOG"
            continue
        fi
        run_sweep_one "$lp" "$lpp"
    done
done
echo "[$(date +%H:%M:%S)] === lambda sweep finished ===" | tee -a "$SWEEP_LOG"

echo "[$(date +%H:%M:%S)] === STAGE 2: aggregation ablation ==="
$PY scripts/48_aggregation_ablation.py \
    --runs attention_kg attention_nokg bilinear_kg bilinear_nokg \
    --output-dir outputs/aggregation_ablation \
    > outputs/aggregation_ablation_driver.log 2>&1
echo "[$(date +%H:%M:%S)] === aggregation ablation finished ==="

echo "[$(date +%H:%M:%S)] === STAGE 3: CoOp baselines ==="
$PY scripts/49_coop_baseline.py \
    --runs coop_cosine coop_gate cocoop_gate \
    --output-dir outputs/coop_baselines \
    > outputs/coop_baselines_driver.log 2>&1
echo "[$(date +%H:%M:%S)] === CoOp baselines finished ==="

echo "[$(date +%H:%M:%S)] ALL STAGES DONE"
