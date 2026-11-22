#!/usr/bin/env bash
# Lambda sensitivity grid for KG-TPCA rebuttal (AAMAS 2027, Reviewer 1).
# Grid: lambda_prior x lambda_pairwise = 4 x 3 = 12 configs.
# The reference config (0.01, 0.005) already exists as outputs/encoder_ablation/kg_enhanced
# (AUROC 0.8404, ECE 0.0782), so it is skipped here; its numbers are reused.
set -u
cd /home/rl/du/x_ray_postdoc

PY=/home/rl/anaconda3/envs/mammo-clip-env/bin/python
export PYTHONPATH=/home/rl/du/x_ray_postdoc
SCRIPT=scripts/46_kg_encoder_ablation.py
OUT_ROOT=outputs/lambda_sweep
MASTER_LOG=$OUT_ROOT/master.log
mkdir -p "$OUT_ROOT"
echo "=== lambda sweep started $(date) ===" | tee -a "$MASTER_LOG"

run_one() {
    lp=$1; lpp=$2
    outdir=$OUT_ROOT/prior_${lp}_pair_${lpp}
    metrics=$outdir/kg_enhanced/metrics.json
    if [ -f "$metrics" ]; then
        auroc=$($PY -c "import json;print(round(json.load(open('$metrics'))['auroc'],4))")
        ece=$($PY -c "import json;print(round(json.load(open('$metrics'))['ece'],4))")
        echo "skip lp=$lp lpp=$lpp (exists, AUROC=$auroc ECE=$ece)" | tee -a "$MASTER_LOG"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start lp=$lp lpp=$lpp" | tee -a "$MASTER_LOG"
    $PY "$SCRIPT" \
        --encoders kg_enhanced \
        --kg-prior "$lp" --kg-pairwise "$lpp" \
        --output-dir "$outdir" \
        >> "$outdir.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$metrics" ]; then
        auroc=$($PY -c "import json;print(round(json.load(open('$metrics'))['auroc'],4))")
        ece=$($PY -c "import json;print(round(json.load(open('$metrics'))['ece'],4))")
        echo "[$(date +%H:%M:%S)] done lp=$lp lpp=$lpp AUROC=$auroc ECE=$ece" | tee -a "$MASTER_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAILED lp=$lp lpp=$lpp rc=$rc" | tee -a "$MASTER_LOG"
    fi
}

for lp in 0.005 0.01 0.02 0.05; do
    for lpp in 0.0025 0.005 0.01; do
        if [ "$lp" = "0.01" ] && [ "$lpp" = "0.005" ]; then
            echo "skip lp=0.01 lpp=0.005 (reference run exists: kg_enhanced)" | tee -a "$MASTER_LOG"
            continue
        fi
        run_one "$lp" "$lpp"
    done
done

echo "=== lambda sweep finished $(date) ===" | tee -a "$MASTER_LOG"
