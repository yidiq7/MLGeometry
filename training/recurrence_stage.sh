#!/bin/bash
# The first stage 2 of the own route: the units of a learned recurrence with the base weight, on the
# bihomogeneous stage 1 of compare_stages.sh, against a tanh stage 2 with a linear output.
# Run it from the root folder of the repository after compare_stages.sh. Set PYTHON to the
# interpreter of the environment. All runs use the stage seed 43 (--seed 42).
PYTHON=${PYTHON:-python}
export PYTHONPATH=src
DATA="--fit data/quintic_fit_500k.npz --validation data/quintic_val_50k.npz --holdout data/quintic_holdout_50k.npz"
BASE=runs/cmp/s1_bihom_seed42
REC="$PYTHON -u training/recurrence_stage.py $DATA --load_dir $BASE --seed 42 --degree 16"

# Random A and B and only c solved: three widths, a second draw, and the Fubini-Study features.
for w in 64 128 256; do
    $REC --width $w --features base --save_dir runs/rec/R$w
done
$REC --width 128 --features base --draw_seed 44 --save_dir runs/rec/R128_draw44
$REC --width 128 --features fs --save_dir runs/rec/R128_fs

# A and B of width 16 fitted by LM with variable projection.
$REC --width 16 --features base --lm_steps 200 --lm_patience 10 --lm_min_delta 0.001 --save_dir runs/rec/T16

# The tanh stage 2 with a linear output: Adam, then the exact solve, then LM.
$PYTHON -u training/spectral_stages.py $DATA --start_stage 2 --stages 2 --load_dir $BASE --seed 42 \
    --network tanh_linear --layers 64_64_64 --out_std 1e-3 --learning_rate 1e-3 \
    --adam_loss weighted_MAPE_amp_scaled --adam_epochs 5000 --adam_patience 5000 \
    --val_patience 300 --val_min_delta 0.005 --output_solves 3 --lm_steps 200 --lm_patience 10 \
    --lm_min_delta 0.001 --lbfgs_epochs 0 --save_dir runs/rec/tanh
