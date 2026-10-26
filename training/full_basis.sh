#!/bin/bash
# The full bases of low degree on the Fermat quintic, with no symmetry: an HN base at k0 = 3, the
# corrections of the levels 4, 5 and 4 + 5 with the base weight and of the level 5 with the FS weight,
# HN at degree 5 from the lift of the base, and the tanh stage 2 on the same base.
# Run it from the root folder of the repository after compare_stages.sh, which writes the datasets.
# Set PYTHON to the interpreter of the environment.
PYTHON=${PYTHON:-python}
export PYTHONPATH=src
DATA="--fit data/quintic_fit_500k.npz --validation data/quintic_val_50k.npz --holdout data/quintic_holdout_50k.npz"
OUT=runs/lowdeg
FB="$PYTHON -u training/full_basis.py $DATA --sections $OUT/sections.npz"

# The orthonormal sections of the degrees 3, 4 and 5, and the HN base at k0 = 3 from H = I.
$FB --task sections --degrees 3 4 5 --save_dir $OUT/sections
$FB --task hn --degree 3 --k0 3 --save_dir $OUT/hn3

# The corrections on the base, each solved exactly. The level 5 with the base weight also gives
# sigma after its C is truncated to each rank.
$FB --task correction --base_dir $OUT/hn3 --levels 4 --weight base --save_dir $OUT/b4
$FB --task correction --base_dir $OUT/hn3 --levels 5 --weight base --ranks 5 10 20 40 80 125 --save_dir $OUT/b5
$FB --task correction --base_dir $OUT/hn3 --levels 4 5 --weight base --save_dir $OUT/b45
$FB --task correction --base_dir $OUT/hn3 --levels 5 --weight fs --save_dir $OUT/f5

# HN at degree 5 in the class 3 H, from the lift of the base.
$FB --task hn --degree 5 --k0 3 --base_dir $OUT/hn3 --save_dir $OUT/hn5

# The tanh stage 2 with a linear output on the same base: Adam, then 3 exact solves, then LM.
$PYTHON -u training/spectral_stages.py $DATA --start_stage 2 --stages 2 --load_dir $OUT/hn3 --seed 42 \
    --network tanh_linear --layers 64_64_64 --out_std 1e-3 --learning_rate 1e-3 \
    --adam_loss weighted_MAPE_amp_scaled --adam_epochs 5000 --adam_patience 5000 \
    --val_patience 300 --val_min_delta 0.005 --output_solves 3 --lm_steps 200 --lm_patience 10 \
    --lm_min_delta 0.001 --lbfgs_epochs 0 --save_dir $OUT/tanh
