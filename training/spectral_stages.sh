#!/bin/bash
# Part 1 of the multi-stage work: Fubini-Study, then three spectral stages on the Fermat quintic.
# Run it from the root folder of the repository. Set PYTHON to the interpreter of the environment.
PYTHON=${PYTHON:-python}
export PYTHONPATH=src

# The stored datasets: a fit set and a hold-out set of 50,000 points, and a fit set of 200,000 points.
for spec in "fit_50k 10000 20261017" "holdout_50k 10000 20261018" "fit_200k 40000 20261019"; do
    set -- $spec
    if [ ! -f "data/quintic_$1.npz" ]; then
        $PYTHON -u training/generate_data.py --n_pairs $2 --seed $3 --out "data/quintic_$1.npz"
    fi
done

# The chain of three stages. Each stage also runs L-BFGS on MSPE from the same Adam result.
$PYTHON -u training/spectral_stages.py --fit data/quintic_fit_50k.npz \
    --holdout data/quintic_holdout_50k.npz \
    --save_dir runs/part1/chain_50k \
    --stages 3 --layers 64_64_64 --seed 42 \
    --adam_epochs 2000 --adam_patience 50 --adam_min_delta 0.01 \
    --batch_size 1000 --learning_rate 0.001 \
    --lbfgs_losses weighted_MAPE_amp_scaled weighted_MSPE_amp_scaled \
    --lbfgs_epochs 1000 --lbfgs_patience 50 --lbfgs_min_delta 1e-4

# One stage again on the fit set of 200,000 points, on top of the earlier stages of the chain.
for stage in 1 2 3; do
    $PYTHON -u training/spectral_stages.py --fit data/quintic_fit_200k.npz \
        --holdout data/quintic_holdout_50k.npz \
        --save_dir runs/part1/stage${stage}_200k \
        --start_stage $stage --stages $stage --load_dir runs/part1/chain_50k \
        --layers 64_64_64 --seed 42 \
        --lbfgs_losses weighted_MAPE_amp_scaled
done
