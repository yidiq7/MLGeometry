#!/bin/bash
# Part 1 of the multi-stage work: Fubini-Study, then three spectral stages on the Fermat quintic.
# Run it from the root folder of the repository. Set PYTHON to the interpreter of the environment.
# The three chains do not depend on each other, so they can run at the same time on three GPUs.
PYTHON=${PYTHON:-python}
export PYTHONPATH=src

# The stored datasets: a fit set and a hold-out set of 50,000 points, and a fit set of 200,000 points.
for spec in "fit_50k 10000 20261017" "holdout_50k 10000 20261018" "fit_200k 40000 20261019"; do
    set -- $spec
    if [ ! -f "data/quintic_$1.npz" ]; then
        $PYTHON -u training/generate_data.py --n_pairs $2 --seed $3 --out "data/quintic_$1.npz"
    fi
done

DATA="--fit data/quintic_fit_50k.npz --holdout data/quintic_holdout_50k.npz"
STOP="--adam_epochs 20000 --adam_patience 500 --adam_min_delta 0.01
      --lbfgs_epochs 10000 --lbfgs_patience 200 --lbfgs_min_delta 1e-4"

# Chain A: Adam and L-BFGS on MAPE. Each stage also runs L-BFGS on MSPE from the same Adam result.
$PYTHON -u training/spectral_stages.py $DATA $STOP --save_dir runs/part1/chain_mape \
    --stages 3 --layers 64_64_64 --seed 42 \
    --lbfgs_losses weighted_MAPE_amp_scaled weighted_MSPE_amp_scaled

# Chain B: Adam on MAPE, then L-BFGS on MSPE, on the stored metric of chain B.
$PYTHON -u training/spectral_stages.py $DATA $STOP --save_dir runs/part1/chain_mspe \
    --stages 3 --layers 64_64_64 --seed 42 \
    --lbfgs_losses weighted_MSPE_amp_scaled

# The SIREN chain with float64 parameters, and its stage 3 again with float32 SirenDense parameters.
$PYTHON -u training/spectral_stages.py $DATA $STOP --save_dir runs/part1/chain_siren \
    --stages 3 --layers 64_64_64 --seed 42 --network siren \
    --lbfgs_losses weighted_MAPE_amp_scaled
amp=$($PYTHON -c "import json; print(json.load(open('runs/part1/chain_siren/stage3.json'))['amp'])")
$PYTHON -u training/spectral_stages.py $DATA $STOP --save_dir runs/part1/siren_stage3_float32 \
    --start_stage 3 --stages 3 --load_dir runs/part1/chain_siren --amp $amp \
    --layers 64_64_64 --seed 42 --network siren --siren_float32 \
    --lbfgs_losses weighted_MAPE_amp_scaled

# Step A: one stage of chain A again on the fit set of 200,000 points. The stage keeps the
# amp and the seed of the chain, so only the fit set changes.
for stage in 1 2 3; do
    amp=$($PYTHON -c "import json; print(json.load(open('runs/part1/chain_mape/stage${stage}.json'))['amp'])")
    $PYTHON -u training/spectral_stages.py --fit data/quintic_fit_200k.npz \
        --holdout data/quintic_holdout_50k.npz $STOP \
        --save_dir runs/part1/stage${stage}_200k \
        --start_stage $stage --stages $stage --load_dir runs/part1/chain_mape \
        --layers 64_64_64 --seed 42 --amp $amp \
        --lbfgs_losses weighted_MAPE_amp_scaled
done
