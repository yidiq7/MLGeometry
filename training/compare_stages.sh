#!/bin/bash
# A controlled comparison of networks at stage 1 and stage 2, each with about 10,000 parameters.
# Run it from the root folder of the repository. Set PYTHON to the interpreter of the environment.
# Each phase returns the parameters with the lowest sigma on the validation set; the hold-out
# set gives the final numbers. The runs of one stage do not depend on each other.
PYTHON=${PYTHON:-python}
export PYTHONPATH=src
RUN="$PYTHON -u training/spectral_stages.py"

# A fit set of 500,000 points and a validation set of 50,000 points, with new seeds.
for spec in "fit_500k 100000 20261020" "val_50k 10000 20261021" "holdout_50k 10000 20261018"; do
    set -- $spec
    if [ ! -f "data/quintic_$1.npz" ]; then
        $PYTHON -u training/generate_data.py --n_pairs $2 --seed $3 --out "data/quintic_$1.npz"
    fi
done

DATA="--fit data/quintic_fit_500k.npz --validation data/quintic_val_50k.npz --holdout data/quintic_holdout_50k.npz"
STOP="--adam_loss weighted_MAPE_amp_scaled --adam_epochs 20000 --adam_patience 500 --adam_min_delta 0.01
      --val_patience 50 --val_min_delta 0.01
      --lbfgs_losses weighted_MSPE_amp_scaled --lbfgs_epochs 10000 --lbfgs_patience 200 --lbfgs_min_delta 1e-4"
LRS="--lr_search 1e-3,3e-4,1e-4 --lr_search_epochs 50"

net_args() {
    case "$1" in
        tanh) echo "--network tanh --layers 64_64_64 --out_std 1e-3";;
        poly8) echo "--network poly --branches 64_64_64 --out_std 1e-3";;
        poly4816) echo "--network poly --branches 34_34+34_34_34+34_34_34_34 --out_std 1e-3";;
        bihom) echo "--network bihom --layers 64_64_64 --base none";;
        s2_tanh) echo "--network tanh --layers 64_64_64 --out_std 1e-3";;
        s2_polybase) echo "--network poly --branches $S2BRANCHES --weight base --out_std 1e-3";;
        s2_polyfs) echo "--network poly --branches $S2BRANCHES --weight fs --out_std 1e-3";;
    esac
}
chosen_lr() {
    $PYTHON -c "import json; s=json.load(open('$1/summary.json')); print(min(s['stages'][-1]['lr_search'], key=lambda r: r['best_val_sigma'])['lr'])"
}
val_sigma() {
    $PYTHON -c "import json; print(json.load(open('$1/summary.json'))['stages'][-1]['eval']['validation']['mape'])"
}

# Stage 1 on FS (tanh, poly8, poly4816) or from no base (bihom): a learning rate search, then two seeds.
for c in tanh poly8 poly4816 bihom; do
    $RUN $DATA $STOP $(net_args $c) --stages 1 --seed 42 $LRS --lr_search_only --save_dir runs/cmp/lr_s1_$c
    for seed in 42 43; do
        $RUN $DATA $STOP $(net_args $c) --stages 1 --seed $seed --learning_rate $(chosen_lr runs/cmp/lr_s1_$c) \
            --save_dir runs/cmp/s1_${c}_seed$seed
    done
done

# Stage 2 on the stage 1 run with the lowest validation sigma. The polynomial candidates use the
# branches of the polynomial stage 1 candidate with the lower validation sigma.
BEST=$(for r in runs/cmp/s1_*_seed*; do echo "$(val_sigma $r) $r"; done | sort -g | head -1 | cut -d' ' -f2)
S2BRANCHES="64_64_64"
if [ "$(for r in runs/cmp/s1_poly*_seed*; do echo "$(val_sigma $r) $r"; done | sort -g | head -1 | grep -c poly4816)" = 1 ]; then
    S2BRANCHES="34_34+34_34_34+34_34_34_34"
fi
export S2BRANCHES
for c in s2_tanh s2_polybase s2_polyfs; do
    $RUN $DATA $STOP $(net_args $c) --start_stage 2 --stages 2 --load_dir $BEST --seed 42 $LRS \
        --lr_search_only --save_dir runs/cmp/lr_$c
    for seed in 42 43; do
        $RUN $DATA $STOP $(net_args $c) --start_stage 2 --stages 2 --load_dir $BEST --seed $seed \
            --learning_rate $(chosen_lr runs/cmp/lr_$c) --save_dir runs/cmp/${c}_seed$seed
    done
done
