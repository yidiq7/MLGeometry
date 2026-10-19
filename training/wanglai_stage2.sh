#!/bin/bash
# The Wang-Lai control at stage 2 (arXiv:2307.08934): a stage 2 with a scaled sine first layer and
# tanh layers, against the plain tanh stage 2, on the same spectral tanh stage 1 and the same points.
# Run it from the root folder of the repository after compare_stages.sh. Set PYTHON to the
# interpreter of the environment. Each phase returns the parameters with the lowest validation sigma.
PYTHON=${PYTHON:-python}
export PYTHONPATH=src
RUN="$PYTHON -u training/spectral_stages.py"

DATA="--fit data/quintic_fit_500k.npz --validation data/quintic_val_50k.npz --holdout data/quintic_holdout_50k.npz"
STAGE="--start_stage 2 --stages 2 --load_dir runs/cmp/s1_tanh_seed42 --layers 64_64_64 --out_std 1e-3"
# Looser stop rules than compare_stages.sh. Only the validation rule and the cap stop Adam.
STOP="--adam_loss weighted_MAPE_amp_scaled --adam_epochs 5000 --adam_patience 5000
      --val_patience 300 --val_min_delta 0.005
      --lbfgs_losses weighted_MSPE_amp_scaled --lbfgs_epochs 20000 --lbfgs_patience 500 --lbfgs_min_delta 1e-4"
SHORT="--lr_search_epochs 50 --lr_search_only"
OUT=runs/wanglai

# The learning rate of the search of a run folder with the lowest validation sigma.
best_lr() {
    $PYTHON -c "import json; s=json.load(open('$1/summary.json')); print(min(s['stages'][-1]['lr_search'], key=lambda r: r['best_val_sigma'])['lr'])"
}
# The kappa of the run folder with the lowest validation sigma in its short run.
best_kappa() {
    $PYTHON -c "import json, sys; st=[json.load(open(d + '/summary.json'))['stages'][-1] for d in sys.argv[1:]]; print(min(st, key=lambda s: s['lr_search'][0]['best_val_sigma'])['kappa'])" "$@"
}
holdout_sigma() {
    $PYTHON -c "import json; print(json.load(open('$1/summary.json'))['stages'][-1]['eval']['holdout']['mape'])"
}

# The plain tanh stage 2: the learning rate by short runs.
$RUN $DATA $STOP $STAGE --network tanh --seed 42 --lr_search 1e-3,3e-4,1e-4 $SHORT --save_dir $OUT/lr_tanh
LR_TANH=$(best_lr $OUT/lr_tanh)

# The Wang-Lai stage 2: kappa by short runs with the learning rate 1e-3, then the learning rate.
for kappa in 1 2 4 8 16 32; do
    $RUN $DATA $STOP $STAGE --network wanglai --kappa $kappa --seed 42 --lr_search 1e-3 $SHORT \
        --save_dir $OUT/kappa_$kappa
done
KAPPA=$(best_kappa $OUT/kappa_1 $OUT/kappa_2 $OUT/kappa_4 $OUT/kappa_8 $OUT/kappa_16 $OUT/kappa_32)
$RUN $DATA $STOP $STAGE --network wanglai --kappa $KAPPA --seed 42 --lr_search 1e-3,3e-4,1e-4 $SHORT \
    --save_dir $OUT/lr_wanglai
LR_WANGLAI=$(best_lr $OUT/lr_wanglai)

# The full runs with the stage seed 43 (--seed 42). With a difference of the hold-out sigma below
# 8 %, two times the spread between the seeds of compare_stages.sh, both run again with the stage seed 44.
for seed in 42 43; do
    $RUN $DATA $STOP $STAGE --network tanh --seed $seed --learning_rate $LR_TANH --save_dir $OUT/s2_tanh_seed$seed
    $RUN $DATA $STOP $STAGE --network wanglai --kappa $KAPPA --seed $seed --learning_rate $LR_WANGLAI \
        --save_dir $OUT/s2_wanglai_seed$seed
    close=$($PYTHON -c "import sys; a, b = map(float, sys.argv[1:]); print(int(abs(b - a) < 0.08 * a))" \
            $(holdout_sigma $OUT/s2_tanh_seed$seed) $(holdout_sigma $OUT/s2_wanglai_seed$seed))
    [ "$close" = 1 ] || break
done
