#!/bin/bash
for psi in 0.5 ; do
    for loss_func in "weighted_MAPE" ; do
        for layers in "300_300_300_1"; do
            python bihomoNN_train.py --seed 1234 \
                                     --n_pairs 100000\
                                     --batch_size 5000\
                                     --function "f0" \
                                     --psi $psi \
                                     --layers $layers \
                                     --load_model "f0_psi${psi}/${layers}" \
                                     --save_dir "experiments.yidi/train_curve/f0_psi${psi}/" \
                                     --save_name "${layers}" \
                                     --optimizer 'lbfgs'\
                                     --learning_rate 0.001 \
                                     --decay_rate 1 \
                                     --max_epochs 1000\
                                     --loss_func ${loss_func}
        done
    done
done
