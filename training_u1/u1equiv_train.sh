#!/bin/bash

mkdir -p cache

# --layers has now two parts separated by '-'
# first is layers inside the U1-equivariant network
# second is layers inside the U1-invariant network

# --g_steps is size of the finite subgroup of U1

for psi in 0.5 ; do
    for loss_func in "weighted_MAPE" ; do
        for layers in "20_20-300_300_300_1"; do
            python u1equiv_train.py --seed 1234 \
                                     --n_pairs 100000\
                                     --batch_size 5000\
                                     --function "f0" \
                                     --psi $psi \
                                     --layers $layers \
                                     --g_steps 16 \
                                     --save_dir "experiments-marek" \
                                     --save_name "${layers}" \
                                     --optimizer 'adam'\
                                     --learning_rate 0.001 \
                                     --decay_rate 1 \
                                     --max_epochs 1000 \
                                     --loss_func ${loss_func} \
                                     --cache_folder "cache"
        done
    done
done
