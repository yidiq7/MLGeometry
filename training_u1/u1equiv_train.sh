#!/bin/bash

mkdir -p cache

# --layers has now two parts separated by '-'
# first is layers inside the U1-equivariant network
# second is layers inside the U1-invariant network

# --g_steps is size of the finite subgroup of U1

for psi in 0.5 ; do
    for loss_func in "weighted_MAPE" ; do
        for layers in "64_128_256_512_1024-512_512_512_1"; do
            python u1equiv_train.py --seed 1234 \
                                     --n_pairs 100000 \
                                     --batch_size 5000 \
                                     --function "f0" \
                                     --psi $psi \
                                     --layers $layers \
                                     --model_name u1_model_tanh \
                                     --g_steps 128 \
                                     --save_dir "experiments-marek" \
                                     --save_name "${layers}" \
                                     --optimizer 'adam' \
                                     --learning_rate 0.001 \
                                     --decay_rate 0.993 \
                                     --max_epochs 500 \
                                     --loss_func ${loss_func} \
                                     --cache_folder "cache" \
                                     --mem_limit 32
        done
    done
done >log_$(basename $0).txt 2>&1
