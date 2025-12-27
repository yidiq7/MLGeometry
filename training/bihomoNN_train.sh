#!/bin/bash
for psi in 0.5 ; do
    for loss_func in "weighted_MAPE" ; do
        for layers in "64_256_1024_1"; do
            python bihomoNN_train.py --seed 1234 \
                                     --n_pairs 100000\
                                     --batch_size 1000\
                                     --function "f0" \
                                     --psi $psi \
                                     --layers $layers \
                                     --load_model "trained_models/f0_psi${psi}/${layers}.pkl" \
                                     --save_dir "trained_models/f0_psi${psi}/" \
                                     --save_name "${layers}_adam" \
                                     --optimizer 'Adam'\
                                     --learning_rate 0.001 \
                                     --decay_rate 1 \
                                     --max_epochs 300\
                                     --loss_func ${loss_func}

            python bihomoNN_train.py --seed 1234 \
                                     --n_pairs 100000\
                                     --batch_size 1000\
                                     --function "f0" \
                                     --psi $psi \
                                     --layers $layers \
                                     --load_model "trained_models/f0_psi${psi}/${layers}_adam.pkl" \
                                     --save_dir "trained_models/f0_psi${psi}/" \
                                     --save_name "${layers}_lbfgs" \
                                     --optimizer 'LBFGS'\
                                     --max_epochs 50\
                                     --loss_func ${loss_func}
        done
    done
done
