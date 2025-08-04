#!/bin/bash

Direct_dataset=('chameleon'  'squirrel')  #  'snap-patents' 'directed-roman-empire'
alphas=(1 -1)
betas=(1 -1)
gammas=(1 -1)

layers=(1 2 3 4 5 6 7 8)
k_plus=(1 2 3 4 5 6 7 8)

Weight_penalties=("None")   # "exp" "lin"


for dataset in "${Direct_dataset[@]}"; do
    for penalty in "${Weight_penalties[@]}"; do
    for layer in "${layers[@]}"; do
    for k in "${k_plus[@]}"; do
    for alpha in "${alphas[@]}"; do
        for beta in "${betas[@]}"; do
            for gamma in "${gammas[@]}"; do
                echo "Running on dataset=${dataset} with alpha=${alpha}, beta=${beta}, gamma=${gamma}"
                python3 run.py  --num_layers="$layer"  --k_plus="$k" --dataset "$dataset" --alpha "$alpha" --beta "$beta" --gamma "$gamma"  --weight_penalty="$penalty"
            done
            done
            done
            done
        done
    done
done
