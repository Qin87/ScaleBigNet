#!/bin/bash

#Direct_dataset=("pokec"  "actor"  "syn-dir")  #  'snap-patents' 'directed-roman-empire'
Direct_dataset=("penn94" "reed98"  "cornell115" "johnshopkins55" "genius")  #  'snap-patents' 'directed-roman-empire'
Num_layers=(1 2 3 4 5 6 7 8)

alphas=(0 0.5 1 -1)
betas=(0 0.5 1 -1)
gammas=(0 0.5 1 -1)

Weight_penalties=("None")   # "exp" "lin"


for dataset in "${Direct_dataset[@]}"; do
    for layers in "${Num_layers[@]}"; do
    for penalty in "${Weight_penalties[@]}"; do
    for alpha in "${alphas[@]}"; do
        for beta in "${betas[@]}"; do
            for gamma in "${gammas[@]}"; do
                echo "Running on dataset=${dataset} with alpha=${alpha}, beta=${beta}, gamma=${gamma}"
                python3 run.py --dataset "$dataset" --num_runs 1   --num_layers "$layers"  --alpha "$alpha" --beta "$beta" --gamma "$gamma"  --weight_penalty="$penalty"
            done
            done
            done
        done
    done
done
