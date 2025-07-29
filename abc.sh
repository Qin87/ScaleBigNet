#!/bin/bash

Direct_dataset=('arxiv-year' 'snap-patents' 'directed-roman-empire')
alphas=(0 0.5 1 -1)
betas=(0 0.5 1 -1)
gammas=(0 0.5 1 -1)

Weight_penalties=("exp" "lin" "None")

for dataset in "${Direct_dataset[@]}"; do
    for penalty in "${Weight_penalties[@]}"; do
    for alpha in "${alphas[@]}"; do
        for beta in "${betas[@]}"; do
            for gamma in "${gammas[@]}"; do
                echo "Running on dataset=${dataset} with alpha=${alpha}, beta=${beta}, gamma=${gamma}"
                python run.py --dataset "$dataset" --alpha "$alpha" --beta "$beta" --gamma "$gamma"  --weight_penalty="$penalty"
            done
            done
        done
    done
done
