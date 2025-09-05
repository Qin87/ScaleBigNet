#!/bin/bash

Direct_dataset=( 'snap-patents' )  # 'arxiv-year'  'snap-patents' 'directed-roman-empire'

K_values=(1 2 3 )
Weight_penalties=(  "lin" )

# Run sequentially
for Didataset in "${Direct_dataset[@]}"; do
    for k in "${K_values[@]}"; do
        for penalty in "${Weight_penalties[@]}"; do
            echo "Running with dataset=$Didataset, k_plus=$k, weight_penalty=$penalty..."
            python3 run.py \
                --dataset="$Didataset" \
                --seed="$k"
#                --weight_penalty="$penalty"
#                --use_best_hyperparams
            echo "Finished dataset=$Didataset, k_plus=$k, weight_penalty=$penalty."
        done
    done
done

echo "All runs completed."