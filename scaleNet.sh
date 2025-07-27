#!/bin/bash

Direct_dataset=( 'chameleon' 'arxiv-year'  'snap-patents' 'directed-roman-empire' )  #

K_values=(1 2 3 4 5)
Weight_penalties=("None" "exp" "lin" )

# Run sequentially
for Didataset in "${Direct_dataset[@]}"; do
    for k in "${K_values[@]}"; do
        for penalty in "${Weight_penalties[@]}"; do
            echo "Running with dataset=$Didataset, k_plus=$k, weight_penalty=$penalty..."
            python -m src.run \
                --dataset="$Didataset" \
                --k_plus="$k" \
                --weight_penalty="$penalty" \
                --use_best_hyperparams
            echo "Finished dataset=$Didataset, k_plus=$k, weight_penalty=$penalty."
        done
    done
done

echo "All runs completed."