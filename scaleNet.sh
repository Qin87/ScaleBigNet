#!/bin/bash

Direct_dataset=( 'chameleon' 'arxiv-year'  'snap-patents' 'directed-roman-empire' )  #

# Run sequentially
for Didataset in "${Direct_dataset[@]}"; do
  for k in 1 2 3 4 5; do
    echo "Running with $Didataset..."
    python -m  src.run   --dataset="$Didataset"  --k_plus="$k"  --use_best_hyperparams
    echo "Finished $layer."
done
done

echo "All runs completed."