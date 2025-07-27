#!/bin/bash

Direct_dataset=( 'chameleon' 'arxiv-year'  'snap-patents' 'directed-roman-empire' )  #

# Run sequentially
for Didataset in "${Direct_dataset[@]}"; do
    echo "Running with $Didataset..."
    python -m  src.run   --dataset="$Didataset"
    echo "Finished $layer."
done

echo "All runs completed."