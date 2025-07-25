#!/bin/bash

Direct_dataset=(     'arxiv-year'  'snap-patents' 'directed-roman-empire' )  #

# Run sequentially
for Didataset in "${Direct_dataset[@]}"; do
    echo "Running with $Didataset..."
    python -m  src.run   --dataset="$Didataset"     > "July17_${Didataset}.log" 2>&1
    echo "Finished seed $layer."
done

echo "All runs completed."