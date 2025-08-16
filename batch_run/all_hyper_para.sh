#!/bin/bash

Direct_dataset=('penn94' )   # 'snap-patents' 'directed-roman-empire'
K_values=(2 3)
Weight_penalties=("None" "exp" "lin")
Dropouts=(0.0 0.5)
Hidden_dims=(64 32  )
Num_layers=(1 2 3 4 5 6 7 8)
Learning_rates=(0.001 0.0005)  # 0.1 0.05 0.01  0.005 0.001
Exponents=(-0.25 -0.5 -0.125)
JK_values=("max" "cat" "None")
Patiences=(100  )
Alphas=(-1)
betas=(0 0.5 1 )
gammas=(0)
Normalize_vals=(0)   # 1

                                    for normalize in "${Normalize_vals[@]}"; do
for Didataset in "${Direct_dataset[@]}"; do
    for k in "${K_values[@]}"; do
        for penalty in "${Weight_penalties[@]}"; do
            for dropout in "${Dropouts[@]}"; do
                            for exp in "${Exponents[@]}"; do
                              for jk in "${JK_values[@]}"; do
                                for patience in "${Patiences[@]}"; do
                                  for alpha in "${Alphas[@]}"; do
        for beta in "${betas[@]}"; do
            for gamma in "${gammas[@]}"; do
                        for lr in "${Learning_rates[@]}"; do
                for hidden in "${Hidden_dims[@]}"; do
                    for layers in "${Num_layers[@]}"; do
                                echo "Running: dataset=$Didataset, k_plus=$k, weight_penalty=$penalty, dropout=$dropout, hidden_dim=$hidden, num_layers=$layers, lr=$lr, exponent=$exp"
                                python3 run.py --self_loops \
                                    --dataset="$Didataset" \
                                    --k_plus="$k" \
                                    --weight_penalty="$penalty" \
                                    --dropout="$dropout" \
                                    --hidden_dim="$hidden" \
                                    --num_layers="$layers" \
                                    --lr="$lr" \
                                    --exponent="$exp" \
                                    --jk="$jk" \
                                    --patience="$patience" \
                                    --alpha="$alpha" --beta "$beta" --gamma "$gamma"  \
                                    --normalize="$normalize"
                                    #--use_best_hyperparams
                                echo "Finished: dataset=$Didataset, k_plus=$k, weight_penalty=$penalty, dropout=$dropout, hidden_dim=$hidden, num_layers=$layers, lr=$lr, exponent=$exp"
                              done
                              done
                              done
                              done
                            done
                        done
                    done
                done
            done
        done
    done
done
    done
done
echo "All runs completed."
