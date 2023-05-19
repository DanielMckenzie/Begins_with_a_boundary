#!/bin/bash

# Define the beta values, datasets, repetitions, and algorithms
beta_values="0.01 0.1 0.3 0.5"
datasets="MNIST CIFAR10"
reps="1 2 3"
algorithms="CVaR_Modified_SGD CVaR_SGD"

# Run the command for each repetition, dataset, beta value, and algorithm
for rep in $reps
do
    for dataset in $datasets
    do
        for algorithm in $algorithms
        do
            for beta in $beta_values
            do
                # Format the beta value by removing the decimal point
                beta_formatted=$(printf "%.2f" $beta | tr -d '.')

                if [ "$beta_formatted" = "" ]; then
                    beta_formatted="0"
                fi
                
                # Set the step size based on the beta value
                if (( $(echo "$beta < 0.1" | bc -l) )); then
                    t_step_size=0.1
                else
                    t_step_size=1.0
                fi

                # Run the command and save the output to a file
                output_file="${dataset}_${algorithm}_mod_$beta_formatted_rep_$rep"

                # Uncomment the following line to run the actual command
                CUDA_VISIBLE_DEVICES=0 python3 -m advbench.scripts.train --dataset $dataset --algorithm $algorithm --output_dir rep_$rep/$output_file --cvar_sgd_beta $beta --cvar_sgd_t_step_size $t_step_size  --evaluators Clean PGD Augmented
            done
        done
    done
done
