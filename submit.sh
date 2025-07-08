#!/usr/bin/env bash
set -euo pipefail
echo "script started"

#learning_rates=( 0.0000005 0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 )
learning_rates=( 0.0001)
#learning_rates=( 0.000001 0.00001 0.00005 0.0001)
#do not use too high punish rates, that will break the system
punish_rates=( 0.0000005 )
lambda_trains=( 2 10 )
lambda_interacts=( 5 )
#acc_sigmas=( 0.0001 0.001 0.01 0.05 )
acc_sigmas=( 0.001 )
which_peaks=( 1 )
random_seeds=( 2 )

task="custom_classification"
qg_method="accuracy"
n_epochs=10000
#n_epochs=5
n_steps_per_epoch=1
n_bees=1
n_peaks=5

parent_dataset_name="custom_classification_testsweep"

for lr in "${learning_rates[@]}"; do
    for pr in "${punish_rates[@]}"; do
        for lt in "${lambda_trains[@]}"; do
            for li in "${lambda_interacts[@]}"; do
                for as in "${acc_sigmas[@]}"; do
                    for wp in "${which_peaks[@]}"; do
                        for rs in "${random_seeds[@]}"; do
                            #echo "Submitting: lr=$lr, pr=$pr, lt=$lt, li=$li, as=$as, seed=$seed"
                            sbatch ./run_sbatch.slurm \
                                --task=$task \
                                --queen_gene_method=$qg_method \
                                --parent_dataset_name=$parent_dataset_name \
                                --n_bees=$n_bees \
                                --n_epochs=$n_epochs \
                                --n_steps_per_epoch=$n_steps_per_epoch \
                                --learning_rate=$lr \
                                --punish_rate=$pr \
                                --accuracy_sigma=$as \
                                --lambda_train=$lt \
                                --lambda_interact=$li \
                                --random_seed=$rs \
                                --regression_n_peaks=$n_peaks \
                                --regression_which_peak=$wp
                        done
                    done
                done
            done
        done
    done
done

: << 'END'
learning_rates=(0.00025 0.0003 0.00035 0.0004)
punish_rates=(0.00000001 0.00000005 0.0000001 0.0000002 0.0000003)
lambda_trains=(0.003 0.004 0.006 0.008 0.01 0.1)
lambda_interacts=(5 10)
acc_sigmas=(0.5)
END
