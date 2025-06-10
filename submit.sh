#!/usr/bin/env bash
set -euo pipefail
echo "script started"

learning_rates=( 0.000000001 )
#do not use too high punish rates, that will break the system
punish_rates=( 0.000001 )
#punish_rates=( 0.0000000000001 0.000000000001 0.00000000001 0.0000000001 0.000000001 0.00000001 0.0000001 0.000001)
#lambda_trains=( 0.1 0.2 0.3 0.4 0.5 )
lambda_trains=( 100000 )
lambda_interacts=( 5 )
acc_sigmas=( 0.5 )
#acc_sigmas=( 0.5 0.4 0.3 0.2 0.1 )

n_realizations=1

task="regression"
qg_method="accuracy"
n_epochs=10000
n_steps_per_epoch=1
n_bees=3

#parent_dataset_name="params_sweep_task_${task}__qg_method_${qg_method}__n_bees_${n_bees}__n_epochs_${n_epochs}__n_steps_per_epoch_${n_steps_per_epoch}"
parent_dataset_name="resetting"

for lr in "${learning_rates[@]}"; do
    for pr in "${punish_rates[@]}"; do
        for lt in "${lambda_trains[@]}"; do
            for li in "${lambda_interacts[@]}"; do
                for as in "${acc_sigmas[@]}"; do
                    for ((i = 1; i <=n_realizations; i++)); do
                        seed=$i
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
                            --random_seed=$seed
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
