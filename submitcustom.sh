#!/usr/bin/env bash
set -euo pipefail
echo "script started"

#parent_dataset_name="custom_classification_realizations"
parent_dataset_name="custom_classification_testsweep"

task="custom_classification"
qg_method="accuracy"
n_epochs=1000
n_steps_per_epoch=5
n_bees=3

#learning_rates=(0.000001 0.00001 0.0001)
learning_rates=(0.001)
punish_rates=(0.001)
lambda_trains=(5)
lambda_interacts=(10)
random_seeds=(1)

features_dimensions=(10)
n_classes=(5)
n_per_class_train=(100)
n_per_class_test=(100)
sampling_gaussian_sigmas=(1.0)
class_center_radii=(5.0)

for lr in "${learning_rates[@]}"; do
    for pr in "${punish_rates[@]}"; do
        for lt in "${lambda_trains[@]}"; do
            for li in "${lambda_interacts[@]}"; do
                for rs in "${random_seeds[@]}"; do
                    for fd in "${features_dimensions[@]}"; do
                        for nc in "${n_classes[@]}"; do
                            for npctrain in "${n_per_class_train[@]}"; do
                                for npctest in "${n_per_class_train[@]}"; do
                                    for sgs in "${sampling_gaussian_sigmas[@]}"; do
                                        for ccr in "${class_center_radii[@]}"; do
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
                                                --lambda_train=$lt \
                                                --lambda_interact=$li \
                                                --random_seed=$rs \
                                                --features_dimension=$fd \
                                                --n_classes=$nc \
                                                --n_per_class_train=$npctrain \
                                                --n_per_class_test=$npctest \
                                                --class_center_radius=$ccr \
                                                --sampling_gauss_sigma=$sgs
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