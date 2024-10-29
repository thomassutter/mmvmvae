#!/bin/bash

source ~/.bashrc
conda activate mvvae

wandb_entity="suttetho"
project_name="mvvae_polymnist_20240116"
dir_experiments="/cluster/work/vogtlab/Group/suttetho/multimodality/experiments/mvvae"
dir_data_base="/cluster/work/vogtlab/Projects/Polymnist"
dataset_name_zip="PolyMNIST_translated_scale075.zip"
dir_clfs_base="/cluster/work/vogtlab/Group/suttetho/multimodality/trained_classifiers/PolyMNIST"
logdir="${dir_experiments}/logs/PolyMNIST"
WD=$(pwd)


device="cuda"  # 'cuda' if you are useing a GPU
models=("mixedprior") # "joint" or "mixedprior" or "drpm" "mixedpriorstdnorm"
aggregation_fs=("avg")
dataset_names=("PMtranslated75")
seeds=(1)
betas=(0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0)
betas=(1.0)
gammas=(0.0001)
latent_dims=(512)
drpm_prior=(False)
alpha_annealing=(False)
alpha_weights=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
alpha_weights=(0.0)
n_epochs=(400)
learning_rates=(5e-4)
log_freq_downstream=50
log_freq_coherence=50
log_freq_lhood=50
log_freq_plotting=50

fn_dataset_zip="${dir_data_base}/${dataset_name_zip}"

for dataset in ${dataset_names[@]}; do
for model in ${models[@]}; do
for seed in ${seeds[@]}; do
for beta in ${betas[@]};do
for ld in ${latent_dims[@]};do
for lr in ${learning_rates[@]};do
for n_ep in ${n_epochs[@]};do
for dp in ${drpm_prior[@]}; do
for aa in ${alpha_annealing[@]}; do
for a_w in ${alpha_weights[@]}; do
for gamma in ${gammas[@]}; do
for n_e in ${n_epochs[@]}; do
for agg in ${aggregation_fs[@]}; do


run_name=""
wandb_logdir=${logdir}
mkdir -p ${wandb_logdir}

sbatch -c 8 -t 20:00:00 --tmp 5G -p gpu --gres=gpu:1 --mem-per-cpu=4096 \
 --wrap="echo \$TMP; cp ${fn_dataset_zip} \$TMP; cd \$TMP; unzip -oq ${dataset_name_zip}; cd $WD; \
python main_mv_wsl.py \
    model="${model}" \
    ++model.device=${device} \
    ++model.seed=${seed} \
    ++model.epochs=${n_e} \
    ++model.beta=${beta} \
    ++model.gamma=${gamma} \
    ++model.latent_dim=${ld} \
    ++model.lr=${lr} \
    ++model.drpm_prior=${dp} \
    ++model.alpha_annealing=${aa} \
    ++model.final_alpha_value=${a_w} \
    ++model.epochs=${n_ep} \
    ++model.aggregation=${agg} \
    dataset=${dataset} \
    dataset.dir_data_base=\$TMP \
    dataset.dir_clfs_base=${dir_clfs_base} \
    ++log.wandb_offline=False \
    ++log.downstream_logging_frequency=${log_freq_downstream} \
    ++log.coherence_logging_frequency=${log_freq_coherence} \
    ++log.likelihood_logging_frequency=${log_freq_lhood} \
    ++log.img_plotting_frequency=${log_freq_plotting} \
    ++log.wandb_local_instance="True" \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group="alpha_weights" \
    ++log.wandb_project_name=${project_name} \
    ++log.dir_logs=${wandb_logdir} "
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
