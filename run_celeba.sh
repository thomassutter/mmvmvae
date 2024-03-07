#!/bin/bash

source ~/.bashrc
enable_modules
conda activate wsl

# local wandb instance
wandb_entity="suttetho"
project_name="mvvae_celeba"
dir_experiments="/cluster/work/vogtlab/Group/suttetho/multimodality/experiments/mvvae"
dir_data="/cluster/work/vogtlab/Projects/CelebA"
dir_clf="/cluster/work/vogtlab/Group/suttetho/multimodality/trained_classifiers/CelebA"
dir_alphabet="/cluster/work/vogtlab/Group/suttetho/multimodality/code/mvvae/utils"
logdir="${dir_experiments}/logs/CelebA"


device="cuda"  # 'cuda' if you are useing a GPU
models=("unimodal" "joint" "mixedprior") # "joint" or "mixedprior" or "drpm" "mixedpriorstdnorm"
dataset_names=("CelebA")
seeds=(1)
betas=(1.0)
gammas=(0.0001)
latent_dims=(128)
drpm_prior=(False)
alpha_annealing=(True)
n_epochs=(350)
learning_rates=(1e-5)
batch_sizes=(128)

for dataset in ${dataset_names[@]}; do
for model in ${models[@]}; do
for seed in ${seeds[@]}; do
for beta in ${betas[@]};do
for gamma in ${gammas[@]};do
for ld in ${latent_dims[@]};do
for n_ep in ${n_epochs[@]};do
for dp in ${drpm_prior[@]}; do
for aa in ${alpha_annealing[@]}; do
for n_e in ${n_epochs[@]}; do
for l_r in ${learning_rates[@]}; do
for bs in ${batch_sizes[@]}; do

run_name=""
wandb_logdir=${logdir}
mkdir -p ${wandb_logdir}

# sbatch -c 8 -t 24:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=8096 --nodelist=gpu-biomed-01,gpu-biomed-15,gpu-biomed-14,gpu-biomed-23 --wrap \
sbatch -c 8 -t 24:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=8096 --wrap \
"python main_mv_wsl.py \
    model="${model}" \
    ++model.device=${device} \
    ++model.seed=${seed} \
    ++model.epochs=${n_e} \
    ++model.batch_size=${bs} \
    ++model.beta=${beta} \
    ++model.gamma=${gamma} \
    ++model.latent_dim=${ld} \
    ++model.drpm_prior=${dp} \
    ++model.alpha_annealing=${aa} \
    ++model.lr=${l_r} \
    ++model.epochs=${n_ep} \
    dataset=${dataset} \
    ++dataset.dir_data=${dir_data} \
    ++dataset.dir_clf=${dir_clf} \
    ++dataset.dir_alphabet=${dir_alphabet} \
    ++log.wandb_offline=False \
    ++log.wandb_local_instance=True \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group="celeba_20240112" \
    ++log.wandb_project_name=${project_name} \
    ++log.dir_logs=${wandb_logdir}"
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

