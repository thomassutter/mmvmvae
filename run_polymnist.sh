#!/bin/bash

source ~/.bashrc
enable_modules
conda activate wsl

wandb_entity="${WANDB_LOCAL_URL}"
wandb_entity="suttetho"
project_name="mvvae"
dir_experiments="/cluster/work/vogtlab/Group/suttetho/multimodality/experiments/mvvae"
logdir="${dir_experiments}/logs"


device="cuda"  # 'cuda' if you are useing a GPU
models=("mixedprior" "unimodal" "joint") # "joint" or "mixedprior" or "drpm" "mixedpriorstdnorm"
dataset_names=("PMtranslated75")
seeds=(1 2 3 4 5)
betas=(0.25 0.5 2.0 4.0 4.0)
gammas=(0.0001)
latent_dims=(512)
drpm_prior=(False)
alpha_annealing=(True)
n_epochs=(500)

for dataset in ${dataset_names[@]}; do
for model in ${models[@]}; do
for seed in ${seeds[@]}; do
for beta in ${betas[@]};do
for ld in ${latent_dims[@]};do
for n_ep in ${n_epochs[@]};do
for dp in ${drpm_prior[@]}; do
for aa in ${alpha_annealing[@]}; do
for gamma in ${gammas[@]}; do
for n_e in ${n_epochs[@]}; do


run_name=""
wandb_logdir=${logdir}
mkdir -p ${wandb_logdir}

sbatch -c 8 -t 24:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=4096  --exclude=gpu-biomed-10,gpu-biomed-12,gpu-biomed-13 --wrap \
"python main_mv_wsl.py \
    model="${model}" \
    ++model.device=${device} \
    ++model.seed=${seed} \
    ++model.epochs=${n_e} \
    ++model.beta=${beta} \
    ++model.gamma=${gamma} \
    ++model.latent_dim=${ld} \
    ++model.drpm_prior=${dp} \
    ++model.alpha_annealing=${aa} \
    ++model.epochs=${n_ep} \
    dataset=${dataset} \
    ++log.wandb_offline=False \
    ++log.wandb_local_instance=True \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group="PolyMNIST_20240112" \
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
