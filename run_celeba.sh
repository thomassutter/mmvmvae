#!/bin/bash

source ~/.bashrc
conda activate mvvae

# local wandb instance
wandb_entity="suttetho"
project_name="mvvae_celeba_20240116"
dir_experiments="/cluster/work/vogtlab/Group/suttetho/multimodality/experiments/mvvae"
dir_data="/cluster/work/vogtlab/Projects/CelebA"
dir_clf="/cluster/work/vogtlab/Group/suttetho/multimodality/trained_classifiers/CelebA"
dir_alphabet="/cluster/work/vogtlab/Group/suttetho/multimodality/code/mvvae/utils"
logdir="${dir_experiments}/logs/CelebA"


device="cuda"  # 'cuda' if you are useing a GPU
models=("unimodal" "joint" "mixedprior") # "joint" or "mixedprior" or "drpm" "mixedpriorstdnorm"
dataset_names=("CelebA")
seeds=(1)
betas=(0.25 0.5 1.0 2.0 4.0)
betas=(1.0)
gammas=(0.0001)
latent_dims=(128)
drpm_prior=(False)
alpha_annealing=(True)
alpha_annealing_n_steps=(150000)
n_epochs=(400)
learning_rates=(5e-4)
batch_sizes=(256)
log_freq_downstream=50
log_freq_coherence=50
log_freq_lhood=500
log_freq_plotting=50

for dataset in ${dataset_names[@]}; do
for model in ${models[@]}; do
for seed in ${seeds[@]}; do
for beta in ${betas[@]};do
for gamma in ${gammas[@]};do
for ld in ${latent_dims[@]};do
for n_ep in ${n_epochs[@]};do
for dp in ${drpm_prior[@]}; do
for aa in ${alpha_annealing[@]}; do
for aa_n_steps in ${alpha_annealing_n_steps[@]}; do
for n_e in ${n_epochs[@]}; do
for l_r in ${learning_rates[@]}; do
for bs in ${batch_sizes[@]}; do

run_name=""
wandb_logdir=${logdir}
mkdir -p ${wandb_logdir}

# sbatch -c 8 -t 24:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=8096 --nodelist=gpu-biomed-01,gpu-biomed-15,gpu-biomed-14,gpu-biomed-23 --wrap \
sbatch -c 8 -t 36:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=8096 --wrap \
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
    ++model.alpha_annealing_steps=${aa_n_steps} \
    ++model.lr=${l_r} \
    ++model.epochs=${n_ep} \
    dataset=${dataset} \
    ++dataset.dir_data=${dir_data} \
    ++dataset.dir_clf=${dir_clf} \
    ++dataset.dir_alphabet=${dir_alphabet} \
    ++log.downstream_logging_frequency=${log_freq_downstream} \
    ++log.coherence_logging_frequency=${log_freq_coherence} \
    ++log.likelihood_logging_frequency=${log_freq_lhood} \
    ++log.img_plotting_frequency=${log_freq_plotting} \
    ++log.wandb_offline="False" \
    ++log.wandb_local_instance="True" \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_run_name=${run_name} \
    ++log.wandb_group="20240216" \
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
done
