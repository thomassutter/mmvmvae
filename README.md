# MMVM VAE

This is the official code release for the Neurips 2024 paper [Unity by Diversity: Improved Representation Learning for Multimodal VAEs](https://arxiv.org/abs/2403.05300).

Comments and questions are always welcome. Just reach out to us!

## MMVM Prior
In this paper, we introduce the *MMVM VAE*, a novel multimodal VAE formulation using a shoft-sharing of information between modalities.

The proposed method is based on the MMVM prior $h(\mathbf{z} | \mathbf{X})$ that acts on the unimodal posterior approximations $q(\mathbf{z}_m | \mathbf{x}_m)$:

![MMVM VAE](arch_mmvamp_vaes_cropped.png)

## Installation

To be able to run the experiments and reproduce the results shown in the paper, you need to install the ```mvvae``` conda environments using

```
conda env create -f environment.yml
```

## Data
The data for the PolyMNIST and the CelebA dataset can be downloaded using the following commands
```
curl -L -o tmp.zip https://drive.google.com/drive/folders/1lr-laYwjDq3AzalaIe9jN4shpt1wBsYM?usp=sharing
unzip tmp.zip
unzip celeba_data.zip -d data/
unzip PolyMNIST.zip -d data/
```
The data for the Rats dataset can be downloaded through the link: https://datadryad.org/stash/dataset/doi:10.7280/D14X30

## Experiments

This repository is for all the benchmark dataset experiments: PolyMNIST, bimodal CelebA, and CUB.

For the Hippocampal neural activity experiment, please see [Hippocampal Neural Activity repository](https://github.com/yangmeng96/mmvmvae-hippocampal).

For the MIMIC-CXR experiments, please seee [MIMIC-CXR repository](https://github.com/agostini335/mmvmvae-mimic)

# Benchmark Experiments

To run the PolyMNIST experiment, you can use the following command from the root dir of the repository after having activated the conda environment

```
python main_mv_wsl.py dataset="PMtranslated75" model="mixedprior"
```

The CelebA experiment can be run by using the following command

```
python main_mv_wsl.py dataset="CelebA" model="mixedprior"
```

The CUB experiment can be run by using the following command

```
python main_mv_wsl.py dataset="cub" model="mixedprior"
```

To run one of the baseline multimodal methods, just change the method to ```joint``` or ```independent```.

In the ```ModelConfig.py```, you can further choose between different aggregation functions when seeting

By setting ```model.aggregation```, you can further choose between different aggregation functions, e.g.
```model.aggregation=moe``` uses the mixture of experts joint distribution.


In addition, to compute the conditional generation coherence, you need to train non-linear classifiers

```
python main_train_clf_PM.py
```
or
```
python main_train_clf_celeba.py
```
or
```
python main_train_clf_cub.py
```


We use [WandB](https://wandb.ai/) and [Hydra](https://hydra.cc/) for logging and configuring our experiments.
So,

- make sure to have a WandB account
- you can easily set any experiment parameters either over the command line or using ```config.yaml``` files




## Citation
If you use our model in your work, please cite us using the following citation

```
@inproceedings{sutter2024,
  title={Unity by Diversity: Improved Representation Learning in Multimodal VAEs},
  author={Sutter, Thomas M and Meng, Yang and Agostini, Andrea and Chopard, Daphné and Fortin, Norbert and Vogt, Julia E. and Shahbaba, Babak and Mandt, Stephan},
  year = {2024},
  booktitle = {arxiv},
}
```


## Questions
For any questions or requests, please reach out to:

[Thomas Sutter](https://thomassutter.github.io/) [(thomas.sutter@inf.ethz.ch)](mailto:thomas.sutter@inf.ethz.ch)


[Yang Meng]() [(mengy13@uci.edu)](mailto:mengy13@uci.edu)
