import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import torch

from clfs.polymnist_clf import ClfPolyMNIST
from clfs.celeba_clf import ClfCelebA
from clfs.cub_clf import ClfCUB


def train_clf_lr_PM(encodings, labels):
    clf = LogisticRegression(max_iter=10000).fit(encodings.cpu(), labels.cpu())
    return clf


def eval_clf_lr_PM(clf, encodings, labels):
    y_pred = clf.predict(encodings.cpu())
    acc = accuracy_score(labels.cpu(), y_pred)
    return np.array(acc)


def train_clf_lr_celeba(encodings, labels):
    n_labels = labels.shape[1]
    clfs = []
    for k in range(0, n_labels):
        clf = LogisticRegression(max_iter=10000).fit(
            encodings.cpu(), labels[:, k].cpu()
        )
        clfs.append(clf)
    return clfs


def eval_clf_lr_celeba(clfs, encodings, labels):
    n_labels = labels.shape[1]
    scores = torch.zeros(n_labels)
    for k in range(0, n_labels):
        clf_k = clfs[k]
        y_pred_k = clf_k.predict(encodings.cpu())
        ap = average_precision_score(labels[:, k].cpu(), y_pred_k)
        scores[k] = ap
    return scores

def train_clf_lr_cub(encodings, labels):
    n_labels = labels.shape[1]
    clfs = []
    for k in range(0, n_labels):
        clf = LogisticRegression(max_iter=10000).fit(
            encodings.cpu(), labels[:, k].cpu()
        )
        clfs.append(clf)
    return clfs

def eval_clf_lr_cub(clfs, encodings, labels):
    n_labels = labels.shape[1]
    scores = torch.zeros(n_labels)
    for k in range(0, n_labels):
        clf_k = clfs[k]
        y_pred_k = clf_k.predict(encodings.cpu())
        auroc = roc_auc_score(labels[:, k].cpu(), y_pred_k)
        scores[k] = auroc
    return scores

def generate_samples(decoders, rep):
    imgs_gen = []
    for dec in decoders:
        img_gen = dec(rep)
        imgs_gen.append(img_gen[0])
    return imgs_gen


def conditional_generation(mvvae, dists):
    imgs_gen = []
    for idx, dist in enumerate(dists):
        mu, lv = dist
        imgs_gen_dist = []
        for m in range(len(mvvae.decoders)):
            z_out = mvvae.reparametrize(mu, lv)
            cond_gen_m = mvvae.cond_generate_samples(m, z_out)[0]
            # cond_gen_m = mvvae.decoders[m](z_out)[0]
            imgs_gen_dist.append(cond_gen_m)
        imgs_gen.append(imgs_gen_dist)
    return imgs_gen


def load_modality_clfs(cfg):
    if cfg.dataset.name.startswith("PM"):
        model = load_modality_clfs_PM(cfg)
    elif cfg.dataset.name.startswith("celeba"):
        model = load_modality_clfs_celeba(cfg)
    elif cfg.dataset.name.startswith("CUB"):
        model = load_modality_clfs_cub(cfg)
    else:
        print("dataset does not exist..exit")
        sys.exit()
    return model


def load_modality_clfs_PM(cfg):
    fp_clf = os.path.join(
        cfg.dataset.dir_clfs_base, cfg.dataset.suffix_clfs, "last.ckpt"
    )
    model = ClfPolyMNIST.load_from_checkpoint(fp_clf)
    return model


def load_modality_clfs_celeba(cfg):
    fp_clf = os.path.join(cfg.dataset.dir_clf, "last.ckpt")
    model = ClfCelebA.load_from_checkpoint(fp_clf)
    return model

def load_modality_clfs_cub(cfg):
    fp_clf = os.path.join(cfg.dataset.dir_clf, "last.ckpt")
    model = ClfCUB.load_from_checkpoint(fp_clf)
    return model


def calc_coherence_acc(cfg, clf, imgs, labels):
    out_clf = clf(cfg, [imgs, labels])
    preds = out_clf[0]
    return preds


def from_preds_to_acc(preds, labels, modality_names):
    n_views = len(modality_names)
    accs = torch.zeros((n_views, n_views, 1))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            acc_m_mtilde = accuracy_score(
                labels.cpu(),
                np.argmax(preds_m_mtilde.cpu().numpy(), axis=1).astype(int),
            )
            accs[m, m_tilde, 0] = acc_m_mtilde
    return accs


def from_preds_to_ap(preds, labels, modality_names):
    n_views = len(modality_names)
    n_labels = labels.shape[1]
    aps = torch.zeros((n_views, n_views, n_labels))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            for k in range(0, n_labels):
                ap_m_mtilde_k = average_precision_score(
                    labels[:, k].cpu(), preds_m_mtilde[:, k].detach().cpu().numpy()
                )
                aps[m, m_tilde, k] = ap_m_mtilde_k
    return aps


def calc_coherence_ap(cfg, clf, mods, labels):
    out_clf = clf(cfg, [mods, labels])
    preds = out_clf[0]
    return preds
