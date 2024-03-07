import os
import json
import textwrap

import numpy as np
import torch
from torchvision import transforms

from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq, alphabet, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples


def text_to_pil_celeba(text_sample, imgsize, font, w=256, h=256):
    blank_img = torch.ones([3, w, h])
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    lines = textwrap.wrap(text_sample, width=16)
    y_text = h
    num_lines = len(lines)
    for idx_l, line in enumerate(lines):
        bbox = font.getbbox(line)
        height = bbox[1] - bbox[3]
        draw.text(
            (0, (h / 2) - (num_lines / 2 - idx_l) * height), line, (0, 0, 0), font=font
        )
        y_text += height
    text_pil = transforms.ToTensor()(
        pil_img.resize((imgsize[1], imgsize[2]), Image.LANCZOS)
    )
    return text_pil


def text_to_pil(text_sample, imgsize, alphabet, font, w=128, h=128, linewidth=8):
    blank_img = torch.ones([imgsize[0], w, h])
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    lines = textwrap.wrap("".join(text_sample), width=linewidth)
    y_text = h
    num_lines = len(lines)
    for idx_l, line in enumerate(lines):
        bbox = font.getbbox(line)
        height = bbox[1] - bbox[3]
        draw.text(
            (0, (h / 2) - (num_lines / 2 - idx_l) * height), line, (0, 0, 0), font=font
        )
        y_text += height
    if imgsize[0] == 3:
        text_pil = transforms.ToTensor()(
            pil_img.resize((imgsize[1], imgsize[2]), Image.LANCZOS)
        )
    else:
        text_pil = transforms.ToTensor()(
            pil_img.resize((imgsize[1], imgsize[2]), Image.LANCZOS).convert("L")
        )
    return text_pil


def create_txt_image(cfg, text_mod):
    alphabet_path = os.path.join(cfg.dataset.dir_alphabet, "alphabet.json")
    with open(alphabet_path) as alphabet_file:
        alphabet = str("".join(json.load(alphabet_file)))
    imgsize = torch.Size((3, 64, 64))
    font = ImageFont.truetype(
        os.path.join(cfg.dataset.dir_alphabet, "FreeSerif.ttf"), 32
    )
    imgs = torch.zeros(text_mod.shape[0], 3, 64, 64)
    text_samples = []
    for idx in range(text_mod.shape[0]):
        text_sample = tensor_to_text(alphabet, text_mod[idx].unsqueeze(0))[0]
        text_sample = "".join(text_sample).translate({ord("*"): None})
        text_samples.append(text_sample)
        img = text_to_pil(
            text_sample,
            imgsize,
            alphabet,
            font,
            w=256,
            h=256,
            linewidth=16,
        )
        imgs[idx] = img
    return imgs, text_samples
