import os
import json
from nltk.tokenize import word_tokenize
from typing import List

from collections import Counter, OrderedDict
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
import PIL.Image as Image

import torch

from utils import text as text


class CelebADataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, cfg, alphabet, partition=0, transform=None):
        filename_text = os.path.join(
            cfg.dataset.dir_data,
            "list_attr_text_"
            + str(cfg.dataset.len_sequence).zfill(3)
            + "_"
            + str(cfg.dataset.random_text_ordering)
            + "_"
            + str(cfg.dataset.random_text_startindex)
            + "_celeba.csv",
        )
        filename_partition = os.path.join(
            cfg.dataset.dir_data, "list_eval_partition.csv"
        )
        filename_attributes = os.path.join(cfg.dataset.dir_data, "list_attr_celeba.csv")

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition)
        df_attributes = pd.read_csv(filename_attributes)

        self.cfg = cfg
        self.img_dir = os.path.join(cfg.dataset.dir_data, "img_align_celeba")
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes
        self.partition_path = filename_partition

        self.alphabet = alphabet
        self.img_names = df_text.loc[df_partition["partition"] == partition][
            "image_id"
        ].values
        self.attributes = df_attributes.loc[df_partition["partition"] == partition]
        self.labels = df_attributes.loc[df_partition["partition"] == partition].values
        self.label_names = list(df_attributes.columns)[1:]
        self.y = df_text.loc[df_partition["partition"] == partition]["text"].values
        self.transform = transform

    def __getitem__(self, index):
        with Image.open(os.path.join(self.img_dir, self.img_names[index])) as img:
            if self.transform is not None:
                img = self.transform(img)
            text_str = text.one_hot_encode(
                self.cfg.dataset.len_sequence, self.alphabet, self.y[index]
            )
            label = torch.from_numpy((self.labels[index, 1:] > 0).astype(int)).float()
            sample = {"img": img, "text": text_str}
            return sample, label

    def __len__(self):
        return self.y.shape[0]

    def get_text_str(self, index):
        return self.y[index]


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first encountered.
    """

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_tensor(data):
    return torch.Tensor(data)


class CelebASentences(Dataset):
    """
    Modified version of https://github.com/iffsid/mmvae/blob/public/src/datasets.py
    Word encoding for mimic report findings
    """

    def __init__(
        self,
        max_squence_len: int,
        data_dir: str,
        findings: pd.DataFrame,
        split: str,
        transform=False,
        min_occ: int = 3,
    ):
        """split: 'train', 'val' or 'test'"""

        super().__init__()
        self.split = split
        self.data_dir = data_dir
        # self.args = args
        self.max_sequence_length = max_squence_len
        self.min_occ = min_occ
        self.transform = to_tensor if transform else None
        self.findings = findings
        self.gen_dir = os.path.join(
            self.data_dir, "oc:{}_msl:{}".format(self.min_occ, self.max_sequence_length)
        )

        self.raw_data_path = os.path.join(data_dir, split + "_findings.csv")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = "mimic.{}.s{}".format(split, self.max_sequence_length)
        self.vocab_file = "mimic.vocab"

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print(
                "Data file not found for {} split at {}. Creating new... (this may take a while)".format(
                    split.upper(), os.path.join(self.gen_dir, self.data_file)
                )
            )
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns tensors/list of length len_sentence
        """
        sent = self.data[str(idx)]["idx"]
        if self.transform is not None:
            sent = self.transform(sent)
        return sent

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    @property
    def unk_idx(self):
        return self.w2i["<unk>"]

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), "rb") as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab["w2i"], vocab["i2w"]

    def _create_data(self):
        if self.split == "train" and not os.path.exists(
            os.path.join(self.gen_dir, self.vocab_file)
        ):
            self._create_vocab()
        else:
            self._load_vocab()

        sentences = self._tokenize_raw_data()

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line.lower())
            # words = word_tokenize(line)
            tok = words[: self.max_sequence_length - 1]
            tok = tok + ["<eos>"]
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(["<pad>"] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i["<exc>"]) for w in tok]

            id = len(data)
            data[id]["tok"] = tok
            data[id]["idx"] = idx
            data[id]["length"] = length

        print(
            "{} out of {} sentences are truncated with max sentence length {}.".format(
                len(sentences) - pad_count, len(sentences), self.max_sequence_length
            )
        )
        with io.open(os.path.join(self.gen_dir, self.data_file), "wb") as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode("utf8", "replace"))

        self._load_data(vocab=False)

    def _tokenize_raw_data(self) -> List:
        """
        Creates a list of all the findings
        """
        report_findings = self.findings
        return [sentence for sentence in report_findings]

    def _create_vocab(self):
        assert (
            self.split == "train"
        ), "Vocabulary can only be created for training file."

        sentences = self._tokenize_raw_data()

        occ_register = OrderedCounter()
        w2i = {}
        i2w = {}

        special_tokens = ["<exc>", "<pad>", "<eos>"]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            # words = word_tokenize(line)
            words = word_tokenize(line.lower())
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print(
            "Vocabulary of {} keys created, {} words are excluded (occurrence <= {}).".format(
                len(w2i), len(unq_words), self.min_occ
            )
        )

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), "wb") as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode("utf8", "replace"))

        with open(os.path.join(self.gen_dir, "mimic.unique"), "wb") as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, "mimic.all"), "wb") as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()
