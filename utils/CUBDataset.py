import numpy as np
import argparse
import torch
import os
import glob

import io
import json
import pickle
from collections import Counter, OrderedDict
from collections import defaultdict

import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torchvision import transforms, models, datasets

from torch.utils.data import Dataset
from torchvision.utils import save_image

from PIL import Image

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class CUB(Dataset):
    def __init__(self, dir_data, train, img_size=64, num_views=2):
        """
        dir_data: Path to the 'cub' directory.
        transform: Optional transform to be applied on a sample.
        """
        self.dir_data = dir_data
        self.idx_to_path = {}
        self.img_size = img_size
        self.num_modalities = num_views
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # Resize to a fixed size
            transforms.ToTensor(),  # Convert image to tensor
        ])

        # Load idx2name mappings
        with open(dir_data + '/idx2name.txt', 'r') as file:
            for line in file:
                idx, img_path = line.strip().split()
                self.idx_to_path[int(idx) - 1] = img_path[:-4]

        # load train or test indices
        self._load_indices()

        self.max_sequence_length = 32
        self.min_occ = 3
        os.makedirs(os.path.join(dir_data, "lang_emb"), exist_ok=True)

        self.gen_dir = os.path.join(self.dir_data, "oc_msl")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format('train' if self.train else 'test', self.max_sequence_length)
        self.vocab_file = 'cub.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format('train' if self.train else 'test', os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

        # Load labels
        idx_label = [idx//10 for i, idx in enumerate(self.indices) if i%10==0]
        label_path = os.path.join(self.dir_data, 'labels.txt')
        with open(label_path, 'r') as file:
            labels = [[int(item) for item in line.strip().split()] for line in file]
        self.labels = torch.Tensor(labels)[idx_label, 248:263] # 248:263 is has_primary_color::xxx
        # Combine primary colors
        color_combine = {
            "blue2red": [0, 2, 3, 4, 8, 9, 13],
            "brown": [1],
            "grey": [5],
            "yellow": [6, 7, 10, 14],
            "black": [11],
            "white": [12]
        }
        self.label_names = list(color_combine.keys())
        self.labels_combined = torch.zeros(len(self.labels), len(color_combine))
        for i, (color, indices) in enumerate(color_combine.items()):
            self.labels_combined[:, i] = torch.max(self.labels[:, indices], dim=1)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx not in range(self.__len__()):
            raise ValueError(f"No data found for index: {idx}")
        # row_num = self.indices[idx]
        # idx = row_num//10
        idx_img = self.indices[idx]//10

        img_path = os.path.join(self.dir_data, 'images', self.idx_to_path[idx_img]+'.jpg')
        # caption_path = os.path.join(self.dir_data, 'captions', self.idx_to_path[idx]+'.txt')
        label_path = os.path.join(self.dir_data, 'labels.txt')
        # print(img_path)

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load captions word to index
        caption = torch.Tensor(self.data_captions[str(idx)]['idx'])

        # Load labels
        label = self.labels_combined[idx//10, :]

        data_dict = {"text": caption, "img": image}
        
        return data_dict, label

    
    def _load_indices(self):
        if self.train:
            with open(self.dir_data + '/train_idx.txt') as file:
                self.indices = [int(line.strip())*10+i for line in file for i in range(10)]
        else:
            with open(self.dir_data + '/test_idx.txt') as file:
                self.indices = [int(line.strip())*10+i for line in file for i in range(10)]
        return self.indices
    
    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data_captions = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.train and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        idx_list = [idx//10 for i, idx in enumerate(self.indices) if i%10==0]
        sentences = []
        for idx in idx_list:
            caption_path = os.path.join(self.dir_data, 'captions', self.idx_to_path[idx]+'.txt')
            with open(caption_path, 'r') as file:
                captions = [line.strip() for line in file]
            sentences.extend(captions)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.train, "Vocablurary can only be created for training file."

        idx_list = [idx//10 for i, idx in enumerate(self.indices) if i%10==0]
        sentences = []
        for idx in idx_list:
            caption_path = os.path.join(self.dir_data, 'captions', self.idx_to_path[idx]+'.txt')
            with open(caption_path, 'r') as file:
                captions = [line.strip() for line in file]
            sentences.extend(captions)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()
