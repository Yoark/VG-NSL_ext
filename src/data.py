import nltk
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data
from transformers import BertTokenizer, BasicTokenizer
from utils import whitespace_tokenize, get_whole_word

class PrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [self.vocab(token) 
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length

# TODO add bert tokenizer here
class BertDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab,
                 load_img=True, img_dim=2048):
        self.vocab = vocab
        vocab_pre = pickle.load(open(os.path.join(data_path, 'vocab.pkl'), 'rb'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #, additional_special_tokens=list(vocab_pre.word2idx.keys()))

        self.basic_tokenizer = BasicTokenizer()
        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                # TODO use bert tokenizer?????
                self.captions.append('[CLS] ' + line.strip().lower() + ' [SEP]')
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        # caption
        # ? is it correct ???
        # TODO log whole word -> DONE
        whole_caption = whitespace_tokenize(self.captions[index])
        tokenizerd_caption = self.tokenizer.tokenize(self.captions[index])
        ind_in = get_whole_word(whole_caption, tokenizerd_caption)
        ind_tensor = torch.zeros(len(tokenizerd_caption), len(whole_caption))
        for i in range(len(whole_caption)):
            #import ipdb
            #ipdb.set_trace()
            ind_tensor[ind_in[i], i] = 1
        ind_tensor = ind_tensor.long()
        caption = self.tokenizer.convert_tokens_to_ids(tokenizerd_caption)
        #return a tensor 
        caption = torch.tensor(caption)
        return image, caption, index, img_id, ind_tensor

    def __len__(self):
        return self.length

def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    # align_tensor = len(tokenized_caption) * len(whole_caption)
    images, captions, ids, img_ids, = zipped_data
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), len(captions[0])).long()
    lengths = [len(cap) for cap in captions]
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ids

def collate_fn_bert(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: x[4].shape[1], reverse=True)
    zipped_data = list(zip(*data))
    whole_length_max = zipped_data[4][0].shape[1]
    # align_tensor = len(tokenized_caption) * len(whole_caption)
    images, captions, ids, img_ids, align_tensors = zipped_data
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    length_max = max(lengths)
    lengths_whole = [align.shape[1] for align in align_tensors]
    targets = torch.zeros(len(captions), length_max).long()
    targets_aligns = torch.zeros(len(captions), length_max, whole_length_max).to(torch.float32)
    for i, tup in enumerate(zip(captions, align_tensors)):
        cap, align_tensor = tup
        end = len(cap)
        tokenized_l = align_tensor.shape[0]
        whole_l = align_tensor.shape[1]
        #import ipdb; ipdb.set_trace()
        targets[i, :end] = cap[:end]
        targets_aligns[i, :tokenized_l, :whole_l]
    return images, targets, lengths, ids, targets_aligns, lengths_whole


def get_precomp_loader(data_path, data_split, vocab, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True, 
                       img_dim=2048):
    # ! Changed to bert
    dset = PrecompDataset(data_path, data_split, vocab, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return data_loader

def get_precomp_loader_bert(data_path, data_split, vocab, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True, 
                       img_dim=2048):
    # ! Changed to bert
    dset = BertDataset(data_path, data_split, vocab, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, 
        collate_fn=collate_fn_bert
    )
    return data_loader


def get_train_loaders(opt, data_path, vocab, batch_size, workers):
    if opt.init_embeddings_type == 'bert':
        train_loader = get_precomp_loader_bert(
           data_path, 'train', vocab, batch_size, True, workers
        )
        val_loader = get_precomp_loader_bert(
            data_path, 'dev', vocab, batch_size, False, workers
        )
    else:
        train_loader = get_precomp_loader(
            data_path, 'train', vocab, batch_size, True, workers
        )
        val_loader = get_precomp_loader(
            data_path, 'dev', vocab, batch_size, False, workers
        )
    return train_loader, val_loader


def get_eval_loader(opt, data_path, split_name, vocab, batch_size, workers, 
                    load_img=False, img_dim=2048):
    if opt.init_embeddings_type == "bert":
        eval_loader = get_precomp_loader_bert(
        data_path, split_name, vocab, batch_size, False, workers,
        load_img=load_img, img_dim=img_dim
        )
    else:
        eval_loader = get_precomp_loader(
        data_path, split_name, vocab, batch_size, False, workers,
        load_img=load_img, img_dim=img_dim
        )
    return eval_loader
