# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from torch.utils.data import DistributedSampler
import numpy as np
import random

from dataset.RECIPE1M import RECIPE1M


class Recipe1MDataset(data.Dataset):
    """RECIPE1M Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, args, vocab, train=True, sort=True):
        self.recipe1m = RECIPE1M(args, train=train)
        self.recipe1m.collect_data()
        self.sort = sort
        self.ids = list(self.recipe1m.ids.keys())
        self.vocab = vocab
        if sort:
            recipe_lengths = [len(self.recipe1m.sentences[id_]) for id_ in self.ids]
            sorted_indices = sorted(range(len(recipe_lengths)), key=lambda i: recipe_lengths[i], reverse=True)
            self.idx2mappedidx = {i: j for i, j in zip(np.arange(len(self.ids)), sorted_indices)}
        else:
            self.idx2mappedidx = {i: i for i in np.arange(len(self.ids))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (ingredients and recipe)."""
        index = self.idx2mappedidx[index]
        recipe1m = self.recipe1m
        vocab = self.vocab
        ann_id = self.ids[index]
        ingredients = torch.tensor(recipe1m.ingredients[ann_id], dtype=torch.float)

        tokens = recipe1m.sentences[ann_id]
        univl_feats = recipe1m.univl_feats[ann_id]

        target_captions = []
        for x in range(0, len(tokens)):
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens[x]])
            caption.append(vocab('<end>'))
            target_captions.append(torch.Tensor(caption))

        return ingredients, target_captions, univl_feats


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (ingredients, recipes).
    Args:
        data: list of tuple (ingredients, recipes).
            - ingredients: torch tensor of shape
            - recipes: torch tensor of shape (?); variable length.
    Returns:
    """
    # Sort a data list by recipe length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    ingredients, target_captions, univl_feats = zip(*data)

    ingredients_v = torch.stack(ingredients, 0)
    univl_feats_v = torch.cat(univl_feats, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths_recipes = [len(caps) for caps in target_captions]
    target_captions = [cap for caps in target_captions for cap in caps]
    lengths_captions = [len(cap) for cap in target_captions]
    captions_v = torch.zeros(len(target_captions), max(lengths_captions)).long()

    for i, cap in enumerate(target_captions):
        end = lengths_captions[i]
        captions_v[i, :end] = cap[:end]

    return ingredients_v, lengths_recipes, captions_v, torch.LongTensor(lengths_captions), univl_feats_v


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_recipe1m_loader(args, batch_size, vocab, shuffle, num_workers, seed=0, train=True, distributed=False,
                        sort=False):
    """Returns torch.utils.data.DataLoader for custom textual datasets."""
    dataset_current = Recipe1MDataset(args, vocab, train=train, sort=sort)

    if shuffle:
        g = torch.Generator()
        g.manual_seed(0)
        worker_init_fn = seed_worker
    else:
        g = None
        worker_init_fn = None

    if distributed:
        sampler = DistributedSampler(dataset_current, shuffle=shuffle, drop_last=True, seed=seed)
        shuffle = None
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(dataset=dataset_current,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              worker_init_fn=worker_init_fn,
                                              generator=g,
                                              collate_fn=collate_fn)
    if distributed:
        return data_loader, sampler
    else:
        return data_loader
