# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import numpy as np
import random

from dataset.YouCookII import YouCookII
from dataset.YouCookII_orig_split import YouCookII_Orig


class YouCookIIDataset(data.Dataset):
    """YouCookII Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, args, vocab, train=True, load_resnet_feats=False, sort=False, test_split=0,
                 split_type="unseen_split"):
        """
        Args:
            test_split: the split from which to use the test set (from 0 t 3), use -1 for sampling videos randomly from
            all the splits
            use_unseen_split: use a test split with no overlapping recipes with the training
            use_original_split: use the original training/validation splits from YouCookII dataset
            split_type: use the original train/validation splits ('original_split'), the seen splits ('seen_split'), or the unseen
            splits ('unseen_split'). Check the paper for more info
        """
        assert split_type in ["unseen_split", "seen_split", "original_split"]
        assert 0 <= test_split <= 3
        self.test_split = test_split
        self.load_resnet_feats = load_resnet_feats
        if split_type in ["unseen_split", "seen_split"]:
            use_unseen_split = split_type == "unseen_split"
            self.youcookii = YouCookII(args, train=train, load_resnet_feats=load_resnet_feats, test_split=test_split,
                                       use_unseen_split=use_unseen_split)
        elif split_type == "original_split":
            self.youcookii = YouCookII_Orig(args, train=train, load_resnet_feats=load_resnet_feats)
        else:
            raise ValueError(f"unknown split_type {split_type}")
        self.ids = list(self.youcookii.ids.keys())
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
        youcookii = self.youcookii
        vocab = self.vocab
        ann_id = self.ids[index]
        ingredients = torch.tensor(youcookii.ingredients[ann_id], dtype=torch.float)

        tokens = youcookii.sentences[ann_id]
        textual_univl_feats = youcookii.textual_univl_feats[ann_id]
        visual_univl_feats = youcookii.visual_univl_feats[ann_id]

        target_captions = []
        for x in range(0, len(tokens)):
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens[x]])
            caption.append(vocab('<end>'))
            target_captions.append(torch.Tensor(caption))

        if self.load_resnet_feats:
            resnet_feats = youcookii.resnet_feats[ann_id]
        else:
            resnet_feats = None
        return ingredients, target_captions, textual_univl_feats, visual_univl_feats, resnet_feats


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
    ingredients, target_captions, textual_univl_feats, visual_univl_feats, resnet_feats = zip(*data)

    ingredients_v = torch.stack(ingredients, 0)
    textual_univl_feats_v = torch.cat(textual_univl_feats, 0)
    visual_univl_feats_v = torch.cat(visual_univl_feats, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths_recipes = [len(caps) for caps in target_captions]
    target_captions = [cap for caps in target_captions for cap in caps]
    lengths_captions = [len(cap) for cap in target_captions]
    captions_v = torch.zeros(len(target_captions), max(lengths_captions)).long()
    for i, cap in enumerate(target_captions):
        end = lengths_captions[i]
        captions_v[i, :end] = cap[:end]

    if resnet_feats is not None and resnet_feats[0] is not None:
        skip = 2
        d = resnet_feats[0][0].shape[-1]
        resnet_feats = [feat for vid_feats in resnet_feats for feat in vid_feats]
        seg_l = [max(int(len(vid_feats)/skip), 1) for vid_feats in resnet_feats]
        max_seg_l = max(seg_l)
        resnet_feats_v = torch.zeros(len(seg_l), max_seg_l, d)
        for i, feats in enumerate(resnet_feats):
            if feats.shape[0] == 0:
                feats = torch.zeros(1, d)
            resnet_feats_v[i, :seg_l[i]] = feats[:seg_l[i]*skip:skip]
        return ingredients_v, lengths_recipes, captions_v, torch.LongTensor(lengths_captions),\
               (resnet_feats_v, torch.LongTensor(seg_l)), (textual_univl_feats_v, visual_univl_feats_v)
    else:
        return ingredients_v, lengths_recipes, captions_v, torch.LongTensor(lengths_captions), \
               (textual_univl_feats_v, visual_univl_feats_v)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_youcookii_loader(args, batch_size, vocab, shuffle, num_workers, train=True, load_resnet_feats=False, sort=False,
                         test_split=0, split_type='unseen_split'):
    """Returns torch.utils.data.DataLoader for custom textual datasets."""
    assert split_type in ["unseen_split", "seen_split", "original_split"]
    if shuffle:
        g = torch.Generator()
        g.manual_seed(0)
        worker_init_fn = seed_worker
    else:
        g = None
        worker_init_fn = None

    dataset_current = YouCookIIDataset(args, vocab, train=train, load_resnet_feats=load_resnet_feats, sort=sort,
                                       test_split=test_split, split_type=split_type)

    data_loader = torch.utils.data.DataLoader(dataset=dataset_current,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              worker_init_fn=worker_init_fn,
                                              generator=g,
                                              collate_fn=collate_fn)
    return data_loader
