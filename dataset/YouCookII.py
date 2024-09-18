# -*- coding: utf-8 -*-
import json
import logging
import pickle

import nltk
import numpy as np
import torch
from tqdm import *


class YouCookII:
    def __init__(self, args=None, train=True, load_resnet_feats=False, test_split=0, use_unseen_split=True):
        logging.info("Using YouCookII Modified Splits")
        self.sentences, self.ingredients, self.ids, self.video_splits = dict(), dict(), dict(), dict()
        self.resnet_feats, self.textual_univl_feats, self.visual_univl_feats = dict(), dict(), dict()
        self.train = train
        self.test_split = test_split
        assert 0 <= test_split <= 3
        self.use_unseen_split = use_unseen_split

        annotations = json.load(open(args.dataset_json, 'r'))  # Load json
        self.splits = json.load(open(args.json_splits, 'r'))  # Load video splits

        if load_resnet_feats:
            assert hasattr(args, 'resnet_frames_file')
            resnet_frames_dict = pickle.load(open(args.resnet_frames_file, 'rb'))
        else:
            resnet_frames_dict = None

        visual_univl_features_dict = pickle.load(open(args.visual_univl_features_file, 'rb'))
        textual_univl_features_dict = pickle.load(open(args.textual_univl_features_file, 'rb'))
        # Center and Whiten
        textual_stats = pickle.load(open(args.textual_univl_features_mean_std_file, 'rb'))
        visual_stats = pickle.load(open(args.visual_univl_features_mean_std_file, 'rb'))
        self._textual_univl_features_mean, self._textual_univl_features_std = textual_stats["mean"], textual_stats[
            "std"]
        self._visual_univl_features_mean, self._visual_univl_features_std = visual_stats["mean"], visual_stats["std"]

        textual_univl_features_dict = {k: ((v - self._textual_univl_features_mean) / self._textual_univl_features_std)
                                       for
                                       k, v in textual_univl_features_dict.items()}
        visual_univl_features_dict = {k: ((v - self._visual_univl_features_mean) / self._visual_univl_features_std) for
                                      k, v in visual_univl_features_dict.items()}

        with open(args.vocab_ing, 'rb') as f:
            vocab_ing = pickle.load(f)
        lenVoc = len(vocab_ing)

        counter_rec = 1
        for i, (c_id, entry) in tqdm(enumerate(annotations.items())):
            if load_resnet_feats:
                if c_id not in resnet_frames_dict:
                    print(f"ignoring recipe {c_id} due to missing visual features\n")
                    continue
            c_ingredients = entry['ingredient_list']
            c_sentences = entry['annotations']
            c_ingredients = np.unique(c_ingredients)
            c_ingredients = c_ingredients[c_ingredients != 30170]  # remove the '<unk>' ingredient

            ingredient_arr = np.zeros(lenVoc)  # ingredients
            for iks in c_ingredients:
                ingredient_arr[vocab_ing(iks)] = 1

            lenSent = len(c_sentences)
            if np.sum(ingredient_arr) > 0 and lenSent > 1:
                video_split = None
                for j, split in self.splits.items():
                    if video_split is None:
                        j = int(j)
                        for k, sub_split in enumerate(split):
                            if c_id in sub_split:
                                video_split = (j, k)
                                break
                    else:
                        break
                if video_split is None:
                    print(f"ignoring recipe {c_id} due to missing visual features\n")
                    continue

                if self.train and video_split[0] == self.test_split:
                    if self.use_unseen_split or video_split[1] == 0:
                        continue
                elif not self.train:
                    if video_split[0] != self.test_split or video_split[1] != 0:
                        continue

                all_sentences = []
                all_segments = []
                for x in range(0, lenSent):
                    instr = c_sentences[x]['sentence']
                    tokens = nltk.word_tokenize(instr)
                    words = [word.lower() for word in tokens if word.isalpha()]
                    if len(words) > 0:
                        all_sentences.append(tokens)
                        all_segments.append(c_sentences[x]['segment'])


                if len(all_sentences) > 0:
                    visual_univl_features = visual_univl_features_dict[c_id]
                    textual_univl_features = textual_univl_features_dict[c_id]
                    if len(all_sentences) != visual_univl_features.shape[0]:
                        print(
                            f"ignoring recipe {c_id} for a misaligned features and instructions\n")
                        continue
                    self.ingredients[str(counter_rec)] = ingredient_arr
                    self.sentences[str(counter_rec)] = all_sentences
                    self.ids[str(counter_rec)] = c_id
                    self.video_splits[str(counter_rec)] = video_split
                    if load_resnet_feats:
                        self.resnet_feats[str(counter_rec)] = resnet_frames_dict[c_id]
                    self.textual_univl_feats[str(counter_rec)] = textual_univl_features
                    self.visual_univl_feats[str(counter_rec)] = visual_univl_features
                    counter_rec = counter_rec + 1

        print(f'total number of loaded videos in YouCookII: ' + str(counter_rec))
