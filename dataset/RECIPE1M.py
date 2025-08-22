# -*- coding: utf-8 -*-
import argparse
import gc
import json
import logging
import os
import pickle
import time

import nltk
import numpy as np
import torch
from tqdm import *
import torch.multiprocessing as mp
from pathlib import Path
import csv

def _collect_data_mini_process(func_args):
    (data_partition_id, univl_features_file, univl_features_mean_std_file, dataset_json, lenVoc, lenVerbVoc, vocab_ing,
     verb2idx, valid_set_ids, train) = func_args

    print(f"process {os.getpid()} is spawned")
    result_file = os.path.join(dataset_json.parent.absolute(), "processed_features",
                               f"processed_features_{'train' if train else 'val'}_{data_partition_id}.pickle")
    if not os.path.exists(result_file):
        print(f"Processing partition {data_partition_id} of {'train' if train else 'val'} set "
              f"and saving it in {result_file}.")
        layer1 = json.load(open(dataset_json, 'r'))  # Load json
        univl_features_dict = pickle.load(open(univl_features_file.replace(".pickle", f"_{data_partition_id}.pickle"), 'rb'))
        # Center and Whiten
        stats = pickle.load(open(univl_features_mean_std_file, 'rb'))
        univl_features_mean = stats["mean"]
        univl_features_std = stats["std"]
        univl_features_dict = {k: ((v - univl_features_mean) / univl_features_std) for k, v in
                               univl_features_dict.items()}
        if train:
            data = [(original_data, univl_features_dict[original_data['id']]) for original_data in layer1 if
                    (original_data['id'] in univl_features_dict.keys() and original_data['id'] not in valid_set_ids)]
        else:
            data = [(original_data, univl_features_dict[original_data['id']]) for original_data in layer1 if
                    (original_data['id'] in univl_features_dict.keys() and original_data['id'] in valid_set_ids)]
        del layer1
        del univl_features_dict
        ingredients_l, sentences_l, verbs_l, verbs_tensor_l, ids_l, univl_feats_l = \
            dict(), dict(), dict(), dict(), dict(), dict()
        for i, (entry, univl_features) in enumerate(data):
            if not entry['valid'] == 'true':
                continue
            if entry['partition'] == 'test':
                continue

            c_id = entry['id']
            c_ingredients = entry['ingredient_list']
            c_sentences = entry['instructions']
            c_ingredients = np.unique(c_ingredients)
            c_ingredients = c_ingredients[c_ingredients != 30170]  # remove the '<unk>' ingredient

            ingredient_arr = np.zeros(lenVoc, dtype=np.int8)  # ingredients
            for iks in c_ingredients:
                ingredient_arr[vocab_ing(iks)] = 1

            lenSent = len(c_sentences)
            if np.sum(ingredient_arr) > 0 and lenSent > 1:
                all_sentences = []
                all_verbs = []
                c_verbs_tensor = torch.zeros(lenSent, lenVerbVoc, dtype=torch.int8)  # verbs
                for x in range(0, lenSent):
                    instr = c_sentences[x]['text']
                    tokens = nltk.word_tokenize(instr)
                    words = [word.lower() for word in tokens if word.isalpha()]
                    verbs = [word for word in words if word in verb2idx]
                    if len(words) > 0:
                        all_sentences.append(tokens)
                        all_verbs.append(verbs)
                sent_arr = [i for i in range(len(all_verbs)) for _ in range(len(all_verbs[i]))]
                verb_indices = [verb2idx[verb] for verbs_ in all_verbs for verb in verbs_]
                c_verbs_tensor[sent_arr, verb_indices] = 1

                if len(all_sentences) > 0:
                    if len(all_sentences) != univl_features.shape[0]:
                        logging.debug(
                            f"ignoring recipe {c_id} for a misaligned features and instructions\n")
                        continue
                    ingredients_l[c_id] = ingredient_arr
                    sentences_l[c_id] = all_sentences
                    verbs_l[c_id] = all_verbs
                    verbs_tensor_l[c_id] = c_verbs_tensor
                    ids_l[c_id] = c_id
                    univl_feats_l[c_id] = univl_features
        result = (ingredients_l, sentences_l, verbs_l, verbs_tensor_l, ids_l, univl_feats_l)
        with open(result_file, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        del data, result
    else:
        print(f"partition {data_partition_id} {'train' if train else 'val'} set is already processed and saved at "
              f"{result_file}")
    print(f"process {os.getpid()} is done")
    return result_file


def cluster_tags(tagged_words_dict, mapping_csv):
    with open(mapping_csv, "r") as f:
        csv_file = csv.DictReader(f)
        words_mapping = {}
        for row in csv_file:
            words_mapping[row['phrase']] = row['mapping']
    result = {}
    for key, instrs in tagged_words_dict.items():
        instrs_updated = []
        for instr in instrs:
            instr_updated = []
            for word in instr:
                if word in words_mapping:
                    if words_mapping[word] == 'none':
                        continue
                    instr_updated.extend(words_mapping[word].split(', '))
                else:
                    instr_updated.append(word)
            instrs_updated.append(instr_updated)
            result[key] = instrs_updated
    return result


class RECIPE1M:
    def __init__(self, args=None, train=True):
        self.args = args
        self.sentences, self.ingredients, self.ids, self.univl_feats = dict(), dict(), dict(), dict()
        self.verbs, self.verbs_tensor = dict(), dict()
        self.train_data = train
        self._univl_features_mean = None
        self._univl_features_std = None
        with open("dataset/recipe1M_valid_subset_ids.json", 'r') as f:
            self.val_set_ids = json.load(f)
        with open(args.vocab_ing, 'rb') as f:
            self.vocab_ing = pickle.load(f)
        with open(args.vocab_verb, 'rb') as f:
            self.vocab_verb = pickle.load(f)
        self.verb2idx = {verb: i for i, verb in enumerate(self.vocab_verb)}

    def collect_data(self):
        logging.info("Collecting Recipe1M Data")
        args = self.args
        lenVoc = len(self.vocab_ing)
        lenVerbVoc = len(self.vocab_verb)
        args.dataset_json = Path(args.dataset_json)
        function_arguments = [(data_partition_id, args.univl_features_file, args.univl_features_mean_std_file,
                               args.dataset_json, lenVoc, lenVerbVoc, self.vocab_ing, self.verb2idx, self.val_set_ids,
                               self.train_data) for data_partition_id in range(15)]
        stats = pickle.load(open(args.univl_features_mean_std_file, 'rb'))
        self._univl_features_mean = stats["mean"]
        self._univl_features_std = stats["std"]
        processed_folder = os.path.join(args.dataset_json.parent.absolute(), "processed_features")
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        logging.info("Launching Processes")
        start = time.time()

        if "world_size" in vars(args):  # TODO code crashes if a pool is created when using multiple gpus
            processed_partitions_files = []
            for func_args in function_arguments:
                processed_partitions_files.append(_collect_data_mini_process(func_args))
        else:
            with mp.Pool(mp.cpu_count(), maxtasksperchild=1) as pool:
                processed_partitions_files = pool.map(_collect_data_mini_process, function_arguments)
                pool.close()
                pool.join()
        counter_rec = 0
        for partition_file in processed_partitions_files:
            with open(partition_file, "rb") as f:
                partition = pickle.load(f)  # TODO remove verbs from partitions and recompute
            self.ingredients.update(partition[0])
            self.sentences.update(partition[1])
            self.ids.update(partition[4])
            self.univl_feats.update(partition[5])
            counter_rec += len(partition[0])
            logging.info(f"processed {counter_rec} recipes")
            del partition
        gc.collect()

        counter_rec = len(self.ids)
        logging.info(f"Tagged recipes: {counter_rec} recipes")
        end = time.time()
        logging.info(f'Loaded {counter_rec} {"train" if self.train_data else "val"} recipes in Recipe1M in ' 
                     f'{end-start:.0f} seconds')
        return
