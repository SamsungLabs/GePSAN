import logging
import pickle
from typing import Optional, Iterable
from abc import ABC, abstractmethod

import nltk
import torch

from Vocabulary import Vocabulary
import torch.multiprocessing as mp
from functools import partial
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor


class BaseEvaluator(ABC):
    def __init__(self, vocab: Vocabulary, vocab_ingredients_path: Optional[str] = None,
                 vocab_verbs_path: Optional[str] = None):
        super(BaseEvaluator, self).__init__()
        self.vocab = vocab
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.tokenizer = PTBTokenizer()
        self.bleu_scorer = Bleu(4)
        self.meteor_scorer = Meteor()

        self.vocab_verb = None
        self.vocab_ings = None
        if vocab_verbs_path is not None:
            with open(vocab_verbs_path, 'rb') as handle:
                self.vocab_verb = pickle.load(handle)

        if vocab_ingredients_path is not None:
            with open(vocab_ingredients_path, 'rb') as handle:
                vocab_ings = pickle.load(handle)
                self.vocab_ings = [self.vocab.idx2word[vocab_ings.idx2word[i]] for i in range(len(vocab_ings))]

    def _indices2sents(self, indices):
        return ' '.join(self.vocab.indices2words(indices, self.start_token, self.end_token))

    def _calculate_scores(self, gts_tokenized, res_tokenized, prefix="", suffix=""):
        scores_dict = {}
        score, scores = self.bleu_scorer.compute_score(gts_tokenized, res_tokenized)
        for i, sc in enumerate(score):
            scores_dict[f"{prefix}bleu_{i + 1}{suffix}"] = sc
        scores_dict[f"{prefix}meteor{suffix}"], _ = self.meteor_scorer.compute_score(gts_tokenized, res_tokenized)

        if self.vocab_verb is not None or self.vocab_ings is not None:
            verbs_count = [0., 0.]  # TP_FN, TP
            ings_count = [0., 0.]  # TP_FN, TP
            for id_, gts_sent in gts_tokenized.items():
                tokens_gt = nltk.tokenize.word_tokenize(gts_sent[0])
                tokens_pr = nltk.tokenize.word_tokenize(res_tokenized[id_][0])
                for token in tokens_gt:
                    if self.vocab_verb is not None and token in self.vocab_verb:
                        verbs_count[0] += 1
                        if token in tokens_pr:
                            verbs_count[1] += 1
                    if self.vocab_ings is not None and token in self.vocab_ings:
                        ings_count[0] += 1
                        if token in tokens_pr:
                            ings_count[1] += 1
            scores_dict[f"{prefix}verbs{suffix}"] = verbs_count[1] / (verbs_count[0] + 1e-9)
            scores_dict[f"{prefix}ingredients{suffix}"] = ings_count[1] / (ings_count[0] + 1e-9)
        return scores_dict

    @abstractmethod
    def extend_gts_res(self, *args):
        pass

    @abstractmethod
    def calculate_scores(self):
        pass


class Evaluator(BaseEvaluator):
    def __init__(self, vocab: Vocabulary, vocab_ingredients_path: Optional[str] = None,
                 vocab_verbs_path: Optional[str] = None, best_k: int = 5):
        super(Evaluator, self).__init__(vocab=vocab, vocab_ingredients_path=vocab_ingredients_path,
                                        vocab_verbs_path=vocab_verbs_path)
        self.gts = {}
        self.sampled_res = {}  # Randomly sampled results
        self.best_sampled_res = {}  # Best sampled results
        self.sentences_number = 0
        self.best_k = best_k
        self.current_scores = {}

    def extend_gts_res(self, r_pred_ids: torch.LongTensor, b_pred_ids: torch.LongTensor, target_ids: torch.LongTensor):
        assert r_pred_ids.shape[0] == b_pred_ids.shape[0] == target_ids.shape[0]
        assert len(r_pred_ids.shape) == 2
        r_pred_ids = r_pred_ids.cpu()
        b_pred_ids = b_pred_ids.cpu()
        target_ids = target_ids.cpu()
        # get instructions
        r_pred_sents = [self._indices2sents(r_pred_ids[i]).replace('|', '').strip() for i in range(r_pred_ids.shape[0])]
        b_pred_sents = [self._indices2sents(b_pred_ids[i]).replace('|', '').strip() for i in range(b_pred_ids.shape[0])]
        target_sents = [self._indices2sents(target_ids[i]).replace('|', '').strip() for i in range(target_ids.shape[0])]
        for i in range(len(target_sents)):
            gts = target_sents[i]
            r_res = r_pred_sents[i]
            b_res = b_pred_sents[i]
            self.gts[str(self.sentences_number)] = [{'caption': gts}]
            self.sampled_res[str(self.sentences_number)] = [{'caption': r_res}]
            self.best_sampled_res[str(self.sentences_number)] = [{'caption': b_res}]
            self.sentences_number += 1

    def calculate_scores(self):
        if self.sentences_number > 0:
            gts = self.tokenizer.tokenize(self.gts)
            r_res = self.tokenizer.tokenize(self.sampled_res)
            b_res = self.tokenizer.tokenize(self.best_sampled_res)
            # Metrics of sampled output
            score, scores = self.bleu_scorer.compute_score(gts, r_res)
            for i, sc in enumerate(score):
                self.current_scores[f"bleu_{i + 1}"] = sc
            self.current_scores["meteor"], _ = self.meteor_scorer.compute_score(gts, r_res)
            # Metrics of best sampled output
            score, scores = self.bleu_scorer.compute_score(gts, b_res)
            for i, sc in enumerate(score):
                self.current_scores[f"bleu_{i + 1}_best{self.best_k}"] = sc
            self.current_scores[f"meteor_best{self.best_k}"], _ = self.meteor_scorer.compute_score(gts, b_res)

            if self.vocab_verb is not None or self.vocab_ings is not None:
                verbs_count = [0., 0., 0.]  # TP_FN, TP random, TP best
                ings_count = [0., 0., 0.]  # TP_FN, TP random, TP best
                for id_, gts_sent in gts.items():
                    tokens_gt = nltk.tokenize.word_tokenize(gts_sent[0])
                    tokens_pr_r = nltk.tokenize.word_tokenize(r_res[id_][0])
                    tokens_pr_b = nltk.tokenize.word_tokenize(b_res[id_][0])
                    for token in tokens_gt:
                        if self.vocab_verb is not None and token in self.vocab_verb:
                            verbs_count[0] += 1
                            if token in tokens_pr_r:
                                verbs_count[1] += 1
                            if token in tokens_pr_b:
                                verbs_count[2] += 1
                        if self.vocab_ings is not None and token in self.vocab_ings:
                            ings_count[0] += 1
                            if token in tokens_pr_r:
                                ings_count[1] += 1
                            if token in tokens_pr_b:
                                ings_count[2] += 1
                self.current_scores["verbs"] = verbs_count[1] / (verbs_count[0] + 1e-9)
                self.current_scores["ingredients"] = ings_count[1] / (ings_count[0] + 1e-9)
                self.current_scores[f'verbs_best{self.best_k}'] = verbs_count[2] / (verbs_count[0] + 1e-9)
                self.current_scores[f'ings_best{self.best_k}'] = ings_count[2] / (ings_count[0] + 1e-9)
            return self.current_scores


class EvaluatorMacro(BaseEvaluator):
    def __init__(self, vocab: Vocabulary, vocab_ingredients_path: Optional[str] = None,
                 vocab_verbs_path: Optional[str] = None, best_k: int = 5):
        super(EvaluatorMacro, self).__init__(vocab=vocab, vocab_ingredients_path=vocab_ingredients_path,
                                             vocab_verbs_path=vocab_verbs_path)
        self.gts = {}
        self.sampled_res = {}  # Randomly sampled results
        self.best_sampled_res = {}  # Best sampled results
        self.sentences_number = 0
        self.best_k = best_k
        self.current_scores = {}

    def extend_gts_res(self, r_pred_ids: torch.LongTensor, b_pred_ids: torch.LongTensor, target_ids: torch.LongTensor):
        assert r_pred_ids.shape[0] == b_pred_ids.shape[0] == target_ids.shape[0]
        assert len(r_pred_ids.shape) == 2
        r_pred_ids = r_pred_ids.cpu()
        b_pred_ids = b_pred_ids.cpu()
        target_ids = target_ids.cpu()
        # get instructions
        r_pred_sents = [self._indices2sents(r_pred_ids[i]) for i in range(r_pred_ids.shape[0])]
        b_pred_sents = [self._indices2sents(b_pred_ids[i]) for i in range(b_pred_ids.shape[0])]
        target_sents = [self._indices2sents(target_ids[i]) for i in range(target_ids.shape[0])]
        for i in range(len(target_sents)):
            gts = target_sents[i]
            r_res = r_pred_sents[i]
            b_res = b_pred_sents[i]
            self.gts[str(self.sentences_number)] = [{'caption': gts}]
            self.sampled_res[str(self.sentences_number)] = [{'caption': r_res}]
            self.best_sampled_res[str(self.sentences_number)] = [{'caption': b_res}]
            self.sentences_number += 1

    def calculate_scores(self):
        if self.sentences_number > 0:
            gts = self.tokenizer.tokenize(self.gts)
            r_res = self.tokenizer.tokenize(self.sampled_res)
            b_res = self.tokenizer.tokenize(self.best_sampled_res)
            for id_ in range(4):
                self.current_scores[f"Mbleu_{id_ + 1}"] = 0
                self.current_scores[f"Mbleu_{id_ + 1}_best{self.best_k}"] = 0
            self.current_scores["Mmeteor"] = 0
            self.current_scores[f"Mmeteor_best{self.best_k}"] = 0
            for i, (id_, g_) in enumerate(gts.items()):
                g = {id_: g_}
                r_r = {id_: r_res[id_]}
                b_r = {id_: b_res[id_]}
                score, scores = self.bleu_scorer.compute_score(g, r_r, verbose=0)
                for j, sc in enumerate(score):
                    self.current_scores[f"Mbleu_{j + 1}"] += sc / len(gts)
                sc, _ = self.meteor_scorer.compute_score(g, r_r)
                self.current_scores["Mmeteor"] += sc / len(gts)
                if (i + 1) % 500 == 0:
                    print(
                        f"{i} samples, Blue-1: {self.current_scores[f'Mbleu_1'] * len(gts) / i},"
                        f" Meteor: {self.current_scores[f'Mmeteor'] * len(gts) / i}")
                # Metrics of best sampled output
                score, scores = self.bleu_scorer.compute_score(g, b_r, verbose=0)
                for j, sc in enumerate(score):
                    self.current_scores[f"Mbleu_{j + 1}_best{self.best_k}"] += sc / len(gts)
                sc, _ = self.meteor_scorer.compute_score(g, b_r)
                self.current_scores[f"Mmeteor_best{self.best_k}"] += sc / len(gts)
                if (i + 1) % 500 == 0:
                    print(
                        f"{i} samples, best Blue-1: {self.current_scores[f'Mbleu_1_best{self.best_k}'] * len(gts) / i},"
                        f" best Meteor: {self.current_scores[f'Mmeteor_best{self.best_k}'] * len(gts) / i}")
        return self.current_scores

