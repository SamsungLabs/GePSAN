# -*- coding: utf-8 -*-

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def indices2words(self, indices, start_token='<start>', end_token='<end>'):
        start_token_idx = self.word2idx[start_token]
        words = [self.idx2word[word_id.item()] for word_id in indices if word_id != start_token_idx]
        try:
            words = words[:words.index(end_token)]
        except ValueError:
            pass
        return words
