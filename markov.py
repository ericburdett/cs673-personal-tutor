import os
import re
from collections import Counter

import numpy as np

_pattern_repls = [
    (re.compile(r'\n'), ' \n '),
    (re.compile(r' {2,}'), ' '),
    (re.compile('\''), ''),
]

def replace(text, pattern_repls):
    for pattern, repl in pattern_repls:
        text = re.sub(pattern, repl, text)
    return text


datadir = 'data'
filename = 'tswift_lyrics.txt'
with open(os.path.join(datadir, filename)) as infile:
    lyrics = infile.read()

text = replace(lyrics, _pattern_repls)


class MarkovModel:
    def __init__(self, text=None):
        self.text = text
        self.probs = None
        self.word_counts = None
        if self.text is not None:
            self.fit_from_text(text)

    def fit_from_text(self, text):
        splits = text.split(' ')

        self.word_counts = Counter(splits)
        self.words = list(self.word_counts)
        self._word_inds = {w: i for i, w in enumerate(self.word_counts)}

        n_words = len(self.word_counts)
        bigram = np.zeros((n_words, n_words))

        for w1, w2 in zip(splits, splits[1:]):
            i1, i2 = self._word_inds[w1], self._word_inds[w2]
            bigram[i1, i2] += 1
        self.bigram = bigram / bigram.sum(axis=1)[:, np.newaxis]

    def get_generator(self, start=None):
        if start is None:
            start = '\n'
        cur = self._word_inds[start]
        yield start
        while True:
            nxt = np.random.choice(len(self.words), p=self.bigram[cur])
            yield self.words[nxt]
            cur = nxt





