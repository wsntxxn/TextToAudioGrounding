import json
import pickle
import argparse
from typing import List


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
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def state_dict(self):
        return self.word2idx
    
    def load_state_dict(self, state_dict):
        self.word2idx = state_dict
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.idx = len(self.word2idx)


def process(items: List, output: str):
    print("Build Vocab")

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<unk>")

    # Add the words to the vocabulary.
    for item in items:
        tokens = item["tokens"].split()
        for token in tokens:
            vocab.add_word(token)
    pickle.dump(vocab.state_dict(), open(output, "wb"))
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved vocab to '{}'".format(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()
    data = []
    for label in args.labels:
        items = json.load(open(label))
        data.extend(items)
    process(data, args.output)
