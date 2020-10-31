from tqdm import tqdm
import pandas as pd
import logging
import pickle
import re
import fire

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
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def process(input:str, output:str):
    print("Build Vocab")
    label_df = pd.read_json(input)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')

    # Add the words to the vocabulary.
    for tokens in label_df["tokens"]:
        for token in tokens:
            vocab.add_word(token)
    with open(output, "wb") as store:
        pickle.dump(vocab, store)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved vocab to '{}'".format(output))


if __name__ == '__main__':
    fire.Fire(process)
