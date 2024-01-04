# coding=utf-8
#!/usr/bin/env python3

import functools
import pickle
from typing import Set, List, Tuple

import numpy as np
import pandas as pd
import gensim
import spacy
import pkg_resources
from tqdm import tqdm
import fire
from symspellpy import SymSpell, Verbosity

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from utils.build_vocab import Vocabulary


@functools.lru_cache(maxsize=64, typed=False)
def load_w2v_model_from_cache(w2v_weights: Path):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        fname=w2v_weights,
        binary=True,
    )
    return model


class W2V_Lookup:
    def __init__(self, w2v):
        self.w2v = w2v
        self.vocab = set(w2v.vocab.keys())

    def __call__(self, key):
        return self.w2v.get_vector(key)


class Tokenizer:
    """For word-level embeddings, we convert words that are absent from the embedding lookup table to a canonical tokens (and then re-check the table). This is to ensure that we get reasonable embeddings for as many words as possible.
    """

    def __init__(self, vocab: Set[str]):
        # we only use spacy for lemmatising, so we don't need NER or the parser.
        # NOTE: all pronouns are mapped to -PRON-, because it's not clear with
        # their lemma should be (we try to handle these via the spellchecker)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
        # Symspell is, in theory, a fast spell checker:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy","frequency_dictionary_en_82_765.txt")
        # term_index is the column of the term and count_index us the
        # column of the term frequency
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell = sym_spell
        self.vocab = vocab
        # For a small number of cases, the tokenizer fauks
        self.custom = {
            "roundtable": ["round", "table"],
            "laughingthen": ["laughing", "then"],
            "aneengine": ["an", "engine"]
        }

    def __call__(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens, failed = [], []
        for token in doc:
            token, lemma = str(token), token.lemma_
            if token in self.vocab:
                tokens.append(token)
            elif lemma in self.vocab:
                tokens.append(lemma)
            elif lemma in self.custom:
                for subtoken in self.custom[lemma]:
                    if subtoken in self.vocab:
                        tokens.append(subtoken)
                    else:
                        failed.append(subtoken)
            else:
                suggestions = self.sym_spell.lookup(
                    phrase = token,
                    verbosity = Verbosity.CLOSEST,
                    max_edit_distance=2,
                )
                success = False
                for suggestion in suggestions:
                    if suggestion.term in self.vocab:
                        success = True
                        tokens.append(suggestion.term)
                        break
                if not success:
                    failed.append(str(token))
        return [tokens, failed]


class W2VEmbedding:

    def __init__(self,
                 dim: int,
                 weights_path: Path,
                 num_samples_for_unknown: int = 50000):
        w2v = load_w2v_model_from_cache(weights_path)
        model = W2V_Lookup(w2v=w2v)
        tokenizer = Tokenizer(vocab=model.vocab)
        vecs = np.zeros((min(num_samples_for_unknown, len(model.vocab)),dim))
        for ii, key in enumerate(sorted(model.vocab)):
            if ii >= num_samples_for_unknown:
                break
            vecs[ii] = model(key)
        self.unknown_vector = np.mean(vecs, 0)
        self.model = model
        self.dim = dim
        self.tokenizer = tokenizer

    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        tokens, failed = self.tokenizer(text)
        embeddings = []
        for token in tokens:
            embeddings.append(self.model(token))
        embeddings = np.array(embeddings)
        msg = (f"Failed to embed any tokens! (text: {text}, failed: {failed})")
        # For empty sequences, we use zeros with the dimensionality of the features on
        if embeddings.size == 0:
            print(f"Warning: {msg}, falling back to unknown vector")
            embeddings = np.array([self.unknown_vector])

        embedding = np.mean(embeddings, axis=0)
        return embedding


def main(vocab_file: str,
         output: str,
         pretrained_weight: str,
         embed_dim: int):

    with open(vocab_file, "rb") as vocab_store:
        vocabulary = pickle.load(vocab_store)

    model = W2VEmbedding(embed_dim, pretrained_weight)

    embedding_matrix = np.zeros((len(vocabulary), embed_dim))
    
    with tqdm(total=len(vocabulary), ascii=True) as pbar:
        for word, idx in vocabulary.word2idx.items():
            embedding = model.text2vec(word)
            embedding_matrix[idx] = embedding
            pbar.update()

    np.save(output, embedding_matrix)
    print("Finish writing word2vec embeddings to " + output)


if __name__ == "__main__":
    fire.Fire(main)



