# -*- coding: utf-8 -*-

import json
from pathlib import Path
import joblib
import argparse
import pickle
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Spectral:

    def spectral_sklearn(self, embedding_path, n_clusters=5, seed=0):

        phrase_to_emb = pickle.load(open(embedding_path, "rb"))
        phrases = []
        embs = []
        for phrase, emb in phrase_to_emb.items():
            phrases.append(phrase)
            embs.append(emb)
        embs = np.stack(embs)

        sims = cosine_similarity(embs, embs)
        sims = (sims + 1) / 2

        clf = SpectralClustering(n_clusters=n_clusters,
                                 affinity="precomputed",
                                 random_state=seed)

        y = clf.fit_predict(sims)

        result = {}
        for text_idx, label_idx in enumerate(y):
            label_idx = str(label_idx)
            if label_idx not in result:
                result[label_idx] = [phrases[text_idx]]
            else:
                result[label_idx].append(phrases[text_idx])
        return {
            "model": clf,
            "result": result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", type=str, required=True)
    parser.add_argument("--n_cluster", "-nc", type=int, required=True)
    parser.add_argument("--output", "-o", type=str, required=True, help="e.g., experiments/256.json")
    parser.add_argument("--seed", type=int, required=False, default=1)

    args = parser.parse_args()
    spectral = Spectral()
    result = spectral.spectral_sklearn(args.embedding, n_clusters=args.n_cluster, seed=args.seed)

    output = Path(args.output)
    ckpt_path = Path(args.output).with_name(Path(args.output).stem + ".jbl")
    
    if not output.parent.exists():
        (output.parent).mkdir(parents=True)

    # joblib.dump(result["model"], ckpt_path)
    with open(output, "w") as writer:
        json.dump(result["result"], writer, indent=2)
