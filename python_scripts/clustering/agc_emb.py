# -*- coding: utf-8 -*-

import json
from pathlib import Path
import argparse
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Runner:

    def agc(self, embedding_path, n_cluster):

        phrase_to_emb = pickle.load(open(embedding_path, "rb"))
        phrases = []
        embs = []
        for phrase, emb in phrase_to_emb.items():
            phrases.append(phrase)
            embs.append(emb)
        embs = np.stack(embs)

        distances = 1 - cosine_similarity(embs, embs)

        clustering = AgglomerativeClustering(
            n_clusters=n_cluster,
            linkage="average",
            affinity="precomputed")

        y = clustering.fit_predict(distances)

        result = {}
        for text_idx, label_idx in enumerate(y):
            label_idx = str(label_idx)
            if label_idx not in result:
                result[label_idx] = [phrases[text_idx]]
            else:
                result[label_idx].append(phrases[text_idx])
        return {
            "result": result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", type=str, required=True)
    parser.add_argument("--n_cluster", "-nc", type=int, required=True)
    parser.add_argument("--output", "-o", type=str, required=True, help="e.g., experiments/256.json")

    args = parser.parse_args()
    obj = Runner()
    result = obj.agc(args.embedding, args.n_cluster)

    output = Path(args.output)
    
    if not output.parent.exists():
        (output.parent).mkdir(parents=True)

    with open(output, "w") as writer:
        json.dump(result["result"], writer, indent=2)
