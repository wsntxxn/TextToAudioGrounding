# -*- coding: utf-8 -*-

import json
from pathlib import Path
import joblib
import argparse
import pickle
from sklearn.cluster import KMeans
import numpy as np


class KmeansClustering:

    def kmeans(self, embedding_path, n_clusters=5):

        phrase_to_emb = pickle.load(open(embedding_path, "rb"))
        phrases = []
        embs = []
        for phrase, emb in phrase_to_emb.items():
            phrases.append(phrase)
            embs.append(emb)
        embs = np.stack(embs)

        clf = KMeans(n_clusters=n_clusters)

        y = clf.fit_predict(embs)
        
        score = clf.inertia_

        result = {}
        for text_idx, label_idx in enumerate(y):
            label_idx = str(label_idx)
            if label_idx not in result:
                result[label_idx] = [phrases[text_idx]]
            else:
                result[label_idx].append(phrases[text_idx])
        return {
            "score": score,
            "model": clf,
            "result": result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", type=str, required=True)
    parser.add_argument("--n_cluster", "-nc", type=int, required=True)
    parser.add_argument("--output", "-o", type=str, required=True, help="e.g., experiments/256.json")

    args = parser.parse_args()
    kmeans = KmeansClustering()
    result = kmeans.kmeans(args.embedding, n_clusters=args.n_cluster)

    score = result["score"]
    output = Path(args.output).with_name(Path(args.output).stem + f"_score={score:.0f}" + Path(args.output).suffix)
    ckpt_path = Path(args.output).with_name(Path(args.output).stem + f"_score={score:.0f}.jbl")
    
    if not output.parent.exists():
        (output.parent).mkdir(parents=True)

    joblib.dump(result["model"], ckpt_path)
    with open(output, "w") as writer:
        json.dump(result["result"], writer, indent=2)
