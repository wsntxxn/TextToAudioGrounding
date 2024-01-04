import json
import argparse
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def main(args):
    phrase_to_count = json.load(open(args.phrase_count_path, "r"))
    phrase_to_emb = pickle.load(open(args.embedding_path, "rb"))

    phrase_to_sim_count = {}
    
    embs = []
    cnts = []
    for phrase, emb in phrase_to_emb.items():
        embs.append(emb)
        cnts.append(phrase_to_count[phrase])

    embs = np.stack(embs)
    cnts = np.array(cnts)

    for phrase in tqdm(phrase_to_emb):
        emb = phrase_to_emb[phrase]
        sims = cosine_similarity(emb.reshape(1, -1), embs)
        indices = np.where(sims >= args.threshold)[1]
        count = np.sum(cnts[indices]).__int__()
        phrase_to_sim_count[phrase] = count

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(phrase_to_sim_count, open(args.output_path, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase_count_path", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(args)

