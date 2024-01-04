import json
import pickle
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def main(phrase_emb,
         label_emb,
         output):
    phrase_to_emb = pickle.load(open(phrase_emb, "rb"))
    label_to_emb = pickle.load(open(label_emb, "rb"))
    data = []

    embeddings = []
    for label, emb in label_to_emb.items():
        embeddings.append(emb)
    embeddings = np.stack(embeddings)

    for phrase, emb in tqdm(phrase_to_emb.items()):
        sim = cosine_similarity(emb.reshape(1, -1), embeddings)
        indice = sim[0].argmax()
        max_sim = sim[0].max()
        data.append({
            "phrase": phrase,
            "index": indice,
            "sim": max_sim
        })

    pd.DataFrame(data).to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    fire.Fire(main)
