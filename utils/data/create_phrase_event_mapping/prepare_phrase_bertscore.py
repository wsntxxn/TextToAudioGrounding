import json
import pickle
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from bert_score import score


def prepare(input,
            output,
            label_encoder,
            model_type="microsoft/mpnet-base",
            debug=False,
            batch_size=32):
    data = json.load(open(input))
    phrases = []

    for idx, audio_item in enumerate(data):
        for phrase_item in audio_item["phrases"]:
            phrase = phrase_item["phrase"]
            phrases.append(phrase)

    phrases = list(set(phrases))

    label_encoder = pickle.load(open(label_encoder, "rb"))
    event_labels = label_encoder.classes_.tolist()
    num_events = len(event_labels)

    data = []
    for idx in trange(0, len(phrases), batch_size):
        phrase_batch = phrases[idx: idx + batch_size]
        candidates = []
        for phrase in phrase_batch:
            candidates += [phrase] * num_events
        p, r, f1 = score(candidates,
                         event_labels * len(phrase_batch),
                         model_type=model_type,
                         batch_size=2048)
        scores = f1.reshape(-1, len(event_labels))
        for single_idx, single_score in enumerate(scores):
            single_score = single_score.cpu().numpy()
            indice = single_score.argmax()
            max_score = single_score.max()
            data.append({
                "phrase": phrase_batch[single_idx],
                "index": indice,
                "sim": max_score
            })
        if debug:
            if idx >= 100:
                break
    pd.DataFrame(data).to_csv(output, sep="\t", index=False)



if __name__ == "__main__":
    fire.Fire(prepare)
