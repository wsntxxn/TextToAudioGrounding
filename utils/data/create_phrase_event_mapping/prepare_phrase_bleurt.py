import json
import pickle
from bleurt import score
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


def prepare(input,
            output,
            checkpoint,
            label_encoder,
            batch_size=16,
            debug=False):
    scorer = score.BleurtScorer(checkpoint)
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
        scores = scorer.score(references=event_labels * len(phrase_batch), candidates=candidates)
        scores = np.array(scores).reshape(-1, num_events)
        for single_idx, single_score in enumerate(scores):
            indice = single_score.argmax()
            max_score = single_score.max()
            data.append({
                "phrase": phrase_batch[single_idx],
                "index": indice,
                "sim": max_score
            })
        if debug:
            if idx >= 20:
                break
    pd.DataFrame(data).to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    fire.Fire(prepare)
