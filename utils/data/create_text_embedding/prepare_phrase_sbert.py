import json
import pickle
import fire
from sentence_transformers import SentenceTransformer


class Executor(object):

    def phrase(self,
               input,
               output,
               model_type="all-MiniLM-L6-v2",
               debug=False):
        model = SentenceTransformer(model_type)
        data = json.load(open(input))
        phrases = []
        
        for idx, audio_item in enumerate(data):
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                phrases.append(phrase)
            if debug:
                if idx >= 50:
                    break

        phrases = list(set(phrases))
        embeddings = model.encode(phrases)
        phrase_to_emb = {}
        for phrase, embedding in zip(phrases, embeddings):
            phrase_to_emb[phrase] = embedding
        pickle.dump(phrase_to_emb, open(output, "wb"))


    def check_add_missing(self,
                          label,
                          phrase_emb,
                          model_type="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_type)
        data = json.load(open(label))
        phrase_to_emb = pickle.load(open(phrase_emb, "rb"))
        phrases = []
        
        for idx, audio_item in enumerate(data):
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                if phrase not in phrase_to_emb:
                    phrases.append(phrase)

        phrases = list(set(phrases))
        print(phrases)
        embeddings = model.encode(phrases)
        for phrase, embedding in zip(phrases, embeddings):
            phrase_to_emb[phrase] = embedding
        pickle.dump(phrase_to_emb, open(phrase_emb, "wb"))


    def label(self,
              input,
              output,
              model_type="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_type)
        label_encoder = pickle.load(open(input, "rb"))

        labels = label_encoder.classes_
        embeddings = model.encode(labels)
        label_to_emb = {}
        for label, embedding in zip(labels, embeddings):
            label_to_emb[label] = embedding
        pickle.dump(label_to_emb, open(output, "wb"))


if __name__ == "__main__":
    fire.Fire(Executor)
