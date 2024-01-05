import sys
import os
import json
import pickle
from pathlib import Path
import fire
import torch
from tqdm import trange
from transformers import AutoTokenizer

sys.path.insert(1, os.getcwd())
import audio_text_retrieval_models.audio_encoder as audio_encoder_arch
import audio_text_retrieval_models.text_encoder as text_encoder_arch
import audio_text_retrieval_models.audio_text_model as module_arch

tokenizer_cache = {
    "prajjwal1/bert-medium": "/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/bert_medium_tokenizer",
    "bert-base-uncased": "/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/bert_base_uncased_tokenizer"
}


def get_model(config):
    audio_encoder = getattr(
        audio_encoder_arch, config["audio_encoder"]["type"])(
        **config["audio_encoder"]["args"])
    text_encoder = getattr(
        text_encoder_arch, config["text_encoder"]["type"])(
        **config["text_encoder"]["args"])
    model = getattr(module_arch, config["type"])(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        **config["args"])
    return model


def load_text_encoder(experiment_path, device):
    experiment_path = Path(experiment_path)
    config = json.load(open(experiment_path / "models" / "config.json"))
    model = get_model(config["model"])
    checkpoint = torch.load(experiment_path / "models" / "trained_model.pth",
                            "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer_type = config["data_loader"]["collate_fn"]["args"]["tokenizer_type"]
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    except:
        text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache[tokenizer_type])
    return {
        "tokenizer": text_tokenizer,
        "model": model,
        "config": config
    }


class Executor(object):

    def phrase(self,
               experiment_path,
               phrase_input,
               output,
               debug=False,
               batch_size=128,
               with_proj=False):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_text_encoder(experiment_path, device)
        model = ckpt["model"]
        tokenizer = ckpt["tokenizer"]
        config = ckpt["config"]

        data = json.load(open(phrase_input))
        phrases = []

        for idx, audio_item in enumerate(data):
            for phrase_item in audio_item["phrases"]:
                phrase = phrase_item["phrase"]
                phrases.append(phrase)
            if debug:
                if idx >= 50:
                    break

        phrases = list(set(phrases))

        phrase_to_emb = {}

        for i in trange(0, len(phrases), batch_size):
            texts = phrases[i: i + batch_size]
            tokens = dict(tokenizer(texts, padding="max_length",
                max_length=config["data_loader"]["collate_fn"][
                    "args"]["max_text_length"],
                truncation=True, return_tensors="pt"))
            for k, v in tokens.items():
                tokens[k] = v.to(device)
            with torch.no_grad():
                text_emb = model.text_encoder(**tokens)["clip_emb"]
                if with_proj:
                    text_emb = model.text_proj(text_emb)
                    norm = text_emb.norm(p=2, dim=-1, keepdim=True)
                    text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
                text_emb = text_emb.cpu().numpy()

            for text, emb in zip(texts, text_emb):
                phrase_to_emb[text] = emb

        pickle.dump(phrase_to_emb, open(output, "wb"))

    def add_phrase(self,
                   experiment_path,
                   extra_phrase,
                   phrase_emb,
                   output,
                   batch_size=128,
                   with_proj=False):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_text_encoder(experiment_path, device)
        model = ckpt["model"]
        tokenizer = ckpt["tokenizer"]
        config = ckpt["config"]

        phrases = []
        with open(extra_phrase) as reader:
            for line in reader.readlines():
                phrase = line.strip()
                phrases.append(phrase)

        phrase_to_emb = pickle.load(open(phrase_emb, "rb"))

        for i in trange(0, len(phrases), batch_size):
            texts = phrases[i: i + batch_size]
            tokens = dict(tokenizer(texts, padding="max_length",
                max_length=config["data_loader"]["collate_fn"][
                    "args"]["max_text_length"],
                truncation=True, return_tensors="pt"))
            for k, v in tokens.items():
                tokens[k] = v.to(device)
            with torch.no_grad():
                text_emb = model.text_encoder(**tokens)["clip_emb"]
                if with_proj:
                    text_emb = model.text_proj(text_emb)
                    norm = text_emb.norm(p=2, dim=-1, keepdim=True)
                    text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
                text_emb = text_emb.cpu().numpy()

            for text, emb in zip(texts, text_emb):
                phrase_to_emb[text] = emb

        pickle.dump(phrase_to_emb, open(output, "wb"))

    def label(self,
              experiment_path,
              label_encoder_input,
              output,
              batch_size=128,
              with_proj=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_text_encoder(experiment_path, device)
        model = ckpt["model"]
        tokenizer = ckpt["tokenizer"]
        config = ckpt["config"]

        label_encoder = pickle.load(open(label_encoder_input, "rb"))

        labels = label_encoder.classes_.tolist()
        label_to_emb = {}

        for i in trange(0, len(labels), batch_size):
            texts = labels[i: i + batch_size]
            tokens = dict(tokenizer(texts, padding="max_length",
                max_length=config["data_loader"]["collate_fn"][
                    "args"]["max_text_length"],
                truncation=True, return_tensors="pt"))
            for k, v in tokens.items():
                tokens[k] = v.to(device)
            with torch.no_grad():
                text_emb = model.text_encoder(**tokens)["clip_emb"]
                if with_proj:
                    text_emb = model.text_proj(text_emb)
                    norm = text_emb.norm(p=2, dim=-1, keepdim=True)
                    text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
                text_emb = text_emb.cpu().numpy()

            for text, emb in zip(texts, text_emb):
                label_to_emb[text] = emb

        pickle.dump(label_to_emb, open(output, "wb"))


if __name__ == "__main__":
    fire.Fire(Executor)
