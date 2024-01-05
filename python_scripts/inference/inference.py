import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import pickle
import torch
import librosa
import fire
import matplotlib.pyplot as plt
import numpy as np

import utils.train_util as train_util
import utils.eval_util as eval_util
from utils.build_vocab import Vocabulary


def get_model(config):

    model = train_util.init_obj_from_str(config["model"])
    return model


def print_pass(*args):
    pass


class Runner:

    def inference_single_text_model(self,
                                    experiment_path,
                                    audio,
                                    phrase,
                                    output,
                                    sample_rate=32000):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        exp_dir = Path(experiment_path)
        config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")

        model = get_model(config)
        ckpt = torch.load(exp_dir / "best.pth", "cpu")
        train_util.load_pretrained_model(model, ckpt, output_fn=print_pass)
        model = model.to(device)
        
        vocabulary = config["data"]["train"]["collate_fn"]["tokenizer"]["args"]["vocabulary"]
        vocabulary = pickle.load(open(vocabulary, "rb"))
        waveform, _ = librosa.core.load(audio, sr=sample_rate)
        duration = waveform.shape[0] / sample_rate
        text = [vocabulary(token) for token in phrase.split()]

        input_dict = {
            "waveform": torch.as_tensor(waveform).unsqueeze(0).to(device),
            "waveform_len": [len(waveform)],
            "text": torch.as_tensor(text).unsqueeze(0).to(device),
            "text_len": [len(text)],
            "specaug": False
        }

        model.eval()
        with torch.no_grad():
            model_output = model(input_dict)
        prob = model_output["prob"].squeeze(0).cpu().numpy()

        filtered_prob = eval_util.median_filter(
            prob[None, :], window_size=1, threshold=0.5)[0]
        change_indices = eval_util.find_contiguous_regions(filtered_prob)
        time_resolution = config["data"]["train"]["dataset"]["args"]["time_resolution"]
        results = []
        for row in change_indices:
            onset = row[0] * time_resolution
            offset = row[1] * time_resolution
            results.append([onset, offset])
        print(results)
        
        plt.figure(figsize=(14, 5))
        plt.plot(prob)
        plt.axhline(y=0.5, color='r', linestyle='--')
        xlabels = [f"{x:.2f}" for x in np.arange(0, duration, duration / 5)]
        plt.xticks(ticks=np.arange(0, len(prob), len(prob) / 5),
                   labels=xlabels,
                   fontsize=15)
        plt.xlabel("Time / second", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        plt.ylim(0, 1)
        if not Path(output).parent.exists():
            Path(output).parent.mkdir(parents=True)
        plt.savefig(output, bbox_inches="tight", dpi=150)


    def inference_multi_text_model(self,
                                   experiment_path,
                                   audio,
                                   phrase,
                                   output,
                                   sample_rate=32000,
                                   threshold=0.5):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        exp_dir = Path(experiment_path)
        config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")

        model = get_model(config)
        ckpt = torch.load(exp_dir / "best.pth", "cpu")
        train_util.load_pretrained_model(model, ckpt, output_fn=print_pass)
        model = model.to(device)
        
        vocab_path = config["data"]["train"]["collate_fn"]["tokenizer"]["args"]["vocabulary"]
        vocabulary = Vocabulary()
        vocabulary.load_state_dict(pickle.load(open(vocab_path, "rb")))

        waveform, _ = librosa.core.load(audio, sr=sample_rate)
        duration = waveform.shape[0] / sample_rate
        text = [vocabulary(token) for token in phrase.split()]

        input_dict = {
            "waveform": torch.as_tensor(waveform).unsqueeze(0).to(device),
            "waveform_len": [len(waveform)],
            "text": torch.as_tensor(text).long().reshape(1, 1, -1).to(device),
            "text_len": torch.tensor(len(text)).reshape(1, 1),
            "specaug": False
        }

        model.eval()
        with torch.no_grad():
            model_output = model(input_dict)
        prob = model_output["frame_sim"][0, :, 0].cpu().numpy()

        filtered_prob = eval_util.median_filter(
            prob[None, :], window_size=1, threshold=threshold)[0]
        change_indices = eval_util.find_contiguous_regions(filtered_prob)
        time_resolution = model.audio_encoder.time_resolution
        results = []
        for row in change_indices:
            onset = row[0] * time_resolution
            offset = row[1] * time_resolution
            results.append([onset, offset])
        print(results)
        
        plt.figure(figsize=(14, 5))
        plt.plot(prob)
        plt.axhline(y=threshold, color='r', linestyle='--')
        xlabels = [f"{x:.2f}" for x in np.arange(0, duration, duration / 5)]
        plt.xticks(ticks=np.arange(0, len(prob), len(prob) / 5),
                   labels=xlabels,
                   fontsize=15)
        plt.xlabel("Time / second", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        plt.ylim(0, 1)
        if not Path(output).parent.exists():
            Path(output).parent.mkdir(parents=True)
        plt.savefig(output, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    fire.Fire(Runner)
