import os
from pathlib import Path, PosixPath
import argparse
import pickle

import numpy as np
import pandas as pd
import hydra
import librosa
import torchaudio
import h5py
import torch
from tqdm import tqdm
import sed_scores_eval

import utils.train_util as train_util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(audio_path, sample_rate):
    if isinstance(audio_path, PosixPath):
        audio_path = audio_path.__str__()
    waveform, orig_sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_sr, sample_rate)
    waveform = waveform.mean(0)
    return waveform


def read_from_h5(key, key_to_h5, cache):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    try:
        return cache[hdf5_path][key][()]
    except KeyError:  # audiocaps compatibility
        key = "Y" + key + ".wav"
        return cache[hdf5_path][key][()]


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, wav_df, sample_rate=32000):
        super().__init__()
        if "file_name" in wav_df.columns:
            self.aid_to_fname = dict(
                zip(wav_df["audio_id"], wav_df["file_name"])
            )
        elif "hdf5_path" in wav_df.columns:
            self.aid_to_h5 = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
            self.h5_cache = {}
        self.aids = wav_df["audio_id"].unique()
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        audio_id = self.aids[index]
        if hasattr(self, "aid_to_fname"):
            waveform = load_audio(
                self.aid_to_fname[audio_id], self.sample_rate
            )
        elif hasattr(self, "aid_to_h5"):
            waveform = read_from_h5(audio_id, self.aid_to_h5, self.h5_cache)
        else:
            raise Exception("Unknown audio input format")
        return {"audio_id": audio_id, "waveform": waveform}


def resume_checkpoint(model, config, print_fn=print):
    ckpt = torch.load(config["resume"], "cpu")
    train_util.load_pretrained_base(
        model=model, ckpt_or_state_dict=ckpt["model"], output_fn=print_fn
    )


def compute_psds(scores, ground_truth, psds_cfg):
    duration = pd.read_csv(
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/sound_event_detection/desed/data/dcase2021/dataset/metadata/eval/public_durations.tsv",
        sep="\t"
    )
    audio_durations = dict(zip(duration["filename"], duration["duration"]))
    psds, _, single_class_rocs = (
        sed_scores_eval.intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=psds_cfg["dtc_threshold"],
            gtc_threshold=psds_cfg["gtc_threshold"],
            cttc_threshold=psds_cfg["cttc_threshold"],
            alpha_st=psds_cfg["alpha_st"],
            alpha_ct=psds_cfg["alpha_ct"],
            max_efpr=100,
            num_jobs=8
        )
    )
    return psds


def load_model_data(args):
    exp_path = args.exp_path
    exp_dir = Path(exp_path)
    config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")
    config["resume"] = exp_dir / "best.pth"
    model = train_util.instantiate_model(config["model"], print)
    resume_checkpoint(model, config)
    model = model.to(DEVICE)
    model.eval()

    wav_df = pd.read_csv(args.wav, sep="\t")
    dataset = InferDataset(wav_df)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4
    )

    tokenizer = hydra.utils.instantiate(
        config["data"]["train_dataloader"]["collate_fn"]["tokenizer"],
        _convert_="all"
    )

    return model, dataloader, tokenizer


classes = [
    "Speech", "Frying", "Dishes", "Running_water", "Blender",
    "Electric_shaver_toothbrush", "Alarm_bell_ringing", "Cat", "Dog",
    "Vacuum_cleaner"
]

class_to_phrase = {
    "Speech": "speaking",
    "Frying": "frying",
    "Dishes": "dishes clanking",
    "Running_water": "water",
    "Blender": "machine running",
    "Electric_shaver_toothbrush": "electric shaver",
    "Alarm_bell_ringing": "ringing",
    "Cat": "cat meowing",
    "Dog": "dog barking",
    "Vacuum_cleaner": "vacuum cleaner running"
}


def evaluate_psds(args):

    model, dataloader, tokenizer = load_model_data(args)
    time_resolution = args.time_resolution

    gt_df = pd.read_csv(
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/sound_event_detection/desed/data/raw_datasets/desed_real/metadata/eval/public.tsv",
        sep="\t"
    )
    score_buffer = {}
    gt_dict = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform = batch["waveform"].to(DEVICE).float()
            audio_id = batch["audio_id"][0]
            sample_df = gt_df[gt_df["filename"] == audio_id]
            gt_dict[audio_id] = []
            for _, row in sample_df.iterrows():
                if row["event_label"] not in classes:
                    continue
                gt_dict[audio_id].append((
                    row["onset"],
                    row["offset"],
                    row["event_label"],
                ))
            input_dict = {
                "specaug": False,
                "waveform": waveform,
                "waveform_len": [waveform.shape[1]]
            }
            scores_arr = []
            for class_ in classes:
                text = class_to_phrase[class_]
                tokens = tokenizer([text])
                for k in model.text_forward_keys:
                    tokens[k] = train_util.unsqueeze_at_1(tokens[k])
                for k, v in tokens.items():
                    if isinstance(v, torch.Tensor):
                        tokens[k] = v.long().to(DEVICE)
                input_dict.update(tokens)
                output = model(input_dict)
                prob = output["frame_sim"][0, :, 0]
                prob = torch.clamp(prob, min=0.0, max=1.0)
                prob = prob.cpu().numpy()
                scores_arr.append(prob)
            scores_arr = np.stack(scores_arr).transpose()
            timestamps = np.arange(scores_arr.shape[0] + 1) * \
                time_resolution
            score_buffer[audio_id] = (
                sed_scores_eval.utils.create_score_dataframe(
                    scores_arr, timestamps=timestamps, event_classes=classes
                )
            )

    psds1_cfg = {
        "dtc_threshold": 0.7,
        "gtc_threshold": 0.7,
        "cttc_threshold": None,
        "alpha_ct": 0,
        "alpha_st": 1
    }
    psds2_cfg = {
        "dtc_threshold": 0.1,
        "gtc_threshold": 0.1,
        "cttc_threshold": 0.3,
        "alpha_ct": 0.5,
        "alpha_st": 1
    }
    psds1 = compute_psds(score_buffer, gt_dict, psds1_cfg)
    psds2 = compute_psds(score_buffer, gt_dict, psds2_cfg)
    print(f"psds1: {psds1:.4f}, psds2: {psds2:.4f}")


def evaluate_op_macro_f1(args):

    import utils.sed_utils as sed_utils
    from psds_eval import PSDSEval

    model, dataloader, tokenizer = load_model_data(args)
    time_resolution = args.time_resolution

    threshold = args.threshold
    if len(threshold) == 1:
        postprocessing_method = sed_utils.binarize
    elif len(threshold) == 2:
        postprocessing_method = sed_utils.double_threshold
    else:
        raise Exception(f"unknown threshold {threshold}")

    pred_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform = batch["waveform"].to(DEVICE).float()
            audio_id = batch["audio_id"][0]
            input_dict = {
                "specaug": False,
                "waveform": waveform,
                "waveform_len": [waveform.shape[1]]
            }
            probs = []
            for class_ in classes:
                text = class_to_phrase[class_]
                tokens = tokenizer([text])
                # TODO this only works for MultiTextBiEncoder, maybe
                # the unsqueeze operation should be done in the model
                for k in model.text_forward_keys:
                    tokens[k] = train_util.unsqueeze_at_1(tokens[k])
                for k, v in tokens.items():
                    if isinstance(v, torch.Tensor):
                        tokens[k] = v.long().to(DEVICE)
                input_dict.update(tokens)
                output = model(input_dict)
                prob = output["frame_sim"][0, :, 0]
                prob = torch.clamp(prob, min=0.0, max=1.0)
                prob = prob.cpu().numpy()
                probs.append(prob)
            probs = np.stack(probs).transpose()[np.newaxis, :, :]
            thresholded_predictions = postprocessing_method(probs, *threshold)
            labelled_predictions = sed_utils.decode_with_timestamps(
                classes, thresholded_predictions
            )

            prediction = labelled_predictions[0]
            for event_label, onset, offset in prediction:
                pred_list.append({
                    "filename": audio_id,
                    "event_label": event_label,
                    "onset": onset,
                    "offset": offset
                })

    output_df = pd.DataFrame(
        pred_list, columns=['filename', 'event_label', 'onset', 'offset']
    )
    output_df = sed_utils.predictions_to_time(output_df, time_resolution)

    output_pred = args.output_pred
    exp_path = Path(args.exp_path)
    if output_pred is None:
        output_pred = exp_path / "pred_desed.csv"
    if not Path(output_pred).parent.exists():
        Path(output_pred).parent.mkdir(parents=True)
    output_df.to_csv(output_pred, sep="\t", index=False, float_format="%.3f")

    groundtruth = pd.read_csv(
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/sound_event_detection/desed/data/raw_datasets/desed_real/metadata/eval/public.tsv",
        sep="\t"
    )
    metadata = pd.read_csv(
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/sound_event_detection/desed/data/dcase2021/dataset/metadata/eval/public_durations.tsv",
        sep="\t"
    )
    psds_eval = PSDSEval(ground_truth=groundtruth, metadata=metadata)
    macro_f, class_f = psds_eval.compute_macro_f_score(output_df)

    output_score = args.output_score
    if output_score is None:
        output_score = exp_path / "score_desed.txt"
    if not Path(output_score).parent.exists():
        Path(output_score).parent.mkdir(parents=True)
    with open(output_score, "w") as writer:
        print(f"macro F-score: {macro_f*100:.2f}", file=writer)
        print(f"macro F-score: {macro_f*100:.2f}")
        for clsname, f in class_f.items():
            print(f"  {clsname}: {f*100:.2f}", file=writer)
            print(f"  {clsname}: {f*100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str)
    parser.add_argument(
        "--wav",
        type=str,
        default=
        "/hpc_stor03/sjtu_home/xuenan.xu/workspace/sound_event_detection/audioset_event_detection/desed_data/wav.csv"
    )
    parser.add_argument("--time_resolution", type=float, default=0.04)
    parser.add_argument(
        "--metric",
        choices=["psds", "op_macro_f1"],
        type=str,
        default="op_macro_f1"
    )
    parser.add_argument(
        "--threshold", nargs="+", type=float, default=[0.75, 0.25]
    )
    parser.add_argument("--output_pred", type=str)
    parser.add_argument("--output_score", type=str)
    args = parser.parse_args()

    if args.metric == "psds":
        evaluate_psds(args)
    elif args.metric == "op_macro_f1":
        evaluate_op_macro_f1(args)
    else:
        raise Exception(f"Unknown metric {args.metric}")
