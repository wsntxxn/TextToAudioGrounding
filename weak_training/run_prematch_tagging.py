import sys
import os

sys.path.insert(1, os.getcwd())
import warnings
from pathlib import Path
import math

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
import yaml

import utils.train_util as train_util
import utils.eval_util as eval_util
from utils.build_vocab import Vocabulary
from weak_training.run_tagging import Runner as Base


class Runner(Base):


    def forward(self, batch, training=True):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "text":
                    batch[k] = v.long().to(self.device)
                else:
                    batch[k] = v.float().to(self.device)

        if not hasattr(self, "mixup_augmenter"):
            self.mixup_augmenter = train_util.Mixup(mixup_alpha=1.)

        if training and self.config["mixup"]:
            mixup_lambdas = self.mixup_augmenter.get_lambda(batch["waveform"].size(0))
        else:
            mixup_lambdas = None

        if training and self.config["specaug"]:
            specaug = True
        else:
            specaug = False

        input_dict = {
            "specaug": specaug,
            "mixup_lambda": mixup_lambdas
        }
        input_dict.update(batch)
        output = self.model(input_dict)
        output.update(batch)
        if training and self.config["mixup"]:
            output["label"] = train_util.do_mixup(output["label"], mixup_lambdas)

        return output


    def evaluate_tagging(self, experiment_path, eval_config):
        from sklearn.metrics import average_precision_score

        eval_config = train_util.parse_config_or_kwargs(eval_config)
        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)

        self.resume_checkpoint(finetune=True)
        
        if "vocabulary" in self.config["data"]["train"]["dataset"]["args"]:
            eval_config["data"]["test"]["dataset"]["args"][
                "vocabulary"] = self.config["data"]["train"]["dataset"]["args"][
                    "vocabulary"]
        dataset = train_util.init_obj_from_str(
            self.config["data"]["val"]["dataset"])

        collate_fn = train_util.init_obj_from_str(
            self.config["data"]["val"]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1)

        self.model = self.model.to(self.device)
        
        probs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch", ascii=True, ncols=100):
                output = self.forward(batch, training=False)
                prob = output["clip_sim"].cpu().numpy()
                label = batch["label"].cpu().numpy()
                probs.append(prob)
                labels.append(label)
        
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        ap = average_precision_score(labels, probs, average=None)
        num_classes = labels.shape[1]
        result = pd.DataFrame({"label_index": range(num_classes), "ap": ap})
        result = result.fillna(0)
        result.to_csv(exp_dir / eval_config["output"], sep="\t", index=False)


    def eval_psds_sed_scores(self, dataloader, data_duration, return_score=True):

        import sed_scores_eval
        from sed_scores_eval import intersection_based

        ground_truth = {}
        fname_to_aid = {}
        for audio_item in dataloader.dataset.data:
            audiocap_id = audio_item["audiocap_id"]
            audio_id = audio_item["audio_id"]
            for phrase_item in audio_item["phrases"]:
                start_index = phrase_item["start_index"]
                fname = f"{audiocap_id}_{start_index}"
                ground_truth[fname] = []
                fname_to_aid[fname] = audio_id
                for onset, offset in phrase_item["segments"]:
                    if onset == 0 and offset == 0:
                        continue
                    ground_truth[fname].append((
                        onset,
                        offset,
                        "fake_event"
                    ))


        time_resolution = self.config["inference_args"]["time_resolution"]
        event_classes = ["fake_event"]
        scores = {}

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for idx in range(len(batch["audiocap_id"])):
                    audiocap_id = batch["audiocap_id"][idx]
                    start_index = batch["start_index"][idx]
                    text_idx = batch["text_idx"][idx]
                    fname = f"{audiocap_id}_{start_index}"
                    if fname not in ground_truth:
                        continue
                    scores_arr = output["frame_sim"][idx, :, text_idx].unsqueeze(-1).cpu().numpy()
                    timestamps = np.arange(output["frame_sim"].shape[1] + 1) * \
                        time_resolution
                    scores[fname] = sed_scores_eval.utils.create_score_dataframe(
                        scores_arr, timestamps=timestamps,
                        event_classes=event_classes)

        output = {
            "scores": scores,
            "ground_truth": ground_truth,
            "fname_to_aid": fname_to_aid
        }

        if return_score:
            psds = eval_util.compute_psds_sed_scores(
                scores=scores,
                ground_truth=ground_truth,
                duration=data_duration,
                fname_to_aid=fname_to_aid,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )
            output["psds"] = psds

        return output


    def eval_inference(self, dataloader):
        import sed_scores_eval

        gt_list = []
        gt_dict = {}
        fname_to_aid = {}
        for audio_item in dataloader.dataset.data:
            audiocap_id = audio_item["audiocap_id"]
            audio_id = audio_item["audio_id"]
            for phrase_item in audio_item["phrases"]:
                start_index = phrase_item["start_index"]
                fname = f"{audiocap_id}_{start_index}"
                gt_dict[fname] = []
                fname_to_aid[fname] = audio_id
                for onset, offset in phrase_item["segments"]:
                    if onset == 0 and offset == 0:
                        continue
                    gt_list.append({
                        "filename": fname,
                        "event_label": "fake_event",
                        "onset": onset,
                        "offset": offset,
                        "audio_id": audio_id,
                    })
                    gt_dict[fname].append((
                        onset,
                        offset,
                        "fake_event"
                    ))
        gt_df = pd.DataFrame(gt_list)

        event_classes = ["fake_event"]
        n_thresholds = self.config["eval_config"]["n_thresholds"]
        thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 / n_thresholds)
        window_size = self.config["inference_args"]["window_size"]
        time_resolution = self.config["inference_args"]["time_resolution"]
        n_connect = math.ceil(0.5 / time_resolution)

        pred_buffer = {th: [] for th in thresholds}
        score_buffer = {}

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for idx in range(len(batch["audiocap_id"])):
                    audiocap_id = batch["audiocap_id"][idx]
                    start_index = batch["start_index"][idx]
                    text_idx = batch["text_idx"][idx]
                    fname = f"{audiocap_id}_{start_index}"
                    if fname not in gt_df["filename"].unique():
                        continue
                    scores_arr = output["frame_sim"][idx, :, text_idx].unsqueeze(-1).cpu().numpy()
                    timestamps = np.arange(output["frame_sim"].shape[1] + 1) * \
                        time_resolution
                    score_buffer[fname] = sed_scores_eval.utils.create_score_dataframe(
                        scores_arr, timestamps=timestamps,
                        event_classes=event_classes)
                    for th in thresholds:
                        filtered_prob = eval_util.median_filter(
                            output["frame_sim"][idx, :, text_idx].unsqueeze(0).cpu(),
                            window_size=window_size,
                            threshold=th
                        )[0]
                        change_indices = eval_util.find_contiguous_regions(
                            eval_util.connect_clusters(
                                filtered_prob,
                                n_connect
                            )
                        )
                        for row in change_indices:
                            pred_buffer[th].append({
                                "filename": fname,
                                "event_label": "fake_event",
                                "onset": row[0],
                                "offset": row[1]
                                })

        for th in thresholds:
            if len(pred_buffer[th]) > 0:
                pred_df = pd.DataFrame(pred_buffer[th])
            else:
                pred_df = pd.DataFrame({
                    "filename": [],
                    "event_label": [],
                    "onset": [],
                    "offset": []
                })
            pred_df = eval_util.predictions_to_time(
                pred_df, ratio=time_resolution)
            pred_buffer[th] = pred_df

        output = {
            "pred_buffer": pred_buffer,
            "gt_df": gt_df,
            "score_buffer": score_buffer,
            "gt_dict": gt_dict,
            "fname_to_aid": fname_to_aid
        }
        
        return output


    def eval_psds(self, dataloader, data_duration, return_score=True):

        ground_truth = []
        for audio_item in dataloader.dataset.data:
            audiocap_id = audio_item["audiocap_id"]
            audio_id = audio_item["audio_id"]
            for phrase_item in audio_item["phrases"]:
                start_index = phrase_item["start_index"]
                fname = f"{audiocap_id}_{start_index}"
                for onset, offset in phrase_item["segments"]:
                    if onset == 0 and offset == 0:
                        continue
                    ground_truth.append({
                        "filename": fname,
                        "event_label": "fake_event",
                        "onset": onset,
                        "offset": offset,
                        "audio_id": audio_id
                    })
        ground_truth = pd.DataFrame(ground_truth)

        n_thresholds = self.config["eval_config"]["n_thresholds"]
        thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 / n_thresholds)
        psds_buffer = {th: [] for th in thresholds}

        window_size = self.config["inference_args"]["window_size"]
        time_resolution = self.config["inference_args"]["time_resolution"]
        n_connect = math.ceil(0.5 / time_resolution)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for th in thresholds:
                    for idx in range(len(batch["audiocap_id"])):
                        audiocap_id = batch["audiocap_id"][idx]
                        start_index = batch["start_index"][idx]
                        text_idx = batch["text_idx"][idx]
                        fname = f"{audiocap_id}_{start_index}"
                        if fname not in ground_truth["filename"].unique():
                            continue
                        filtered_probs = eval_util.median_filter(
                            output["frame_sim"][idx].cpu(),
                            window_size=window_size,
                            threshold=th
                        )
                        change_indices = eval_util.find_contiguous_regions(
                            eval_util.connect_clusters(
                                filtered_probs[:, text_idx],
                                n_connect
                            )
                        )
                        for row in change_indices:
                            psds_buffer[th].append({
                                "filename": fname,
                                "event_label": "fake_event",
                                "onset": row[0],
                                "offset": row[1]
                            })

        for th in thresholds:
            if len(psds_buffer[th]) > 0:
                pred_df = pd.DataFrame(psds_buffer[th])
            else:
                pred_df = pd.DataFrame({
                    "filename": [],
                    "event_label": [],
                    "onset": [],
                    "offset": []
                })
            pred_df = eval_util.predictions_to_time(
                pred_df, ratio=time_resolution)
            psds_buffer[th] = pred_df

        output = {
            "psds_buffer": psds_buffer,
            "ground_truth": ground_truth
        }

        if return_score:
            psds_score = eval_util.compute_psds(
                psds_buffer,
                ground_truth,
                data_duration,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )
            output["psds"] = psds_score
        
        return output


    def evaluate_psds(self, experiment_path, eval_config):
        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        
        key_copy_from_train = ["phrase_embed", "as_label_embed", "cluster_model"]
        for key in key_copy_from_train:
            if key in self.config["data"]["train"]["dataset"]["args"]:
                eval_config["data"]["test"]["dataset"]["args"][
                    key] = self.config["data"]["train"]["dataset"]["args"][key]

        dataset = train_util.init_obj_from_str(
            eval_config["data"]["test"]["dataset"])
        
        # for idx in range(len(dataset)):
            # try:
                # item = dataset[idx]
            # except KeyError:
                # audio_idx, phrase_idx = dataset.idxs[idx]
                # audio_item = dataset.data[audio_idx]
                # phrase_item = audio_item["phrases"][phrase_idx]
                # text = phrase_item["phrase"]
                # print(text)

        collate_fn = train_util.init_obj_from_str(
            eval_config["data"]["test"]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1)

        self.model = self.model.to(self.device)

        if "eval_config" not in self.config:
            self.config["eval_config"] = {}
        if "inference_args" not in self.config:
            self.config["inference_args"] = {}
        self.config["eval_config"]["n_thresholds"] = eval_config["n_thresholds"]
        self.config["inference_args"]["window_size"] = eval_config["window_size"]
        self.config["inference_args"]["time_resolution"] = eval_config[
            "time_resolution"]
        
        output = self.eval_psds(dataloader,
                                eval_config["data"]["test"]["duration"],
                                return_score=False)

        psds_buffer = output["psds_buffer"]
        ground_truth = output["ground_truth"]

        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for th in psds_buffer:
            psds_buffer[th].to_csv(
                pred_dir / f"predictions_th_{th:.2f}.tsv",
                sep="\t",
                index=False,
            )

        psds_dir = eval_config.get("psds_dir", "psds")

        psds_score_scenario1 = eval_util.compute_psds(
            psds_buffer,
            ground_truth,
            eval_config["data"]["test"]["duration"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            save_dir=exp_dir / psds_dir,
        )

        psds_score_scenario2 = eval_util.compute_psds(
            psds_buffer,
            ground_truth,
            eval_config["data"]["test"]["duration"],
            dtc_threshold=0.5,
            gtc_threshold=0.5,
            save_dir=exp_dir / psds_dir,
        )

        psds_score_scenario3 = eval_util.compute_psds(
            psds_buffer,
            ground_truth,
            eval_config["data"]["test"]["duration"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            save_dir=exp_dir / psds_dir,
        )

        f_output = exp_dir / eval_config["output_psds"]
        if not f_output.parent.exists():
            f_output.parent.mkdir(parents=True)
        with open(f_output.__str__(), "w") as writer:
            print(f"psds_scenario1: {psds_score_scenario1:.1%}")
            print(f"psds_scenario1: {psds_score_scenario1:.1%}", file=writer)
            print(f"psds_scenario2: {psds_score_scenario2:.1%}")
            print(f"psds_scenario2: {psds_score_scenario2:.1%}", file=writer)
            print(f"psds_scenario3: {psds_score_scenario3:.1%}")
            print(f"psds_scenario3: {psds_score_scenario3:.1%}", file=writer)


    def evaluate_psds_single_example(self, experiment_path, eval_config):
        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        
        if "vocabulary" in self.config["data"]["train"]["dataset"]["args"]:
            eval_config["data"]["test"]["dataset"]["args"][
                "vocabulary"] = self.config["data"]["train"]["dataset"]["args"][
                    "vocabulary"]
        dataset = train_util.init_obj_from_str(
            eval_config["data"]["test"]["dataset"])
        collate_fn = train_util.init_obj_from_str(
            eval_config["data"]["test"]["collate_fn"])

        self.model = self.model.to(self.device)

        all_data = dataset.data
        results = []
        for single_data in tqdm(all_data, ascii=True):
            if single_data["audiocap_id"] != 766:
                continue
            dataset.data = [single_data]
            dataset.generate_index()
            dataloader = torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=1)
            
            output = self.eval_psds(dataloader,
                                    eval_config["data"]["test"]["duration"])

            results.append({
                "audiocap_id": dataset.data[0]["audiocap_id"],
                "psds": output["psds"]
            })

        pd.DataFrame(results).to_csv(exp_dir / eval_config["output"],
                sep="\t", index=False)


    def eval_th_auc(self, dataloader, return_score=True):
        ground_truth = []
        for audio_item in dataloader.dataset.data:
            audiocap_id = audio_item["audiocap_id"]
            for phrase_item in audio_item["phrases"]:
                start_index = phrase_item["start_index"]
                fname = f"{audiocap_id}_{start_index}"
                for onset, offset in phrase_item["segments"]:
                    if onset == 0 and offset == 0:
                        continue
                    ground_truth.append({
                        "filename": fname,
                        "event_label": "fake_event",
                        "onset": onset,
                        "offset": offset,
                    })
        ground_truth = pd.DataFrame(ground_truth)

        n_thresholds = self.config["eval_config"]["n_thresholds"]
        thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 / n_thresholds)
        pred_buffer = {th: [] for th in thresholds}

        window_size = self.config["inference_args"]["window_size"]
        time_resolution = self.config["inference_args"]["time_resolution"]
        n_connect = math.ceil(0.5 / time_resolution)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for idx in range(len(batch["audiocap_id"])):
                    audiocap_id = batch["audiocap_id"][idx]
                    start_index = batch["start_index"][idx]
                    text_idx = batch["text_idx"][idx]
                    fname = f"{audiocap_id}_{start_index}"
                    if fname not in ground_truth["filename"].unique():
                        continue
                    prob = output["frame_sim"][idx, :, text_idx].cpu().numpy()
                    for th in thresholds:
                        filtered_prob = eval_util.median_filter(
                            prob[:, None],
                            window_size=window_size,
                            threshold=th
                        )[:, 0]
                        change_indices = eval_util.find_contiguous_regions(
                            eval_util.connect_clusters(
                                filtered_prob,
                                n_connect
                            )
                        )
                        for row in change_indices:
                            pred_buffer[th].append({
                                "filename": fname,
                                "event_label": "fake_event",
                                "onset": row[0],
                                "offset": row[1]
                            })

        for th in thresholds:
            if len(pred_buffer[th]) > 0:
                pred_df = pd.DataFrame(pred_buffer[th])
            else:
                pred_df = pd.DataFrame({
                    "filename": [],
                    "event_label": [],
                    "onset": [],
                    "offset": []
                })
            pred_df = eval_util.predictions_to_time(
                pred_df, ratio=time_resolution)
            pred_buffer[th] = pred_df

        output = {
            "pred_buffer": pred_buffer,
            "ground_truth": ground_truth
        }

        if return_score:
            th_auc = eval_util.compute_th_auc(
                pred_buffer,
                ground_truth,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )
            output["th_auc"] = th_auc
        
        return output


    def evaluate_th_auc(self,
                        experiment_path,
                        eval_config):

        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        
        key_copy_from_train = ["phrase_embed", "as_label_embed", "cluster_model"]
        for key in key_copy_from_train:
            if key in self.config["data"]["train"]["dataset"]["args"]:
                eval_config["data"]["test"]["dataset"]["args"][
                    key] = self.config["data"]["train"]["dataset"]["args"][key]
        self.config["eval_config"] = eval_config
        dataset = train_util.init_obj_from_str(
            eval_config["data"]["test"]["dataset"])
        collate_fn = train_util.init_obj_from_str(
            eval_config["data"]["test"]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1)

        self.model = self.model.to(self.device)

        if "eval_config" not in self.config:
            self.config["eval_config"] = {}
        if "inference_args" not in self.config:
            self.config["inference_args"] = {}
        self.config["eval_config"]["n_thresholds"] = eval_config["n_thresholds"]
        self.config["inference_args"]["window_size"] = eval_config["window_size"]
        self.config["inference_args"]["time_resolution"] = eval_config[
            "time_resolution"]
        output = self.eval_th_auc(dataloader, return_score=False)
                                  
        
        pred_buffer = output["pred_buffer"]
        ground_truth = output["ground_truth"]

        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for th in pred_buffer:
            pred_buffer[th].to_csv(
                pred_dir / f"predictions_th_{th:.2f}.tsv",
                sep="\t",
                index=False,
            )

        th_auc_dir = eval_config.get("th_auc_dir", "th_auc")

        th_auc_scenario1 = eval_util.compute_th_auc(
            pred_buffer,
            ground_truth,
            dtc_threshold=0.5,
            gtc_threshold=0.5,
            save_dir=exp_dir / th_auc_dir,
        )

        f_output = exp_dir / eval_config["output_th_auc"]
        if not f_output.parent.exists():
            f_output.parent.mkdir(parents=True)
        with open(f_output.__str__(), "w") as writer:
            print(f"th_auc_scenario1: {th_auc_scenario1:.1%}")
            print(f"th_auc_scenario1: {th_auc_scenario1:.1%}", file=writer)


    def evaluate(self, experiment_path, eval_config):
        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        
        # key_copy_from_train = ["phrase_embed", "as_label_embed", "cluster_model"]
        key_copy_from_train = ["as_label_embed", "cluster_model"]
        for key in key_copy_from_train:
            if key in self.config["data"]["train"]["dataset"]["args"]:
                eval_config["data"]["test"]["dataset"]["args"][
                    key] = self.config["data"]["train"]["dataset"]["args"][key]

        dataset = train_util.init_obj_from_str(
            eval_config["data"]["test"]["dataset"])
        collate_fn = train_util.init_obj_from_str(
            eval_config["data"]["test"]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1)

        self.model = self.model.to(self.device)

        if "eval_config" not in self.config:
            self.config["eval_config"] = {}
        if "inference_args" not in self.config:
            self.config["inference_args"] = {}
        self.config["eval_config"]["n_thresholds"] = eval_config["n_thresholds"]
        self.config["inference_args"]["window_size"] = eval_config["window_size"]
        self.config["inference_args"]["time_resolution"] = eval_config[
            "time_resolution"]

        output = self.eval_inference(dataloader)

        pred_buffer = output["pred_buffer"]

        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for th in pred_buffer:
            pred_buffer[th].to_csv(
                pred_dir / f"predictions_th_{th:.2f}.tsv",
                sep="\t",
                index=False,
            )

        psds_dir = eval_config.get("psds_dir", "psds")

        f_output = exp_dir / eval_config["output"]
        if not f_output.parent.exists():
            f_output.parent.mkdir(parents=True)
        writer = open(f_output.__str__(), "w")

        for max_efpr in eval_config["max_efprs"]:
            # psds = eval_util.compute_psds(
                # pred_buffer,
                # output["gt_df"],
                # eval_config["data"]["test"]["duration"],
                # dtc_threshold=0.5,
                # gtc_threshold=0.5,
                # save_dir=exp_dir / psds_dir,
                # max_efpr=max_efpr
            # )
            psds = eval_util.compute_psds_sed_scores(
                scores=output["score_buffer"],
                ground_truth=output["gt_dict"],
                duration=eval_config["data"]["test"]["duration"],
                fname_to_aid=output["fname_to_aid"],
                dtc_threshold=0.5,
                gtc_threshold=0.5,
                save_dir=exp_dir / psds_dir,
                max_efpr=max_efpr
            )
            print(f"max_efpr: {max_efpr}, psds: {psds:.1%}")
            print(f"max_efpr: {max_efpr}, psds: {psds:.1%}", file=writer)

        th_auc_dir = eval_config.get("th_auc_dir", "th_auc")
        for min_th, max_th in zip([0.0, 0.2, 0.0], [1.0, 0.8, 0.8]):
            th_auc = eval_util.compute_th_auc(
                pred_buffer,
                output["gt_df"].drop(["event_label", "audio_id"], axis=1),
                dtc_threshold=0.5,
                gtc_threshold=0.5,
                min_threshold=min_th,
                max_threshold=max_th,
                save_dir=exp_dir / th_auc_dir
            )
            print(f"threshold: {min_th:.2f} ~ {max_th:.2f}, th_auc: {th_auc:.1%}")
            print(f"threshold: {min_th:.2f} ~ {max_th:.2f}, th_auc: {th_auc:.1%}",
                  file=writer)

        writer.close()


    def inference_clotho(self, experiment_path, eval_config):
        
        import json

        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        
        key_copy_from_train = ["phrase_embed", "as_label_embed", "cluster_model"]
        for key in key_copy_from_train:
            if key in self.config["data"]["train"]["dataset"]["args"]:
                eval_config["data"]["test"]["dataset"]["args"][
                    key] = self.config["data"]["train"]["dataset"]["args"][key]

        dataset = train_util.init_obj_from_str(
            eval_config["data"]["test"]["dataset"])
        collate_fn = train_util.init_obj_from_str(
            eval_config["data"]["test"]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1)

        self.model = self.model.to(self.device)

        outputs = []
        output_aids = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, leave=False, ncols=100):
                output = self.forward(batch, training=False)
                for idx in range(len(batch["audiocap_id"])):
                    text_idx = batch["text_idx"][idx]
                    audiocap_id = batch["audiocap_id"][idx]
                    start_index = batch["start_index"][idx]
                    filtered_prob = eval_util.median_filter(
                        output["frame_sim"][idx, :, text_idx].unsqueeze(0).cpu(),
                        window_size=1,
                        threshold=0.5
                    )[0]
                    fname = f"{audiocap_id}_{start_index}"
                    change_indices = eval_util.find_contiguous_regions(
                        filtered_prob,
                    )
                    audio_id = batch["audio_id"][idx]
                    if len(change_indices) > 1 and len(change_indices) < 5:
                        if audio_id not in output_aids:
                            output_aids.append(audio_id)
                            outputs.append(fname)

        output_file = exp_dir / eval_config["output"]
        if not output_file.parent.exists():
            output_file.parent.mkdir()
        json.dump(outputs, open(output_file, "w"), indent=4)


    def calc_label_num(self, config, output):
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.config = train_util.parse_config_or_kwargs(config)
        train_loader = self.get_train_dataloader()
        labels = []
        for batch in tqdm(train_loader):
            labels.append(batch["label"])
        labels = np.concatenate(labels)
        label_num = labels.sum(0)
        pd.DataFrame({"number": label_num}).to_csv(output)


if __name__ == "__main__":
    fire.Fire(Runner)
