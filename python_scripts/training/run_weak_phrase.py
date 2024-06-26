import os
import sys
sys.path.insert(1, os.getcwd())
import warnings
import math
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
import yaml

import utils.train_util as train_util
import utils.eval_util as eval_util


class Runner(object):
    """Main class to run experiments"""
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        self.device = torch.device(device)


    def get_train_dataloader(self):
        cfg = self.config["data"]["train"]
        dataset = train_util.init_obj_from_str(cfg["dataset"])
        collate_fn = train_util.init_obj_from_str(cfg["collate_fn"])
        kwargs = {
            "collate_fn": collate_fn,
            "shuffle": True
        }
        kwargs.update(cfg["dataloader_args"])
        dataloader = torch.utils.data.DataLoader(
            dataset, **kwargs)
        return dataloader


    def get_val_dataloader(self):
        cfg = self.config["data"]["val"]
        dataset = train_util.init_obj_from_str(cfg["dataset"])
        collate_fn = train_util.init_obj_from_str(cfg["collate_fn"])
        kwargs = {
            "collate_fn": collate_fn,
            "shuffle": False
        }
        kwargs.update(cfg["dataloader_args"])
        dataloader = torch.utils.data.DataLoader(
            dataset, **kwargs)
        return dataloader


    def get_model(self, print_fn):

        cfg = self.config["model"]

        kwargs = {}

        for k in cfg:
            if k not in ["type", "args", "pretrained"]:
                sub_model = train_util.init_obj_from_str(cfg[k])
                if "pretrained" in cfg[k]:
                    train_util.load_pretrained_model(
                        sub_model,
                        cfg[k]["pretrained"],
                        print_fn)
                kwargs[k] = sub_model

        model = train_util.init_obj_from_str(cfg, **kwargs)

        return model


    def forward(self, batch, training=True):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "text":
                    batch[k] = v.long().to(self.device)
                else:
                    batch[k] = v.float().to(self.device)

        if not training:
            batch["text"] = batch["text"].unsqueeze(1)
            batch["text_len"] = np.array(batch["text_len"])[:, np.newaxis]

        input_dict = {
            "specaug": False
        }
        input_dict.update(batch)
        output = self.model(input_dict)

        if training:
            output.update(batch)

        return output


    def train_epoch(self):
        loss_history = []
        self.model.train()

        for iteration in trange(self.epoch_length, ascii=True, ncols=100,
                                desc=f"Epoch {self.epoch}/{self.epochs}",
                                leave=False):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            if self.lr_update_interval == "iteration":
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            output = self.forward(batch, training=True)
            loss = self.loss_fn(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            if torch.isnan(loss):
                self.optimizer.zero_grad()
            self.optimizer.step()
            
            if not torch.isnan(loss):
                loss_history.append(loss.item())
            self.iteration += 1

        return {
            "loss": np.mean(loss_history)
        }


    def val_epoch(self):
        loss_history = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, ascii=True,
                              ncols=100, leave=False):
                output = self.forward(batch, training=True)
                loss = self.loss_fn(output)
                loss_history.append(loss.item())
        
        return {
            "loss": np.mean(loss_history),
        }


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
                        "audio_id": audio_id,
                    })
        ground_truth = pd.DataFrame(ground_truth)

        n_thresholds = self.config["eval_config"]["n_thresholds"]
        thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 / n_thresholds)
        psds_buffer = {th: [] for th in thresholds}

        window_size = self.config["inference_args"]["window_size"]
        time_resolution = self.config["data"]["train"]["dataset"][
            "args"]["time_resolution"]
        n_connect = math.ceil(0.5 / time_resolution)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for th in thresholds:
                    filtered_probs = eval_util.median_filter(
                        output["frame_sim"].cpu(),
                        window_size=window_size,
                        threshold=th
                    )
                    for idx in range(len(batch["audiocap_id"])):
                        audiocap_id = batch["audiocap_id"][idx]
                        start_index = batch["start_index"][idx]
                        fname = f"{audiocap_id}_{start_index}"
                        if fname not in ground_truth["filename"].unique():
                            continue
                        change_indices = eval_util.find_contiguous_regions(
                            eval_util.connect_clusters(
                                filtered_probs[idx],
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
                        "onset": onset,
                        "offset": offset,
                    })
        ground_truth = pd.DataFrame(ground_truth)

        n_thresholds = self.config["eval_config"]["n_thresholds"]
        thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 / n_thresholds)
        pred_buffer = {th: [] for th in thresholds}

        window_size = self.config["inference_args"]["window_size"]
        time_resolution = self.config["data"]["train"]["dataset"][
            "args"]["time_resolution"]
        n_connect = math.ceil(0.5 / time_resolution)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch",
                              ascii=True, ncols=100, leave=False):
                output = self.forward(batch, training=False)
                for th in thresholds:
                    filtered_probs = eval_util.median_filter(
                        output["frame_sim"].cpu(),
                        window_size=window_size,
                        threshold=th
                    )
                    for idx in range(len(batch["audiocap_id"])):
                        audiocap_id = batch["audiocap_id"][idx]
                        start_index = batch["start_index"][idx]
                        fname = f"{audiocap_id}_{start_index}"
                        if fname not in ground_truth["filename"].unique():
                            continue
                        change_indices = eval_util.find_contiguous_regions(
                            eval_util.connect_clusters(
                                filtered_probs[idx],
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

    def eval_sed_scores(self, dataloader):
        import sed_scores_eval

        ground_truth = {}
        for audio_item in dataloader.dataset.data:
            audiocap_id = audio_item["audiocap_id"]
            for phrase_item in audio_item["phrases"]:
                start_index = phrase_item["start_index"]
                fname = f"{audiocap_id}_{start_index}"
                ground_truth[fname] = []
                for onset, offset in phrase_item["segments"]:
                    if onset == 0 and offset == 0:
                        continue
                    ground_truth[fname].append((
                        onset,
                        offset,
                        "fake_event"
                    ))

        event_classes = ["fake_event"]
        time_resolution = self.config["data"]["train"]["dataset"][
            "args"]["time_resolution"]
        scores = {}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output = self.forward(batch, training=False)
                for idx in range(len(batch["audiocap_id"])):
                    audiocap_id = batch["audiocap_id"][idx]
                    start_index = batch["start_index"][idx]
                    fname = f"{audiocap_id}_{start_index}"
                    if fname not in ground_truth.keys():
                        continue
                    scores_arr = output["frame_sim"][idx].unsqueeze(-1).cpu().numpy()
                    timestamps = np.arange(output["frame_sim"].shape[1] + 1) * \
                        time_resolution
                    scores[fname] = sed_scores_eval.utils.create_score_dataframe(
                        scores_arr, timestamps=timestamps,
                        event_classes=event_classes)

        return {
            "ground_truth": ground_truth,
            "scores": scores
        }


    def save_checkpoint(self, ckpt_path):
        model_dict = self.model.state_dict()
        ckpt = {
            "model": model_dict,
            "epoch": self.epoch,
            "metric_monitor": self.metric_improver.state_dict(),
            "not_improve_cnt": self.not_improve_cnt
        }
        if self.include_optim_in_ckpt:
            ckpt["optimizer"] = self.optimizer.state_dict()
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, ckpt_path)


    def resume_checkpoint(self, finetune=False, print_fn=print, training=True):
        ckpt = torch.load(self.config["resume"], "cpu")
        load_args = {"training": training}
        if "resume_args" in self.config:
            load_args.update(self.config["resume_args"])
        train_util.load_pretrained_model(self.model, ckpt, print_fn, **load_args)
        if not finetune:
            self.epoch = ckpt["statistics"]["epoch"]
            self.metric_improver.load_state_dict(ckpt["metric_monitor"])
            self.not_improve_cnt = ckpt["not_improve_cnt"]
            if self.optimizer.__class__.__name__ == "Adam":
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        

    def train(self, config, **kwargs):
        self.config = train_util.parse_config_or_kwargs(config, **kwargs)

        if "seed" not in self.config:
            self.config["seed"] = 1

        train_util.set_seed(self.config["seed"])

        exp_dir = Path(self.config["experiment_path"])

        if exp_dir.exists():
            warnings.warn(f"experiment directory {exp_dir} already exists")

        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "config.yaml", "w") as writer:
            yaml.dump(self.config, writer, default_flow_style=False, indent=4)

        self.logger = train_util.init_logger(exp_dir / "train.log")
        if "SLURM_JOB_ID" in os.environ:
            self.logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            self.logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
        elif "JobID" in os.environ:
            self.logger.info(f"Job ID: {os.environ['JobID']}")
        train_util.pprint_dict(self.config, self.logger.info)

        self.train_loader = self.get_train_dataloader()
        self.val_loader = self.get_val_dataloader()
        # self.test_loader = self.get_test_dataloader()
        self.model = self.get_model(self.logger.info).to(self.device)
        train_util.pprint_dict(self.model, self.logger.info, format="pretty")
        num_params = train_util.count_parameters(self.model)
        self.logger.info(f"{num_params} parameters in total")

        self.optimizer = train_util.init_obj_from_str(
            self.config["optimizer"],
            params=self.model.parameters())
        train_util.pprint_dict(self.optimizer, self.logger.info, format="pretty")

        self.loss_fn = train_util.init_obj_from_str(self.config["loss"])

        self.lr_scheduler = train_util.init_obj_from_str(
            self.config["lr_scheduler"],
            optimizer=self.optimizer)

        self.__dict__.update(self.config["trainer"])

        self.metric_improver = train_util.MetricImprover(
            self.metric_monitor["mode"])

        if "resume" in self.config:
            assert "finetune" in self.__dict__, "finetune not being set"
            self.resume_checkpoint(finetune=self.finetune,
                                   print_fn=self.logger.info)

        self.epoch = 1
        self.iteration = 1
        self.not_improve_cnt = 0
        self.train_iter = iter(self.train_loader)
        
        if not hasattr(self, "epoch_length"):
            self.epoch_length = len(self.train_loader)

        for _ in range(self.epochs):
            train_output = self.train_epoch()
            val_result = self.val_epoch()

            val_score = val_result[self.metric_monitor["name"]]

            if self.lr_update_interval == "epoch":
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(val_score)
                else:
                    self.lr_scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_output["loss"]
            output_str = f"epoch: {self.epoch}  train_loss: {train_loss:.2g}" \
                         f"  val_loss: {val_result['loss']:.2g}" \
                         f"  lr: {lr:.2g}"
            self.logger.info(output_str)

            if self.metric_improver(val_score):
                self.not_improve_cnt = 0
                self.save_checkpoint(exp_dir / "best.pth")
            else:
                self.not_improve_cnt += 1

            if self.epoch % self.save_interval == 0:
                self.save_checkpoint(exp_dir / "last.pth")

            if self.not_improve_cnt == self.early_stop:
                break
            
            self.epoch += 1

        self.save_checkpoint(exp_dir / "last.pth")

        return exp_dir

    
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
                    fname = f"{audiocap_id}_{start_index}"
                    if fname not in gt_df["filename"].unique():
                        continue

                    prob = output["frame_sim"][idx, :, 0]
                    prob = torch.clamp(prob, min=0.0, max=1.0)

                    scores_arr = prob.unsqueeze(-1).cpu().numpy()
                    timestamps = np.arange(prob.shape[0] + 1) * \
                        time_resolution
                    score_buffer[fname] = sed_scores_eval.utils.create_score_dataframe(
                        scores_arr, timestamps=timestamps,
                        event_classes=event_classes)
                    for th in thresholds:
                        filtered_prob = eval_util.median_filter(
                            prob.unsqueeze(0).cpu(),
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


    def evaluate(self, experiment_path, eval_config):
        eval_config = train_util.parse_config_or_kwargs(eval_config)

        exp_dir = Path(experiment_path)
        self.config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml" )
        self.config["resume"] = exp_dir / eval_config["resume"]
        self.model = self.get_model(print)
        self.resume_checkpoint(finetune=True)
        

        key_copy_from_train = ["vocabulary"]
        train_util.copy_args_recursive(self.config["data"]["train"],
                                       eval_config["data"]["test"],
                                       key_copy_from_train)

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


    def train_evaluate(self,
                       train_config,
                       eval_config,
                       **kwargs):
        experiment_path = self.train(train_config, **kwargs)
        self.evaluate(experiment_path, eval_config)
        return experiment_path


    def debug(self, config, **kwargs):
        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        train_loader = self.get_train_dataloader()
        self.model = self.get_model(print).to(self.device)
        loss_fn = train_util.init_obj_from_str(self.config["loss"])
        
        train_iter = iter(train_loader)
        for _ in trange(10):
            batch = next(train_iter)
            output = self.forward(batch, training=True)
            loss = loss_fn(output)
            loss.backward()


    def calc_params_macs(self, config, **kwargs):
        from thop import profile, clever_format
        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        self.model = self.get_model(print)
        input_dict = {
            "waveform": torch.randn(1, 320000),
            "waveform_len": torch.tensor([320000]),
            "text": torch.empty(1, 1, 5).random_(10).long(),
            "text_len": torch.tensor([[5]]),
            "specaug": False,
            "text_key": "phrases",
        }
        macs, params = profile(self.model, inputs=(input_dict, ))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
        


if __name__ == "__main__":
    fire.Fire(Runner)
