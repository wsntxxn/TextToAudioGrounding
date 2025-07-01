import os
import sys
import warnings
import math
from pathlib import Path

import fire
import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
import yaml

import utils.train_util as train_util
import utils.eval_util as eval_util
from python_scripts.training.run_weak_phrase import Runner as Base


class Runner(Base):
    def get_teacher(self, print_fn):
        return train_util.instantiate_model(self.config["teacher"], print_fn)

    def forward(self, batch, training=True):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "text":
                    batch[k] = v.long().to(self.device)
                else:
                    batch[k] = v.float().to(self.device)

        if not training:
            for key in self.model.text_forward_keys:
                batch[key] = batch[key].unsqueeze(1)

        input_dict = {"specaug": False}
        input_dict.update(batch)
        output = self.model(input_dict)

        if training:
            output.update(batch)

        with torch.no_grad():
            teacher_output = self.teacher(input_dict)
            output["label"] = torch.max(
                torch.stack([batch["label"], teacher_output["clip_sim"]]),
                dim=0,
            )[0]
            output["frame_label"] = teacher_output["frame_sim"]

        return output

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
        self.model = self.get_model(self.logger.info).to(self.device)
        train_util.pprint_dict(self.model, self.logger.info, format="pretty")
        num_params = train_util.count_parameters(self.model)
        self.logger.info(f"{num_params} parameters in total")

        self.teacher = self.get_teacher(print).to(self.device)
        self.teacher.eval()

        self.optimizer = hydra.utils.instantiate(
            self.config["optimizer"],
            params=self.model.parameters(),
            _convert_="all"
        )
        train_util.pprint_dict(
            self.optimizer, self.logger.info, format="pretty"
        )

        self.loss_fn = hydra.utils.instantiate(
            self.config["loss"], _convert_="all"
        )

        self.__dict__.update(self.config["trainer"])
        self.epoch = 1
        self.iteration = 1
        self.not_improve_cnt = 0
        self.train_iter = iter(self.train_loader)
        if not hasattr(self, "epoch_length"):
            self.epoch_length = len(self.train_loader)

        self.total_steps = self.epochs * self.epoch_length

        lr_scheduler_params = {"optimizer": self.optimizer}
        if self.config["lr_scheduler"][
            "_target_"] == "transformers.get_cosine_schedule_with_warmup":
            lr_scheduler_params["num_training_steps"] = self.total_steps
        self.lr_scheduler = hydra.utils.instantiate(
            self.config["lr_scheduler"],
            _convert_="all",
            **lr_scheduler_params
        )

        self.metric_improver = train_util.MetricImprover(
            self.metric_monitor["mode"]
        )

        if "resume" in self.config:
            assert "finetune" in self.__dict__, "finetune not being set"
            self.resume_checkpoint(
                finetune=self.finetune, print_fn=self.logger.info
            )

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
            output_str = f"epoch: {self.epoch}  train_loss: {train_loss:.3g}" \
                         f"  val_loss: {val_result['loss']:.3g}" \
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

    def debug(self, config, **kwargs):
        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        train_loader = self.get_train_dataloader()
        self.model = self.get_model(print).to(self.device)
        self.teacher = self.get_teacher(print).to(self.device)
        loss_fn = train_util.init_obj_from_str(self.config["loss"])

        train_iter = iter(train_loader)
        for _ in trange(10):
            batch = next(train_iter)
            output = self.forward(batch, training=True)
            loss = loss_fn(output)
            loss.backward()


if __name__ == "__main__":
    fire.Fire(Runner)
