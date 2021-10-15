import json
import random
import datetime
import uuid
import pickle
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.engine.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss

from dataset import GroundingDataset, collate_fn
import models.audio_encoder
import models.text_encoder
import models.score
import models.model
import losses
import utils.train_util as train_util
from utils.build_vocab import Vocabulary

class Runner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super(Runner, self).__init__()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def _get_dataloaders(self, config):
        vocabulary = config["vocabulary"]
        dataloaders = {}
        for split in ["train", "val"]:
            audio_feature = config["data"][split]["audio_feature"]
            label_file = config["data"][split]["label"]
            audio_to_h5 = train_util.load_dict_from_csv(
                audio_feature, ("audio_id", "hdf5_path"))
            label = json.load(open(label_file))
            dataloader = torch.utils.data.DataLoader(
                GroundingDataset(
                    audio_to_h5,
                    label,
                    vocabulary),
                shuffle=split == "train",
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"])
            dataloaders[split] = dataloader

        return dataloaders["train"], dataloaders["val"]

    def _get_model(self, config):
        vocabulary = config["vocabulary"]

        audio_encoder = getattr(
            models.audio_encoder, config["audio_encoder"]["type"])(
            **config["audio_encoder"]["args"])
        if "pretrained" in config["audio_encoder"]:
            models.audio_encoder.load_pretrained(
                audio_encoder, config["audio_encoder"]["pretrained"])

        text_encoder = getattr(
            models.text_encoder, config["text_encoder"]["type"])(
            vocab_size=len(vocabulary),
            **config["text_encoder"]["args"])
        if "pretrained_word_embedding" in config["text_encoder"]:
            assert "freeze_word_embedding" in config["text_encoder"]
            weight = np.load(config["text_encoder"]["pretrained_word_embedding"])
            text_encoder.load_pretrained_embedding(
                weight, freeze=config["text_encoder"]["freeze_word_embedding"])

        similarity_fn = getattr(
            models.score, config["similarity_fn"]["type"])(
            **config["similarity_fn"]["args"])

        model = getattr(
            models.model, config["model"]["type"])(
            audio_encoder,
            text_encoder,
            similarity_fn,
            **config["model"]["args"])

        return model

    def _forward(self, model, batch, inference=False):
        audio_feat = batch[0]
        text = batch[1]
        audio_feat_len = batch[-2]
        text_len = batch[-1]
        audio_feat = audio_feat.float().to(self.device)
        text = text.long().to(self.device)

        input_dict = {
            "audio_feat": audio_feat,
            "audio_feat_len": audio_feat_len,
            "text": text,
            "text_len": text_len
        }
        output = model(input_dict)

        if not inference:
            label = batch[2]
            label = label.float().to(self.device)
            output["label"] = label
        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed

        outputdir = Path(conf["outputpath"]) / conf["model"]["type"] / \
            "{}_{}_{}".format(conf["audio_encoder"]["type"],
                              conf["text_encoder"]["type"],
                              conf["similarity_fn"]["type"]) / \
            "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%m"), 
                           uuid.uuid1().hex)

        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            "run",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(str(outputdir / "train.log"))
        # print passed config parameters
        logger.info(f"Storing files in: {outputdir}")
        train_util.pprint_dict(conf, logger.info)

        vocabulary = pickle.load(open(conf["data"]["vocab_file"], "rb"))
        conf["vocabulary"] = vocabulary
        train_loader, val_loader = self._get_dataloaders(conf)
        model = self._get_model(conf)
        model = model.to(self.device)

        optimizer = getattr(
            torch.optim, conf["optimizer"]["type"]
        )(model.parameters(), **conf["optimizer"]["args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        loss_fn = getattr(losses, conf["loss"]["type"])(**conf["loss"]["args"])
        crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch)
                loss = loss_fn(output)
                loss.backward()
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                output = self._forward(model, batch)
                return output

        def thresholded_output_transform(output):
            y_pred = torch.round(output["score"])
            return y_pred, output["label"]

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1 = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            "loss": losses.Loss(loss_fn), 
            "precision": Precision(thresholded_output_transform),
            "recall": Recall(thresholded_output_transform),
            "accuracy": Accuracy(thresholded_output_transform),
            "f1": f1
        }

        evaluator = Engine(_inference)

        metrics["loss"].attach(trainer, "loss")
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
            logger.info, ["loss"], metrics.keys())

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "loss", {
                "model": model.state_dict(),
                "config": conf,
        }, outputdir / "best.pth")

        lr_scheduler = getattr(torch.optim.lr_scheduler, conf["lr_scheduler"]["type"])(
            optimizer, **conf["lr_scheduler"]["args"])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.update_lr,
            lr_scheduler, "loss")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        early_stop_handler = EarlyStopping(
            patience=conf["early_stop"],
            score_function=lambda engine: -engine.state.metrics["loss"],
            trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=conf["epochs"])
        return outputdir

    def evaluate(self,
                 experiment_path,
                 audio_feature,
                 label_file,
                 meta_file,
                 pred_file="hard_predictions.txt",
                 event_file="event.txt",
                 psds_file="psds.txt",
                 time_ratio=10. / 500,
                 threshold=None,
                 window_size=None):
        from tqdm import tqdm
        from psds_eval import PSDSEval

        from dataset import GroundingEvalDataset
        import utils.eval_util as eval_util
        import utils.metrics as metrics

        experiment_path = Path(experiment_path)
        dump = torch.load(experiment_path / "best.pth", map_location="cpu")
        config = dump["config"]
        # vocabulary = pickle.load(open(config["vocab_file"], "rb"))
        vocabulary = config["vocabulary"]
        model = self._get_model(config)
        model.load_state_dict(dump["model"])

        audio_id_to_h5 = train_util.load_dict_from_csv(audio_feature, ("audio_id", "hdf5_path"))
        label = json.load(open(label_file))
        label = [item for item in label if item["timestamps"][0][1] != 0]
        dataset = GroundingEvalDataset(
            audio_id_to_h5, label, vocabulary)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            batch_size=1)

        model = model.to(self.device).eval()

        strong_data = []
        for item in label:
            item_id = f"{item['audiocap_id']}/{item['start_word']}"
            for onset, offset in item["timestamps"]:
                strong_data.append({
                    "filename": item_id,
                    "event_label": "fake_event",
                    "onset": onset,
                    "offset": offset
                })
        strong_label_df = pd.DataFrame(strong_data)

        if threshold is None:
            threshold = 0.5
        thresholds = np.arange(0.0, 1.1, 0.1).tolist()
        if threshold not in thresholds:
            thresholds.append(threshold)
        time_predictions = {th: [] for th in thresholds}
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch", ascii=True, ncols=100):
                infos = batch[2]
                output = self._forward(model, batch, inference=True)

                if window_size is None:
                    window_size = 1

                for th in thresholds:
                    filtered_probs = eval_util.median_filter(
                        output["score"].cpu(), window_size=window_size, threshold=th)
                    for idx in range(len(infos)):
                        info = infos[idx]
                        item_id = f"{info['audiocap_id']}/{info['start_word']}"
                        change_indices = eval_util.find_contiguous_regions(filtered_probs[idx])
                        for row in change_indices:
                            time_predictions[th].append({
                                "filename": item_id,
                                "event_label": "fake_event",
                                "onset": row[0],
                                "offset": row[1]
                            })

        if not (experiment_path / "predictions_for_psds").exists():
            (experiment_path / "predictions_for_psds").mkdir()

        for th in thresholds:
            if len(time_predictions[th]) > 0:
                pred_df = pd.DataFrame(time_predictions[th])
            else:
                pred_df = pd.DataFrame({"filename": [], "event_label": [], "onset": [], "offset": []})
            pred_df = eval_util.predictions_to_time(pred_df, ratio=time_ratio)
            pred_df.to_csv(Path(experiment_path) / f"predictions_for_psds/th_{th:.1f}.csv", index=False, sep="\t")
            if th == threshold:
                pred_df.to_csv(str(Path(experiment_path) / pred_file), index=False, sep="\t")
                event_result, segment_result = metrics.compute_metrics(
                    strong_label_df, pred_df, time_resolution=1.0)
                # event_results_dict = event_result.results_class_wise_metrics()

                with open(str(Path(experiment_path) / event_file), "w") as f:
                    f.write(event_result.__str__())

        meta_df = pd.read_csv(meta_file, sep="\t")
        psds_eval = PSDSEval(0.5, 0.5, 0.3, ground_truth=strong_label_df, metadata=meta_df)
        for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
            csv_file = Path(experiment_path) / f"predictions_for_psds/th_{th:.1f}.csv"
            det_t = pd.read_csv(csv_file, sep="\t")
            info = {"name": f"Op {i + 1}", "threshold": th}
            psds_eval.add_operating_point(det_t, info=info)
            print(f"\rOperating point {i+1} added", end=" ")
        print()
        psds = psds_eval.psds(0.0, 0.0, 100)
        with open(str(Path(experiment_path) / psds_file), "w") as f:
            f.write(f"PSDS-Score: {psds.value:.1%}\n")

    def train_evaluate(self,
                       config,
                       eval_audio_feature,
                       eval_label_file,
                       eval_meta_file,
                       pred_file="hard_predictions.txt",
                       event_file="event.txt",
                       psds_file="psds.txt",
                       time_ratio=10. / 500,
                       threshold=None,
                       window_size=None,
                       **kwargs):
        experiment_path = self.train(config, **kwargs)
        self.evaluate(experiment_path, eval_audio_feature, eval_label_file,
            eval_meta_file, pred_file, event_file, psds_file, time_ratio,
            threshold, window_size)
        return experiment_path

if __name__ == "__main__":
    fire.Fire(Runner)
