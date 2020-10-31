import datetime
import uuid
import pickle
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pytorch_model_summary import summary
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.engine.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, Loss, RunningAverage, Average

from datasets.GroundingDataset import QueryAudioDataset, collate_fn
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary

class Runner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super(Runner, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def _get_dataloaders(self, config, vocabulary):
        train_df = pd.read_json(config["train_label"])
        val_df = pd.read_json(config["val_label"])

        train_loader = torch.utils.data.DataLoader(
            QueryAudioDataset(
                config["audio_feature"],
                train_df,
                query_form="wordids",
                vocabulary=vocabulary),
            shuffle=True,
            collate_fn=collate_fn([0,1]),
            **config["dataloader_args"])

        val_loader = torch.utils.data.DataLoader(
            QueryAudioDataset(
                config["audio_feature"],
                val_df,
                query_form="wordids",
                vocabulary=vocabulary),
            shuffle=False,
            collate_fn=collate_fn([0,1]),
            **config["dataloader_args"])

        return train_loader, val_loader

    def _get_model(self, config, vocabulary):
        embeddim = config["model_args"]["embeddim"]
        audio_encoder = getattr(
            models.AudioEncoder, config["audio_encoder"])(
            inputdim=config["audio_inputdim"],
            embeddim=embeddim)
        if "pretrained_encoder" in config:
            encoder_state_dict = torch.load(
                config["pretrained_encoder"],
                map_location="cpu")
            audio_encoder.load_state_dict(encoder_state_dict, strict=False)
        text_encoder = getattr(
            models.TextEncoder, config["text_encoder"])(
            vocabsize=len(vocabulary), embeddim=embeddim)
        if "pretrained_embedding" in config:
            weight = np.load(config["pretrained_embedding"])
            text_encoder.load_pretrained_embedding(weight, tune=config["tune_embedding"])
        model = getattr(
            models.model, config["model"])(
            inputdim=embeddim,
            embeddim=config["model_args"]["projdim"],
            audioEncoder=audio_encoder,
            phraseEncoder=text_encoder,
            use_siamese=config["model_args"]["use_siamese"])
        if config["pretrained"]:
            for param in model.audioEncoder.parameters():
                param.requires_grad = False
            for param in model.phraseEncoder.parameters():
                param.requires_grad = False
        return model

    def _forward(self, model, batch):
        # if len(batch) == 4:
        # print(batch)
        audio_feats = batch[0]
        query_ids = batch[1]
        query_lens = batch[-1]
        audio_feats = audio_feats.float().to(self.device)
        query_ids = query_ids.long().to(self.device)
        dis = model(audio_feats, query_ids, query_lens)

        if isinstance(batch[2], torch.Tensor):
            labels = batch[2]
            labels = labels.float().to(self.device)
            return {"distance": dis, "targets": labels}
        else:
            return {"distance": dis}

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed

        outputdir = str(Path(conf["outputpath"]) / conf["model"] /
            "{}_{}".format(conf["audio_encoder"], conf["text_encoder"]) /
            "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%m"), uuid.uuid1().hex))

        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            "run",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(str(Path(outputdir) / "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(conf, logger.info)

        with open(conf["vocab_file"], "rb") as vocab_reader:
            vocabulary = pickle.load(vocab_reader)
        train_loader, val_loader = self._get_dataloaders(conf, vocabulary)
        model = self._get_model(conf, vocabulary)
        logger.info(
            summary(model, torch.randn(4, 501, 64), torch.empty(1, 4).random_(len(vocabulary)).long(),
                torch.tensor([4,]), show_input=True, show_hierarchical=True))
        model = model.to(self.device)

        optimizer = getattr(
            torch.optim, conf["optimizer"]
        )(model.parameters(), **conf["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        criterion = torch.nn.BCELoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        def _train_batch(engine, batch):
            model.train()
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            if conf["pretrained"]:
                model.apply(set_bn_eval)
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch)
                loss = criterion(output["distance"], output["targets"])
                if loss.requires_grad:
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
            # output: {"distance": xxx, "targets": xxx)
            y_pred = torch.round(output["distance"])
            return y_pred, output["targets"]

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1 = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            "loss": Loss(criterion, output_transform=lambda x: (x["distance"], x["targets"])),
            "Precision": Precision(thresholded_output_transform),
            "Recall": Recall(thresholded_output_transform),
            "Accuracy": Accuracy(thresholded_output_transform),
            "F1": f1
        }

        evaluator = Engine(_inference)

        for name, metric in metrics.items():
            if name == "loss":
                metric.attach(trainer, name)
            metric.attach(evaluator, name)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
            logger.info, ["loss"], metrics.keys())

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "loss", {
                "model": model.state_dict(),
                "config": conf,
        }, str(Path(outputdir) / "saved.pth"))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **conf["scheduler_args"])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
            scheduler, "loss")

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
                 pred_file="hard_predictions.txt",
                 event_file="event.txt",
                 segment_file="segment.txt",
                 time_ratio=10. / 500,
                 threshold=None,
                 window_size=None):
        from tqdm import tqdm

        from datasets.GroundingDataset import QueryAudioDatasetEval
        import utils.eval_util as eval_util
        import utils.metrics as metrics

        saved = torch.load(str(Path(experiment_path) / "saved.pth"), map_location="cpu")
        config = saved["config"]
        with open(config["vocab_file"], "rb") as vocab_reader:
            vocabulary = pickle.load(vocab_reader)
        model = self._get_model(config, vocabulary)
        model.load_state_dict(saved["model"], strict=False)

        label_df = pd.read_json(label_file)
        label_df = label_df[label_df["timestamps"].apply(lambda x: x[0][1] != 0)]
        dataset = QueryAudioDatasetEval(
            audio_feature, label_df, query_form="wordids", vocabulary=vocabulary)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn([0,1]),
            **config["dataloader_args"])

        model = model.to(self.device).eval()

        strong_data = []
        for idx, row in label_df.iterrows():
            for onset, offset in row["timestamps"]:
                strong_data.append({
                    "filename": "{}/{}".format(row["audiocap_id"], row["start_word"]),
                    "event_label": "fake_event",
                    "onset": onset,
                    "offset": offset
                })
        strong_label_df = pd.DataFrame(strong_data)

        time_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch", ascii=True, ncols=100):
                infos = batch[2]
                output = {}
                if Path(experiment_path).stem == "random":
                    output["distance"] = torch.as_tensor(np.random.rand(batch[0].size(0), batch[0].size(1)))
                else:
                    output = self._forward(model, batch)

                if threshold is None:
                    threshold = 0.5
                if window_size is None:
                    window_size = 1

                filtered_distance = eval_util.median_filter(
                    output["distance"].cpu(), window_size=window_size, threshold=threshold)
                for idx in range(len(infos)):
                    info = infos[idx]
                    change_indices = eval_util.find_contiguous_regions(filtered_distance[idx])
                    for row in change_indices:
                        time_predictions.append({
                            "filename": "{}/{}".format(info["audiocap_id"], info["start_word"]),
                            "event_label": "fake_event",
                            "onset": row[0],
                            "offset": row[1]
                        })

        assert len(time_predictions) > 0, "No outputs, lower threshold?"
        pred_df = pd.DataFrame(time_predictions)
        pred_df = eval_util.predictions_to_time(pred_df, ratio=time_ratio)
        pred_df.to_csv(str(Path(experiment_path) / pred_file), index=False, sep="\t")

        event_result, segment_result = metrics.compute_metrics(
            strong_label_df, pred_df, time_resolution=1.0)
        event_results_dict = event_result.results_class_wise_metrics()

        with open(str(Path(experiment_path) / event_file), "w") as f:
            f.write(event_result.__str__())

        with open(str(Path(experiment_path) / segment_file), "w") as f:
            f.write(segment_result.__str__())

    def predict(self,
                experiment_path,
                audio_feature,
                label_file,
                pred_file="hard_predictions.txt",
                time_ratio=10. / 500,
                window_size=None):
        from tqdm import tqdm

        from datasets.GroundingDataset import QueryAudioDatasetEval
        import utils.eval_util as eval_util
        import utils.metrics as metrics

        saved = torch.load(str(Path(experiment_path) / "saved.pth"), map_location="cpu")
        config = saved["config"]
        with open(config["vocab_file"], "rb") as vocab_reader:
            vocabulary = pickle.load(vocab_reader)
        model = self._get_model(config, vocabulary)
        model.load_state_dict(saved["model"], strict=False)

        label_df = pd.read_json(label_file)
        dataset = QueryAudioDatasetEval(
            audio_feature, label_df, query_form="wordids", vocabulary=vocabulary)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn([0,1]),
            **config["dataloader_args"])

        model = model.to(self.device).eval()

        strong_data = []
        for idx, row in label_df.iterrows():
            for onset, offset in row["timestamps"]:
                strong_data.append({
                    "filename": "{}/{}".format(row["audiocap_id"], row["start_word"]),
                    "event_label": "fake_event",
                    "onset": onset,
                    "offset": offset
                })
        strong_label_df = pd.DataFrame(strong_data)

        time_predictions = {th: [] for th in np.arange(0.1, 1.1, 0.1)}
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="batch", ascii=True, ncols=100):
                infos = batch[2]
                audio_feats = batch[0]
                output = {}
                if Path(experiment_path).stem == "random":
                    output["distance"] = torch.as_tensor(np.random.rand(batch[0].size(0), batch[0].size(1)))
                else:
                    output = self._forward(model, batch)

                if window_size is None:
                    window_size = 1

                for threshold in np.arange(0.1, 1.0, 0.1):
                    filtered_probs = eval_util.median_filter(
                        output["distance"].cpu(), window_size=window_size, threshold=threshold)
                    for idx in range(len(infos)):
                        info = infos[idx]
                        change_indices = eval_util.find_contiguous_regions(filtered_probs[idx])
                        for row in change_indices:
                            time_predictions[threshold].append({
                                "filename": "{}/{}".format(info["audiocap_id"], info["start_word"]),
                                "event_label": "fake_event",
                                "onset": row[0],
                                "offset": row[1]
                            })

        for threshold in np.arange(0.1, 1.0, 0.1):
            pred_df = pd.DataFrame(time_predictions[threshold])
            pred_df = eval_util.predictions_to_time(pred_df, ratio=time_ratio)
            pred_file_th = pred_file.split(".")[0] + "_TH_{:.1f}.".format(threshold) + pred_file.split(".")[-1]
            pred_df.to_csv(str(Path(experiment_path) / pred_file_th), index=False, sep="\t")

        # event_result, segment_result = metrics.compute_metrics(
        # strong_label_df, pred_df, time_resolution=1.0)
        # event_results_dict = event_result.results_class_wise_metrics()

        # with open(str(Path(experiment_path) / event_file), "w") as f:
        # f.write(event_result.__str__())

        # with open(str(Path(experiment_path) / segment_file), "w") as f:
        # f.write(segment_result.__str__())

if __name__ == "__main__":
    fire.Fire(Runner)
