# Text-to-Audio Grounding

This repository provides the data and source code for Text-to-Audio Grounding (TAG) task.

## Data

The *AudioGrounding* dataset is an augmented audio captioning dataset. It is based on [AudioCaps](https://www.aclweb.org/anthology/N19-1011.pdf), which is established using part of a audio event dataset, [AudioSet](https://research.google.com/audioset). Therefore, audio files can be downloaded from [AudioSet](https://research.google.com/audioset/download.html). 

The updated *AudioGrounding* v2 is available in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7269161.svg)](https://doi.org/10.5281/zenodo.7269161).
Changes in version 2:
1. Train/val/test sets are re-split and refined.
2. Data are re-formatted and audio files are renamed.

The current label format: a list of `audio_item`, each containing
- audiocap_id: `id` in the [AudioCaps](https://github.com/cdjkim/audiocaps/tree/master/dataset) label file
- audio_id: audio filename, in the form of `Y[youtube_id].wav`
- tokens: caption tokenized by NLTK
- phrases: a list of `phrase_item`
  - phrase: tokens of the current query phrase
  - start_index: index of the first token of `phrase`, starting from 0
  - end_index: index of the last token of `phrase`
  - segments: a list of `[onset, offset]`  timestamp annotations

## TAG Baseline

We provide a baseline approach for TAG in this repository. To run the baseline: 
1. checkout the code and install the required python packages:
```bash
git clone https://github.com/wsntxxn/TextToAudioGrounding
pip install -r requirements.txt
```
2. download audio clips and labels from Zenodo.
3. pack waveforms, assume the audio files are in `$AUDIO`:
```bash
mkdir data/audiogrounding
for split in train val test; do
  python utils/data/prepare_wav_csv.py $AUDIO/$split data/audiogrounding/$split/wav.csv
  python utils/data/pack_waveform.py data/audiogrounding/$split/wav.csv \
      -o data/audiogrounding/$split/waveform.h5 \
      --sample_rate 32000
done
python utils/data/prepare_duration.py data/audiogrounding/test/wav.csv data/audiogrounding/test/duration.csv
```
4. prepare vocabulary file:
```bash
python utils/build_vocab.py data/audiogrounding/train/label.json data/audiogrounding/train/vocab.pkl
```
5. run the training and evaluation:
```bash
python python_scripts/training/run_strong.py train_evaluate \
    --train_config $TRAIN_CFG \
    --eval_config $EVAL_CFG
```
Or alternatively,
```bash
python python_scripts/training/run_strong.py train \
    --config $TRAIN_CFG
python python_scripts/training/run_strong.py evaluate \
    --experiment_path $EXP_PATH \
    --eval_config $EVAL_CFG
```
`$TRAIN_CFG` and `$EVAL_CFG` are yaml-formatted configuration files.
`$EXP_PATH` is the checkpoint directory set in `$TRAIN_CFG`.
Example configuration files are provided [here](eg_configs/strongly_supervised/audiogrounding/biencoder).

## Weakly-Supervised Text-to-Audio Grounding (WSTAG)

### Inference

We provide the best-performing WSTAG model, downloaded [here](https://drive.google.com/file/d/1xDQT_KQ6l9Hzcn4QkO1G3XBJmdw1LCVe/view?usp=drive_link). Unzip it into `$MODEL_DIR`:
```bash
unzip audiocaps_cnn8rnn_w2vmean_dp_ls_clustering_selfsup.zip -d $MODEL_DIR
```
Remember to modify the training data vocabulary path in `$MODEL_DIR/config.yaml` (*data.train.collate_fn.tokenizer.args.vocabulary*) to `$MODEL_DIR/vocab.pkl`.
To ensure that the vocabulary file used for training is loaded for inference, the inference script uses the vocabulary path specified in `$MODEL_DIR/config.yaml`.


Inference:
```bash
python python_scripts/inference/inference.py inference_multi_text_model \
    --experiment_path $MODEL_DIR \
    --audio $AUDIO \
    --phrase $PHRASE \
    --output ./prob.png
```

### Training

For all settings, training is done in the same way as in the baseline:
```bash
python $TRAIN_SCRIPT train_evaluate \
    --train_config $TRAIN_CFG \
    --eval_config $EVAL_CFG
```
The training scripts and configurations vary for different settings.
We provide the training script and example configuration file in each setting.

#### Data Format
WSTAG uses audio captioning data for training.
The format of training data is the same as *AudioGrounding*, with the only difference that there is no `segments` in `phrase_item`.
You can convert the original captioning data into this format by yourself.
The phrase parsing rules are provided [here](utils/data/phrase_parser.py).
The waveform packing and vocabulary preparation process is also the same as in the baseline.

#### Sentence-level WSTAG
* `TRAIN_SCRIPT`: [run_weak_sentence.py](python_scripts/training/run_weak_sentence.py)
* `TRAIN_CFG`: [cnn8rnn_w2vmean_dp_amean_tmean.yaml](eg_configs/weakly_supervised/audiocaps/sentence_level/phrase_wise/cnn8rnn_w2vmean_dp_amean_tmean.yaml)
* `EVAL_CFG`: [eval.yaml](eg_configs/weakly_supervised/audiocaps/sentence_level/phrase_wise/eval.yaml)

#### Phrase-level WSTAG

For all phrase-level settings, the example `EVAL_CFG` is [eval.yaml](eg_configs/weakly_supervised/audiocaps/phrase_level/eval.yaml).
For all phrase-level settings except "X + self-supervision", `TRAIN_SCRIPT` is [run_weak_phrase.py](python_scripts/training/run_weak_phrase.py).

##### random sampling

* `TRAIN_CFG`: [cnn8rnn_w2vmean_random.yaml](eg_configs/weakly_supervised/audiocaps/phrase_level/cnn8rnn_w2vmean_random.yaml)

##### similarity-based sampling

* `TRAIN_CFG`: [cnn8rnn_w2vmean_similarity.yaml](eg_configs/weakly_supervised/audiocaps/phrase_level/cnn8rnn_w2vmean_similarity.yaml)

Similarity-based sampling requires pre-computed phrase embeddings.
We use the contrastive audio-text model trained on AudioCaps to extract phrase embeddings.
Download the model from [here](https://drive.google.com/file/d/13Iz0AOcbXc8JAmELLGMoD0BI1BadxMV4/view?usp=drive_link), unzip it into `$CLAP_DIR`, then extract embeddings:
```bash
unzip audiocaps_cnn14_bertm.zip -d $CLAP_DIR
python utils/data/create_text_embedding/prepare_phrase_clap.py phrase \
    --experiment_path $CLAP_DIR \
    --phrase_input $DATA \
    --output $OUTPUT \
    --with_proj True
```
Then modify the *data.train.dataset.args.phrase_embed* item in the training configuration file to `$OUTPUT` accordingly.

##### clustering-based sampling

* `TRAIN_CFG`: [cnn8rnn_w2vmean_clustering.yaml](eg_configs/weakly_supervised/audiocaps/phrase_level/cnn8rnn_w2vmean_clustering.yaml)

Clustering-based sampling requires clustering models.
We train clustering models based on the pre-computed phrase embeddings.
```bash
python python_scripts/clustering/kmeans_emb.py \
    --embedding $PHRASE_EMB \
    --n_cluster $N_CLUSTER \
    --output $OUTPUT
```
`$PHRASE_EMB` is the phrase embedding file, i.e., `$OUTPUT` of the previous step.
Remember to modify the *data.train.dataset.args.cluster_map* to the corresponding mapping file (not exactly `$OUTPUT`).

##### X (any sampling) + self-supervision

* `TRAIN_SCRIPT`: [run_weak_phrase_self_supervision.py](python_scripts/training/run_weak_phrase_self_supervision.py)
* `TRAIN_CFG`: [cnn8rnn_w2vmean_clustering_selfsup.yaml](eg_configs/weakly_supervised/audiocaps/phrase_level/cnn8rnn_w2vmean_clustering_selfsup.yaml)

The *teacher.pretrained* should be set to the checkpoint path of the pretrained WSTAG model.
