# Text-to-Audio Grounding

This repository provides the data and source code for Text-to-Audio Grounding (TAG) task.

## Dataset

### Audio

The dataset in this repository is an augmented audio captioning dataset. It is based on [Audiocaps](https://www.aclweb.org/anthology/N19-1011.pdf), which is established using part of a audio event dataset, [AudioSet](https://research.google.com/audioset). Therefore, audio files can be downloaded from [AudioSet](https://research.google.com/audioset/download.html). 

### Label

The labels for TAG are provided in `data/*.json`. The json file can be read by `pandas`. Each file includes these columns:
* audiocap_id: `id` used in the [Audiocaps](https://github.com/cdjkim/audiocaps/tree/master/dataset) label file
* filename: raw wave filename, in the form of `[youtube_id]_[start_time]_[end_time].wav`, you can use this to grab the file from Youtube
* caption: audio caption provided by Audiocaps
* tokens: tokenized audio caption
* soundtag: phrase representing sound which is extracted from `caption`
* start_word: the index of the first word of `soundtag`, starting from 0 (the index of the first token in `tokens` is 0)
* timestamps: a list of starting and ending timestamps of the corresponding `soundtag`; for example, if the `soundtag` is "a dog is barking" and `timestamps` is [[1, 3], [7, 8]], it means the dog barks twice in the audio: the first bark continues from 1s to 3s and the second from 7s to 8s

## TAG baseline

We provide a baseline approach for TAG in this repository. To run the baseline, complete the following steps:
1. download the raw audio files and rename them in order that your computer can find the audio files via `filename` in `data/*.json` (this also requires a little processing).
2. checkout the code and install the required python packages:
```bash
git clone https://github.com/wsntxxn/TextToAudioGrounding
pip install -r requirements.txt
```
3. extract audio feature using `utils/extract_feature.py`.
4. modify the configuration file `config/conf.yaml` and run the training command:
```bash
python run.py train config/conf.yaml
```
5. evaluate the trained model:
```bash
export EXP_PATH=***
export AUDIO_FEATURE=***
python run.py evaluate $EXP_PATH $AUDIO_FEATURE data/test.json data/test_meta.csv
```
where `EXP_PATH` and `AUDIO_FEATURE` are the experiment checkpoint directory and audio feature file corresponding to your configuration.
