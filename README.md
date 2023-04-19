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
  - start_index: index of the first token of `phrase`, starting from 0 (the index of the first token in `tokens` is 0)
  - end_index: index of the last token of `phrase`
  - segments: a list of [`onset`, `offset`]  timestamp annotations

## TAG baseline

We provide a baseline approach for TAG in this repository. To run the baseline, complete the following steps:
1. checkout the code and install the required python packages:
```bash
git clone https://github.com/wsntxxn/TextToAudioGrounding
pip install -r requirements.txt
```
2. download audio clips and labels from Zenodo.
3. pack waveforms, assume the audio files are in $AUDIO:
```bash
cd data
for split in train val test; do
  python prepare_wav_csv.py $AUDIO/$split $split/wav.csv
  python pack_waveform.py $split/wav.csv -o $split/waveform.h5 --sample_rate 32000
done
python prepare_duration.py test/wav.csv test/duration.csv
cd ..
```
4. prepare vocabulary file:
```bash
python utils/build_vocab.py data/train/label.json data/train/vocab.pkl
```
5. modify the configuration file `configs/strongly_supervised/biencoder/cdur_w2vmean.yaml` (if needed) and run the training and evaluation:
```bash
python run.py train_evaluate configs/strongly_supervised/biencoder/cdur_w2vmean.yaml configs/strongly_supervised/biencoder/eval.yaml
```
Experiment directory is returned after training and evaluation. Results are also stored in this directory.
