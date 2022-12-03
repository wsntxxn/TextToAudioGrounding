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
2. download audio clips and labels from [google drive](https://drive.google.com/file/d/1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs/view?usp=sharing).
```bash
unzip AudioTextGrounding.zip -d data
```
3. extract audio feature:
```bash
cd data
for split in train val test; do
  python prepare_wav_csv.py $split/audio $split/wav.csv
  python prepare_audio_feature.py $split/wav.csv $split/lms.h5 $split/lms.csv lms -n_mels 64 -win_length 640 -hop_length 320
done
cd ..
```
4. prepare vocabulary file:
```bash
python utils/build_vocab.py data/train/label.json data/train/vocab.pkl
```
5. modify the configuration file `config/conf.yaml` (if needed) and run the training and evaluation:
```bash
python run.py train_evaluate config/conf.yaml data/test/lms.csv data/test/label.json data/test/meta.csv
```
Experiment directory is returned after training and evaluation. Results are also stored in this directory.
