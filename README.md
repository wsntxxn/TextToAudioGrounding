# Text-to-Audio Grounding

This repository provides the data and source code for Text-to-Audio Grounding (TAG) task.

## Dataset

The *AudioGrounding* dataset is an augmented audio captioning dataset. It is based on [AudioCaps](https://www.aclweb.org/anthology/N19-1011.pdf), which is established using part of a audio event dataset, [AudioSet](https://research.google.com/audioset). Therefore, audio files can be downloaded from [AudioSet](https://research.google.com/audioset/download.html). 

For convenience, we provide the raw audio files and label files together in [google drive](https://drive.google.com/file/d/1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs/view?usp=sharing).
The labels for TAG are provided in `{train,val,test}/label.json`. Each file contains an item list, with each item containing:
* audiocap_id: `id` in the [AudioCaps](https://github.com/cdjkim/audiocaps/tree/master/dataset) label file
* filename: raw wave filename, in the form of `[youtube_id]_[start_time]_[end_time].wav`, you can use this to grab the file from Youtube
* tokens: caption provided by AudioCaps, converted to lower case
* phrase: phrase representing sound, extracted from `tokens`
* start_word: the index of the first word of `phrase`, starting from 0 (the index of the first token in `tokens` is 0)
* timestamps: a list of onsets and offsets of the corresponding `phrase`; for example, if the `phrase` is "a dog is barking" and `timestamps` is [[1, 3], [7, 8]], it means the dog barks twice in the audio: the first bark continues from 1s to 3s and the second from 7s to 8s

### AudioGrounding v2
The updated *AudioGrounding* v2 is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7269161.svg)](https://doi.org/10.5281/zenodo.7269161), where train/val/test sets are re-split and validation and test sets are refined. Besides, annotations are re-formatted.

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
