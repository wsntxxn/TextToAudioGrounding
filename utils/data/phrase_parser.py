import re
from pathlib import Path
import json
from typing import Union
import pandas as pd
import fire
import nltk
from tqdm import tqdm


class PhraseParser:
    def __init__(self) -> None:
        self.conj_preps = [
            "and then", "is followed by", "follow by", "followed by",
            "are followed by", "accompanied by", "is accompanied by",
            "are accompanied by", "interrupted by", "is interrupted by",
            "are interrupted by", "meanwhile", "all the while", "before which",
            "after which", "during which time", "while", "which", "as well as",
            "during", "afterward", "afterwards", "before and after",
            "proceeded by", "before", "after", "though", "although",
            "despite that", "simultaneously with", "then", "along with",
            "alongside", "following by", "following", "when", "punctuated by",
            "overlapped by"
        ]
        pattern = "|".join([",? " + x + " " for x in self.conj_preps]) + "|" + \
            "|".join(["^" + x + " " for x in self.conj_preps]) + \
            "|,? as well" + \
            "|,? and (?!forth|down|backward|over|out|off|more|\w+er)" + \
            "|,?(?<!play)(?<!playing) with (?!one\sanother|each\sother)" + \
            "|,? ?(?<!w) as (?!a\sresult)" + \
            "|, |; "
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.ignore = ["is", "are", "and"]
        er_words = [
            "another", "there", "thunder", "water", "other", "emergency",
            "several", "chatter", "clatter", "person", "artillery", "camera",
            "ceramic", "cheer", "computer", "convers", "decelerat",
            "accelerat", "laughter", "helicopter", "paper", "propeller",
            "silver", "rooster", "whimper", "drawer", "everyone", "flutter",
            "hammer", "holler", "laser", "later", "member", "mother", "father",
            "operate", "passenger", "patter", "peeper", "percussion",
            "persistent", "photographer", "power", "river", "rubber",
            "sneaker", "starter", "spatter", "splatter", "sputter", "toddler",
            "twitter", "typewriter", "verbaliz", "very", "whisper", "wiper",
            "wrapper"
        ]
        self.er_pattern = re.compile(
            ",? ?and (?=" + "|".join(er_words) + ")", re.IGNORECASE
        )

    def split(self, pattern, sentence):
        results = []
        for x in pattern.split(sentence):
            x = x.rstrip().strip().lower()
            if x in self.ignore:
                continue
            if len(x) > 0:
                results.append(x)
        return results

    def __call__(self, sentence):
        phrases = []
        for phrase in self.split(self.pattern, sentence):
            cand_phrases = self.split(self.er_pattern, phrase)
            if len(cand_phrases) > 1:
                for cand_phrase in cand_phrases:
                    phrases.append(cand_phrase)
            elif cand_phrases[0] != phrase:
                phrases.append(cand_phrases[0])
            else:
                phrases.append(phrase)
        return phrases


class Extractor:
    def extract_ac_with_old_anno(self, caption_file, old_label_file, output):
        parser = PhraseParser()
        df = pd.read_csv(caption_file)
        old_label = pd.read_json(old_label_file)
        acid_to_tokens = dict(
            zip(old_label["audiocap_id"], old_label["tokens"])
        )
        data = []
        for idx, row in df.iterrows():
            audiocap_id = row["audiocap_id"]
            tokens = acid_to_tokens[audiocap_id]
            item = {
                "audiocap_id": audiocap_id,
                "audio_id": "Y" + row["youtube_id"] + ".wav",
                "tokens": tokens,
                "phrases": []
            }
            phrases = parser(tokens)
            for phrase in phrases:
                if tokens.count(phrase) > 1:
                    if len(phrase.split()
                          ) == 1 and tokens.split().count(phrase) == 1:
                        start_index = tokens.split().index(phrase)
                        end_index = start_index
                    else:
                        print(f"audiocap_id: {audiocap_id}, caption: {tokens}")
                        start_index = 0
                        end_index = 0
                else:
                    start_index = tokens.index(phrase)
                    start_index = len(tokens[:start_index].split())
                    end_index = start_index + len(phrase.split()) - 1
                phrase_item = {
                    "phrase": phrase,
                    "start_index": start_index,
                    "end_index": end_index,
                    "segments": []
                }
                item["phrases"].append(phrase_item)
            data.append(item)
        json.dump(data, open(output, "w"), indent=4)

    def extract_audiocaps(self, caption_file, output):
        parser = PhraseParser()
        caption_data = json.load(open(caption_file))
        data = []
        punctuation = ".()"
        missing = []
        for audio_item in tqdm(caption_data["audios"]):
            audio_id = "Y" + audio_item["audio_id"] + ".wav"
            for cap_item in audio_item["captions"]:
                audiocap_id = cap_item["audiocap_id"]
                item = {
                    "audiocap_id": audiocap_id,
                    "audio_id": audio_id,
                    "phrases": []
                }
                caption = cap_item["caption"]
                caption = re.sub(
                    "[{}]".format(punctuation), "", caption.lower()
                )
                tokens = " ".join(nltk.word_tokenize(caption))
                phrases = parser(tokens)
                item["tokens"] = tokens
                for phrase in phrases:
                    if tokens.count(phrase) > 1:
                        if len(phrase.split()
                              ) == 1 and tokens.split().count(phrase) == 1:
                            start_index = tokens.split().index(phrase)
                            end_index = start_index
                        else:
                            # print(f"audiocap_id: {audiocap_id}, caption: {tokens}")
                            missing.append({
                                "audiocap_id": audiocap_id,
                                "caption": tokens,
                                "phrase": phrase
                            })
                            start_index = -1
                            end_index = -1
                    else:
                        start_index = tokens.index(phrase)
                        start_index = len(tokens[:start_index].split())
                        end_index = start_index + len(phrase.split()) - 1
                    phrase_item = {
                        "phrase": phrase,
                        "start_index": start_index,
                        "end_index": end_index,
                        # "segments": []
                    }
                    item["phrases"].append(phrase_item)
                data.append(item)

        json.dump(data, open(output, "w"), indent=4)
        pd.DataFrame(missing).to_csv(
            Path(output).with_name("missing.csv"), sep="\t", index=False
        )

    def extract_from_caption(self, caption_file, output):
        parser = PhraseParser()
        caption_data = json.load(open(caption_file))
        data = []
        punctuation = ".()"
        missing = []
        for audio_item in tqdm(caption_data["audios"]):
            audio_id = audio_item["audio_id"]
            for cap_item in audio_item["captions"]:
                cap_id = cap_item["cap_id"]
                audiocap_id = f"{audio_id}_{cap_id}"
                item = {
                    "audio_id": audio_id,
                    "audiocap_id": audiocap_id,
                    "phrases": []
                }
                caption = cap_item["caption"]
                caption = re.sub(
                    "[{}]".format(punctuation), "", caption.lower()
                )
                tokens = " ".join(nltk.word_tokenize(caption))
                phrases = parser(tokens)
                item["tokens"] = tokens
                for phrase in phrases:
                    if tokens.count(phrase) > 1:
                        if len(phrase.split()
                              ) == 1 and tokens.split().count(phrase) == 1:
                            start_index = tokens.split().index(phrase)
                            end_index = start_index
                        else:
                            # print(f"audiocap_id: {audiocap_id}, caption: {tokens}")
                            missing.append({
                                "audio_id": audio_id,
                                "caption": tokens,
                                "phrase": phrase
                            })
                            start_index = -1
                            end_index = -1
                    else:
                        start_index = tokens.index(phrase)
                        start_index = len(tokens[:start_index].split())
                        end_index = start_index + len(phrase.split()) - 1
                    phrase_item = {
                        "phrase": phrase,
                        "start_index": start_index,
                        "end_index": end_index,
                        # "segments": []
                    }
                    item["phrases"].append(phrase_item)
                data.append(item)

        output = Path(output)
        json.dump(data, open(output, "w"), indent=4)
        pd.DataFrame(missing).to_csv(
            output.with_name(output.stem + "_missing.csv"),
            sep="\t",
            index=False
        )

    def prepare_wavcaps_data(
        self,
        json_file: str,
        duration_csv: str,
        min_duration: float,
        max_duration: float,
        output_file: str,
        id_prefix: str = "",
        audiogrounding_test_json: Union[str, None] = None,
    ):
        duration_df = pd.read_csv(duration_csv, sep="\t")
        print(f"{len(duration_df)} audio clips before duration filtering")
        duration_df = duration_df[duration_df["duration"].between(
            min_duration, max_duration
        )]
        print(f"{len(duration_df)} audio clips left after duration filtering")
        audio_ids = set(duration_df["audio_id"].values)

        if audiogrounding_test_json:
            test_data = json.load(open(audiogrounding_test_json))
            test_audio_ids = set([x["audio_id"] for x in test_data])

        data = json.load(open(json_file))
        result = []
        parser = PhraseParser()
        for item in tqdm(data["data"]):
            orig_caption = item["caption"]
            if item["id"].endswith(".wav"):  # ASSL
                assert audiogrounding_test_json is not None
                if item["id"] in test_audio_ids:
                    continue
                audio_id = Path(item["id"]).with_suffix(".flac").__str__()
            else:
                audio_id = f"{item['id']}.flac"

            if audio_id not in audio_ids:
                continue

            caption = re.sub("[{}]".format(".()"), "", orig_caption.lower())
            tokens = " ".join(nltk.word_tokenize(caption))
            phrases = parser(tokens)
            result.append({
                "audio_id":
                    f"{id_prefix}_{audio_id}" if id_prefix else audio_id,
                "caption":
                    orig_caption,
                "phrases":
                    phrases,
            })
        json.dump(result, open(output_file, "w"), indent=2)

    def prepare_audiocaps_v2(
        self,
        audiocaps_v2_dir: str,
        audioset_wav_csv: str,
        audiogrounding_test_json: str,
        output_file: str,
    ):
        test_data = json.load(open(audiogrounding_test_json))
        test_audio_ids = set([x["audio_id"] for x in test_data])

        result = []
        parser = PhraseParser()
        audiocaps_v2_dir = Path(audiocaps_v2_dir)
        audioset_df = pd.read_csv(audioset_wav_csv, sep="\t")
        available_audio_ids = set(audioset_df["audio_id"].values)
        for split in ["train", "val", "test"]:
            data_df = pd.read_csv(audiocaps_v2_dir / f"{split}.csv")
            with tqdm(
                desc=f"Processing {split} set", total=len(data_df)
            ) as pbar:
                for _, row in data_df.iterrows():
                    audio_id = f"Y{row['youtube_id']}.wav"
                    skip = False
                    if audio_id in test_audio_ids:
                        skip = True
                    if audio_id not in available_audio_ids:
                        skip = True
                    if skip:
                        pbar.update(1)
                        continue
                    orig_caption = row["caption"]
                    if pd.isna(row["caption"]):
                        continue
                    caption = re.sub(
                        "[{}]".format(".()"), "", orig_caption.lower()
                    )
                    tokens = " ".join(nltk.word_tokenize(caption))
                    phrases = parser(tokens)
                    result.append({
                        "audio_id": audio_id,
                        "caption": orig_caption,
                        "tokens": tokens,
                        "phrases": phrases,
                    })
                    pbar.update(1)
        json.dump(result, open(output_file, "w"), indent=2)

    def test_phrase_parser(self):
        parser = PhraseParser()
        sentence = "the interior of a car, as a manual sunroof being wound shut"
        print(parser(sentence))


if __name__ == "__main__":
    fire.Fire(Extractor)
