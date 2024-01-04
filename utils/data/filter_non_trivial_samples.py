import json
import pandas as pd
import argparse


def main(args):
    data = json.load(open(args.data))
    duration = pd.read_csv(args.duration, sep="\t")
    aid_to_dur = dict(zip(duration["audio_id"], duration["duration"]))

    flt_data = []
    for item in data:
        new_item = item.copy()
        dur = aid_to_dur[item["audio_id"]]
        del new_item["phrases"]
        new_phrases = []
        for phrase_item in item["phrases"]:
            segments = phrase_item["segments"]
            
            inter_sum = 0
            for segment in segments:
                inter = min(segment[1], dur) - max(segment[0], 0)
                inter_sum += inter

            if args.sum_inter:
                if inter_sum / dur >= args.percent / 100:
                    continue
                else:
                    new_phrases.append(phrase_item)
            else:
                if len(segments) == 1 and inter_sum / dur >= args.percent / 100:
                    continue
                else:
                    new_phrases.append(phrase_item)
        if len(new_phrases) > 0:
            new_item["phrases"] = new_phrases
            flt_data.append(new_item)
    json.dump(flt_data, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--duration", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--percent", type=float, default=99)
    parser.add_argument("--sum_inter", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
