import argparse
from pathlib import Path
import json
import csv


def main(args):
    data = json.load(open(args.label))

    aid_to_fname = {}
    with open(args.audioset_wav_csv, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        header = next(reader)
        for row in reader:
            aid = row[0]
            fname = row[1]
            aid_to_fname[aid] = fname

    output_dir = Path(args.output_dir)

    for audio_item in data:
        aid = audio_item["audio_id"]
        if aid in aid_to_fname:
            if not (output_dir / aid).exists():
                (output_dir / aid).symlink_to(aid_to_fname[aid])
        else:
            assert (output_dir / aid).exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--audioset_wav_csv", type=str)
    args = parser.parse_args()
    main(args)
