import argparse
from pathlib import Path
import random
import json


def main(args):
    
    random.seed(args.seed)

    with open(args.input, "r") as f:
        data = json.load(f)

    if "segments" in data[0]["phrases"][0]:
        target_type = "strong"
    else:
        target_type = "weak"

    if target_type == "weak":
        num_sample = int(len(data) * args.frac)
        random.shuffle(data)
        data = data[:num_sample]
    elif target_type == "strong":
        raise Exception

    if not Path(args.output).parent.exists():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    main(args)
