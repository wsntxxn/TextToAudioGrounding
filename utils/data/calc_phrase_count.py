import json
import argparse


def main(args):
    with open(args.input, "r") as f:
        data = json.load(f)

    phrase_count = {}
    for item in data:
        for phrase_item in item["phrases"]:
            phrase = phrase_item["phrase"]
            if phrase not in phrase_count:
                phrase_count[phrase] = 0
            phrase_count[phrase] += 1

    with open(args.output, "w") as f:
        json.dump(phrase_count, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args)
