import argparse
import json
from pathlib import Path


# exclude data with audio_ids overlapped with test files
def main(args):
    data = json.load(open(args.data))
    test_data = []
    for data_i in args.test_data:
        test_data.extend(json.load(open(data_i)))
    test_aids = [item["audio_id"] for item in test_data]

    fil_data = []
    for item in data:
        if item["audio_id"] not in test_aids:
            fil_data.append(item)
    
    p_dir = Path(args.output).parent
    p_dir.mkdir(exist_ok=True, parents=True)
    print(f"{len(fil_data)} data left after filtering")
    json.dump(fil_data, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True, nargs="+")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    main(args)
