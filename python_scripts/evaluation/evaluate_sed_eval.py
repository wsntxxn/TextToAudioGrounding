from pathlib import Path
import json
import fire
import pandas as pd

import utils.eval_util as eval_util


def evaluate(ground_truth,
             prediction,
             event_output,
             segment_output,
             t_collar=0.200,
             time_resolution=1.0):
    gt_data = json.load(open(ground_truth))
    gt = []
    for audio_item in gt_data:
        audiocap_id = audio_item["audiocap_id"]
        for phrase_item in audio_item["phrases"]:
            start_index = phrase_item["start_index"]
            fname = f"{audiocap_id}_{start_index}"
            for onset, offset in phrase_item["segments"]:
                if onset == 0 and offset == 0:
                    continue
                gt.append({
                    "filename": fname,
                    "event_label": "fake_event",
                    "onset": onset,
                    "offset": offset
                })
    gt = pd.DataFrame(gt)
    pred = pd.read_csv(prediction, sep="\t")

    event_result, segment_result = eval_util.compute_sed_eval(
        gt, pred, t_collar, time_resolution
    )

    Path(event_output).parent.mkdir(parents=True, exist_ok=True)
    Path(segment_output).parent.mkdir(parents=True, exist_ok=True)
    with open(event_output, "w") as wp:
        wp.write(event_result.__str__())
    with open(segment_output, "w") as wp:
        wp.write(segment_result.__str__())


if __name__ == "__main__":
    fire.Fire(evaluate)
