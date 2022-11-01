from pathlib import Path
import sklearn.preprocessing as pre
import scipy
import numpy as np
import pandas as pd
import sed_eval
from psds_eval import PSDSEval, plot_psd_roc
from psds_eval.psds import WORLD


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def median_filter(x, window_size, threshold=0.5):
    x = pre.binarize(x, threshold=threshold)
    size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def predictions_to_time(df, ratio):
    if len(df) == 0:
        return df
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df


def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def connect_clusters_(x, n=1):
    """connect_clusters_
    Connects clustered predictions (0,1) in x with range n

    :param x: Input array. zero-one format
    :param n: Number of frames to skip until connection can be made
    """
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


class PSDSEval_Grounding(PSDSEval):

    def _get_dataset_duration(self):
        """Compute duraion of on the source data.

        Compute the duration per class, and total duration for false
        positives."""
        t_filter = self.ground_truth.event_label == WORLD
        if not hasattr(self, "data_duration"):
            gt = self.ground_truth[t_filter].copy()
            gt["audio_id"] = gt["filename"].apply(lambda x: x.split("_")[0]).values
            data_duration = gt.groupby("audio_id")["duration"].apply(lambda x: list(x)[0]).sum()
            self.data_duration = data_duration
        gt_durations = self.ground_truth.groupby("event_label").duration.sum()
        return gt_durations, self.data_duration


def compute_psds(prediction_dfs,
                 ground_truth,
                 duration,
                 dtc_threshold=0.5,
                 gtc_threshold=0.5,
                 # cttc_threshold=0.0,
                 # alpha_ct=0,
                 # alpha_st=0,
                 max_efpr=400,
                 save_dir=None):
    if not isinstance(ground_truth, pd.DataFrame):
        ground_truth = pd.read_csv(ground_truth, sep="\t")
    if not isinstance(duration, pd.DataFrame):
        duration = pd.read_csv(duration, sep="\t")

    duration = duration[duration["filename"].isin(ground_truth["filename"].unique())]
    try:
        assert set(ground_truth["filename"].values) == set(duration["filename"].values)
    except AssertionError:
        import ipdb; ipdb.set_trace()

    psds_eval = PSDSEval_Grounding(
        ground_truth=ground_truth,
        metadata=duration,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        # cttc_threshold=cttc_threshold,
        cttc_threshold=0.0,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        # det["index"] = range(1, len(det) + 1)
        # det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=0,
                                # alpha_ct=alpha_ct,
                                alpha_st=0,
                                # alpha_st=alpha_st,
                                max_efpr=max_efpr,)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # pred_dir = save_dir / f"predictions_dtc{dtc_threshold}_" \
                              # f"gtc{gtc_threshold}_cttc{cttc_threshold}"

        # pred_dir.mkdir(exist_ok=True)
        # for k in prediction_dfs.keys():
            # prediction_dfs[k].to_csv(
                # pred_dir / f"predictions_th_{k:.2f}.tsv",
                # sep="\t",
                # index=False,
            # )

        plot_psd_roc(
            psds_score,
            filename=save_dir / f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_maxefpr{max_efpr}.png",
        )

    return psds_score.value


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file


def event_based_evaluation_df(reference,
                              estimated,
                              t_collar=0.200,
                              percentage_of_length=0.2):
    """
    Calculate EventBasedMetric given a reference and estimated dataframe
    :param reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    reference events
    :param estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    estimated events to be compared with reference
    :return: sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score')

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.):
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution)

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file)

    return segment_based_metric


def compute_sed_eval(valid_df, pred_df, t_collar=0.2, time_resolution=1.):
    metric_event = event_based_evaluation_df(valid_df,
                                             pred_df,
                                             t_collar=t_collar,
                                             percentage_of_length=0.2)
    metric_segment = segment_based_evaluation_df(
        valid_df, pred_df, time_resolution=time_resolution)
    return metric_event, metric_segment


def compute_intersection_based_threshold_auc(
        scores,
        ground_truth,
        dtc_threshold,
        gtc_threshold,
        time_decimals=6,
        num_jobs=4
    ):
    from sed_scores_eval.intersection_based.intermediate_statistics import intermediate_statistics
    from sed_scores_eval.base_modules.precision_recall import (
        fscore_curve_from_intermediate_statistics
    )
    from sed_scores_eval.utils.auc import staircase_auc
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=None,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    f_curve, p_curve, r_curve, scores_curve, stats_curve = fscore_curve_from_intermediate_statistics(
        intermediate_stats
    )
    f_max = f_curve["fake_event"].max()
    score = staircase_auc(f_curve["fake_event"][:-1], scores_curve["fake_event"][:-1])
    return {
        "score": score,
        "f_max": f_max,
        "stats": {
            "f_curve": f_curve,
            "p_curve": p_curve,
            "r_curve": r_curve,
            "scores_curve": scores_curve,
            "stats_curve": stats_curve
        }
    }
