from pathlib import Path
from warnings import warn
import hashlib
from collections import namedtuple
import sklearn.preprocessing as pre
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sed_eval
from sklearn.metrics import auc
from psds_eval import PSDSEval, plot_psd_roc
from psds_eval.psds import WORLD, PSDSEvalError
from sed_scores_eval import intersection_based


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


def binarize(x, threshold=0.5):
    if x.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in x])
    else:
        return pre.binarize(x, threshold=threshold)


def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3: # (batch_size, time_steps, num_classes)
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1: # (batch_size, time_steps)
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1: # (time_steps, num_classes)
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


# class PSDSEval_Grounding(PSDSEval):

    # def _get_dataset_duration(self):
        # """Compute duraion of on the source data.

        # Compute the duration per class, and total duration for false
        # positives."""
        # t_filter = self.ground_truth.event_label == WORLD
        # if not hasattr(self, "data_duration"):
            # gt = self.ground_truth[t_filter].copy()
            # gt["audio_id"] = gt["filename"].apply(lambda x: x.split("_")[0]).values
            # data_duration = gt.groupby("audio_id")["duration"].apply(lambda x: list(x)[0]).sum()
            # self.data_duration = data_duration
        # gt_durations = self.ground_truth.groupby("event_label").duration.sum()
        # return gt_durations, self.data_duration


def compute_psds(prediction_dfs,
                 ground_truth,
                 duration,
                 dtc_threshold=0.5,
                 gtc_threshold=0.5,
                 # cttc_threshold=0.0,
                 # alpha_ct=0,
                 # alpha_st=0,
                 max_efpr=None,
                 save_dir=None):

    if not isinstance(ground_truth, pd.DataFrame):
        ground_truth = pd.read_csv(ground_truth, sep="\t")
    if not isinstance(duration, pd.DataFrame):
        duration = pd.read_csv(duration, sep="\t")

    # duration = duration[duration["filename"].isin(ground_truth["filename"].unique())]

    aid_to_dur = dict(zip(duration["audio_id"], duration["duration"]))

    metadata = []
    for _, row in ground_truth.iterrows():
        dataid = row["filename"]
        aid = row["audio_id"]
        metadata.append({
            "filename": dataid,
            "duration": aid_to_dur[aid],
        })
    duration = pd.DataFrame(metadata)

    if "audio_id" in ground_truth:
        ground_truth = ground_truth.drop("audio_id", axis=1)

    try:
        assert set(ground_truth["filename"].values) == set(duration["filename"].values)
    except AssertionError:
        import ipdb; ipdb.set_trace()

    # psds_eval = PSDSEval_Grounding(
    psds_eval = PSDSEval(
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
        
        psds_eval.operating_points.drop(["id"], axis=1).to_csv(
            save_dir / f"op_table_dtc{dtc_threshold}_gtc{gtc_threshold}.csv",
            sep="\t", index=False, float_format="%.3f")

        plot_psd_roc(
            psds_score,
            filename=save_dir / f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_maxefpr{max_efpr}.png",
        )

    return psds_score.value


def compute_psds_sed_scores(scores,
                            ground_truth,
                            duration,
                            fname_to_aid,
                            dtc_threshold=0.5,
                            gtc_threshold=0.5,
                            max_efpr=None,
                            save_dir=None):

    duration = pd.read_csv(duration, sep="\t")
    aid_to_dur = dict(zip(duration["audio_id"], duration["duration"]))
    metadata = {}
    for fname in ground_truth:
        aid = fname_to_aid[fname]
        metadata[fname] = aid_to_dur[aid]

    psds, psd_roc, _ = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=metadata,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=None,
        alpha_ct=0.,
        alpha_st=0.,
        unit_of_time='hour',
        max_efpr=max_efpr,
        num_jobs=4,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        etpr, efpr = psd_roc
        plt.figure(figsize=(16, 4))
        plt.ylabel('eTPR')
        plt.xlabel('eFPR per hour')
        plt.step(efpr, etpr, lw=2, where='post')
        plt.legend(['sed_scores_eval',], loc='lower right')
        plt.savefig(save_dir / f"PSDS_sedscores_dtc{dtc_threshold}_gtc{gtc_threshold}_maxefpr{max_efpr}.png")

    return psds


def compute_th_auc(prediction_dfs,
                   ground_truth,
                   dtc_threshold=0.5,
                   gtc_threshold=0.5,
                   min_threshold=0.0,
                   max_threshold=1.0,
                   beta=1.,
                   save_dir=None):

    if not isinstance(ground_truth, pd.DataFrame):
        ground_truth = pd.read_csv(ground_truth, sep="\t")

    evaluator = Grounding_PrecisionRecall(dtc_threshold,
                                          gtc_threshold,
                                          ground_truth)

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # if abs(k - 0.71) < 1e-6:
            # import pdb; pdb.set_trace()
        evaluator.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    th_auc = evaluator.th_auc(beta=beta,
                              low_th=min_threshold,
                              high_th=max_threshold)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        evaluator.operating_points.drop(["id"], axis=1).to_csv(
            save_dir / f"op_table_dtc{dtc_threshold}_gtc{gtc_threshold}.csv",
            sep="\t", index=False, float_format="%.3f")
        evaluator.plot_f_threshold(save_dir / "f_vs_th.png")

    return th_auc


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


Thresholds = namedtuple("Thresholds", ["gtc", "dtc"])


class Grounding_PrecisionRecall(PSDSEval):

    detection_cols = ["filename", "onset", "offset"]

    def __init__(self, dtc_threshold, gtc_threshold, ground_truth):
        if dtc_threshold < 0.0 or dtc_threshold > 1.0:
            raise PSDSEvalError("dtc_threshold must be between 0 and 1")
        if gtc_threshold < 0.0 or gtc_threshold > 1.0:
            raise PSDSEvalError("gtc_threshold must be between 0 and 1")

        self.class_names = []
        self.threshold = Thresholds(dtc=dtc_threshold, gtc=gtc_threshold)
        self.operating_points = self._operating_points_table()
        self.ground_truth = None
        self.metadata = None
        self.eps = 1e-15
        self.set_ground_truth(ground_truth)

    @staticmethod
    def _operating_points_table():
        """Returns and empty operating point table with the correct columns"""
        return pd.DataFrame(columns=["id", "precision", "recall", "threshold"])

    def set_ground_truth(self, gt_t):
        if self.ground_truth is not None:
            raise PSDSEvalError("You cannot set the ground truth more than"
                                " once per evaluation")
        if gt_t is None:
            raise PSDSEvalError("The ground truth cannot be set without data")

        self._validate_input_table(
            gt_t, self.detection_cols, "ground truth", allow_empty=True)

        ground_truth_t = gt_t.sort_values(by=self.detection_cols[:2],
                                          axis=0)
        ground_truth_t.dropna(inplace=True)
        ground_truth_t["duration"] = \
            ground_truth_t.offset - ground_truth_t.onset
        ground_truth_t["id"] = ground_truth_t.index

        self.ground_truth = ground_truth_t

    def _init_det_table(self, det_t):
        self._validate_input_table(
            det_t, self.detection_cols, "detection", allow_empty=True)
        detection_t = det_t.sort_values(by=self.detection_cols[:2], axis=0)
        detection_t["duration"] = detection_t.offset - detection_t.onset
        detection_t["id"] = detection_t.index
        return detection_t

    def _add_op(self, recall, precision, info=None):
        """Adds a new operating point into the class"""
        op = {"recall": recall, "precision": precision}
        if not info:
            info = dict()

        if set(op.keys()).isdisjoint(set(info.keys())):
            op.update(info)
            self.operating_points = \
                self.operating_points.append(op, ignore_index=True)
        else:
            raise PSDSEvalError("the 'info' cannot contain the keys "
                                "'recall', 'precision'")

    def _operating_point_id(self, detection_table):
        """Used to produce a unique ID for each operating point

        here we sort the dataframe so that shuffled versions of the same
        detection table results in the same hash
        """

        table_columns = ["filename", "onset", "offset"]
        detection_table_col_sorted = detection_table[
            table_columns]
        detection_table_row_sorted = detection_table_col_sorted.sort_values(
            by=table_columns)
        h = hashlib.sha256(pd.util.hash_pandas_object(
            detection_table_row_sorted, index=False).values)
        uid = h.hexdigest()
        if uid in self.operating_points.id.values:
            warn("A similar operating point exists, skipping this one")
            uid = ""
        return uid

    def add_operating_point(self, detections, info=None):
        if self.ground_truth is None:
            raise PSDSEvalError("Ground Truth must be provided before "
                                "adding the first operating point")

        # validate and prepare tables
        det_t = self._init_det_table(detections)
        op_id = self._operating_point_id(det_t)
        info["id"] = op_id
        if not op_id:
            threshold = info["threshold"]
            last_row = self.operating_points.iloc[-1].to_dict()
            last_row["threshold"] = threshold
            self.operating_points = self.operating_points.append(
                last_row, ignore_index=True)
            return

        precision, recall = self._evaluate_detections(det_t)
        self._add_op(recall=recall, precision=precision, info=info)


    @staticmethod
    def _ground_truth_intersections(detection_t, ground_truth_t):
        comb_t = pd.merge(detection_t, ground_truth_t,
                          how='outer', on='filename',
                          suffixes=("_det", "_gt"))
        # cross_t contains detections that intersect one or more ground truths
        cross_t = comb_t[(comb_t.onset_det <= comb_t.offset_gt) &
                         (comb_t.onset_gt <= comb_t.offset_det) &
                         comb_t.filename.notna()].copy(deep=True)

        cross_t["inter_duration"] = \
            np.minimum(cross_t.offset_det, cross_t.offset_gt) - \
            np.maximum(cross_t.onset_det, cross_t.onset_gt)
        cross_t["det_precision"] = \
            cross_t.inter_duration / cross_t.duration_det
        cross_t["gt_coverage"] = \
            cross_t.inter_duration / cross_t.duration_gt
        return cross_t

    def _recall_criteria(self, cross_t):
        # Group the duplicate detections and sum the det_precision
        if cross_t.empty:
            dtc_t = pd.DataFrame(columns=["id_det",
                                          "det_precision"])
        else:
            dtc_t = cross_t.groupby(
                ["id_det"]
            ).det_precision.sum().reset_index()

        # when calculating gt_coverage_sum, only count detection that satisfies dtc requirement
        dtc_ids = dtc_t[dtc_t.det_precision >= self.threshold.dtc].id_det

        # Group the duplicate detections that exist in the DTC set and sum
        gtc_t = cross_t[cross_t.id_det.isin(dtc_ids)].groupby(
            ["id_gt"]
        ).gt_coverage.sum().reset_index()

        # Join the two into a single true positive table
        if len(dtc_t) or len(gtc_t):
            tmp = pd.merge(cross_t, dtc_t, on=["id_det"],
                           suffixes=("", "_sum")
                           ).merge(gtc_t, on=["id_gt"],
                                   suffixes=("", "_sum"))
        else:
            cols = cross_t.columns.to_list() + \
                   ["det_precision_sum", "gt_coverage_sum"]
            tmp = pd.DataFrame(columns=cols)

        gtc_filter = tmp.gt_coverage_sum >= self.threshold.gtc

        num_tp_refs = tmp[gtc_filter].id_gt.unique().shape[0]
        return num_tp_refs


    def _precision_criteria(self, cross_t):
        # Group the duplicate detections and sum the det_precision
        if cross_t.empty:
            gtc_t = pd.DataFrame(columns=["id_gt", 
                                          "gt_coverage"])
        else:
            gtc_t = cross_t.groupby(
                ["id_gt"]
            ).gt_coverage.sum().reset_index()

        # when calculating det_precision_sum, only count ground truth that satisfies gtc requirement
        gtc_ids = gtc_t[gtc_t.gt_coverage >= self.threshold.gtc].id_gt

        # Group the duplicate detections that exist in the DTC set and sum
        dtc_t = cross_t[cross_t.id_gt.isin(gtc_ids)].groupby(
            ["id_det"]
        ).det_precision.sum().reset_index()

        # Join the two into a single true positive table
        if len(dtc_t) or len(gtc_t):
            tmp = pd.merge(cross_t, gtc_t, on=["id_gt"],
                           suffixes=("", "_sum")
                           ).merge(dtc_t, on=["id_det"],
                                   suffixes=("", "_sum"))
        else:
            cols = cross_t.columns.to_list() + \
                   ["det_precision_sum", "gt_coverage_sum"]
            tmp = pd.DataFrame(columns=cols)

        dtc_filter = tmp.det_precision_sum >= self.threshold.dtc

        num_tp_preds = tmp[dtc_filter].id_det.unique().shape[0]
        return num_tp_preds

    def _evaluate_detections(self, det_t):
        inter_t = self._ground_truth_intersections(det_t, self.ground_truth)
        num_tp_refs = self._recall_criteria(inter_t)
        num_tp_preds = self._precision_criteria(inter_t)
        num_refs = self.ground_truth.shape[0]
        num_preds = det_t.shape[0]

        recall = num_tp_refs / np.maximum(num_refs, self.eps)
        precision = num_tp_preds / np.maximum(num_preds, self.eps)

        return precision, recall

    def th_auc(self, beta=1., low_th=0., high_th=1.):
        precision = self.operating_points.precision
        recall = self.operating_points.recall
        self.operating_points["f_score"] = ((1 + beta**2) * precision * recall) \
            / np.maximum(beta**2 * precision + recall, self.eps)

        sub_table = self.operating_points[
            (self.operating_points.threshold >= low_th) & 
            (self.operating_points.threshold <= high_th)]
        sort_idxs = np.argsort(sub_table.threshold.values)
        score = auc(sub_table.threshold.values[sort_idxs],
                    sub_table.f_score.values[sort_idxs])
        return score / (high_th - low_th)

    def plot_f_threshold(self, fig_path):
        sort_idxs = np.argsort(self.operating_points.threshold.values)
        ths = self.operating_points.threshold.values[sort_idxs]
        fs = self.operating_points.f_score.values[sort_idxs]
        plt.figure(figsize=(14, 5))
        plt.plot(ths, fs)
        plt.ylim(0., 1.)
        plt.xlabel("threshold")
        plt.ylabel("f_score")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        
