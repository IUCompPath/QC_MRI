import argparse
import os
import re

import pandas as pd

from brainmaskQC.utils import csv_to_regex


def evaluate_metrics(args: argparse.Namespace) -> None:
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # convert given ground truth parameter to regex
    gt_rgx = re.compile(csv_to_regex(args.ground_truth))

    # read in r file
    r_values = pd.read_csv(args.result_file)

    # keep track of metrics
    metrics = {
        "set": [],
        "sensitivity": [],
        "specificity": [],
        "accuracy": [],
        "precision": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
    }

    for (
        set_,
        set_id,
    ) in zip(args.set, args.set_id):
        # convert given parameters to regex expression
        set_rgx = re.compile(csv_to_regex(set_))

        # keep track of result
        result = {"file": [], "label": []}

        # keep track of label count for TP, FP, TN, FN
        n_tp = 0
        n_fp = 0
        n_tn = 0
        n_fn = 0

        # keep track of predicted and true labels
        true_label = []
        pred_label = []

        # iterate over original results
        for fname, r in zip(r_values["file"], r_values["r"]):
            # check if file is in analysis set
            if set_rgx.match(fname):
                result["file"].append(fname)
                # see if it is ground truth or not
                if not gt_rgx.match(fname):
                    true_label.append(0)
                    # True negative
                    if r <= 0.5:
                        pred_label.append(0)
                        result["label"].append("TN")
                        n_tn += 1
                    # False positive
                    else:
                        pred_label.append(1)
                        result["label"].append("FP")
                        n_fp += 1
                else:
                    true_label.append(1)
                    # False negative
                    if r <= 0.5:
                        pred_label.append(0)
                        result["label"].append("FN")
                        n_fn += 1
                    # True positive
                    else:
                        pred_label.append(1)
                        result["label"].append("TP")
                        n_tp += 1

        # save results
        fpath = os.path.join(args.output_dir, f"{set_id}.csv")
        df = pd.DataFrame(result)
        df.to_csv(fpath)
        print(f"Saved results to {fpath}")

        # print results
        print(f"###--- RESULTS for {set_id} ---###")
        print(f"Number of files evaluated: {len(result['file'])}")
        print(f"TP count: {n_tp}")
        print(f"FP count: {n_fp}")
        print(f"TN count: {n_tn}")
        print(f"FN count: {n_fn}")
        print(f"Sensitivity: {n_tp/(n_tp+n_fn)}")
        print(f"Specificity: {n_tn/(n_tn+n_fp)}")
        print(f"Accuracy: {(n_tp + n_tn) / (n_tn + n_fn + n_tp + n_fp)}")
        print(f"Precision: {n_tp / (n_tp + n_fp)}")

        metrics["set"].append(set_id)
        metrics["sensitivity"].append(n_tp / (n_tp + n_fn))
        metrics["specificity"].append(n_tn / (n_tn + n_fp))
        metrics["accuracy"].append((n_tp + n_tn) / (n_tn + n_fn + n_tp + n_fp))
        metrics["precision"].append(n_tp / (n_tp + n_fp))
        metrics["TP"].append(n_tp)
        metrics["TN"].append(n_tn)
        metrics["FP"].append(n_fp)
        metrics["FN"].append(n_fn)

    df = pd.DataFrame(metrics)
    df.to_csv(args.metrics_file)