import argparse
import os
import ast
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from brainmaskQC.data import MaskLoader
from brainmaskQC.features import MorphPipe
from brainmaskQC.memquant import (
    RandClustering,
    eval_mahalanobis_metrics,
    eval_intra_cohort_metrics,
    get_nth_percentile,
)


def run(args: argparse.Namespace) -> None:
    # initalize data loader
    loader = MaskLoader(args.source_dir, args.mask_pattern)

    # initialize Pipeline and get features
    pipeline = MorphPipe()
    ids, features = pipeline.get_features(
        loader,
        args.save_features_to,
        (None if args.load_features_from == "" else args.load_features_from),
        override=(args.override != 0),
        n_workers=args.n_workers,
    )

    # initialize clusterer
    clusterer = RandClustering(args.cluster_range, lambda n: KMeans(n, n_init="auto"))

    # initialize preprocessing
    gt_file_ids, gt_features = pipeline.process_features(
        ids, features, regex=args.gt_regex, build=True
    )
    # perform clustering
    print("### Performing Clustering ###")
    gt_labels = clusterer.fit(
        gt_features, n_iterations=args.n_iterations, n_workers=args.n_workers
    )
    # evaluate metrics for existing ground truth
    gt_min_mahal, gt_wmean_mahal, gt_median_mahal = eval_intra_cohort_metrics(
        gt_features, gt_labels
    )

    # get baseline values for r metric calculation
    min_baseline = get_nth_percentile(gt_min_mahal, args.nth_percentile)
    wmean_baseline = get_nth_percentile(gt_wmean_mahal, args.nth_percentile)
    median_baseline = get_nth_percentile(gt_median_mahal, args.nth_percentile)

    # calculate r value for ground truth set
    res_all = {}
    res_all["file"] = gt_file_ids[::]
    res_all["r"] = []
    for i in range(len(gt_min_mahal)):
        min_ratio = gt_min_mahal[i] / min_baseline
        wmean_ratio = gt_wmean_mahal[i] / wmean_baseline
        median_ratio = gt_median_mahal[i] / median_baseline
        res_all["r"].append(
            np.arctan(
                min_ratio * args.min_ratio
                + wmean_ratio * args.wmean_ratio
                + median_ratio * args.median_ratio
            )
            * 2
            / np.pi
        )

    # create dir if doesn't exist
    if not os.path.exists(args.csv_dir):
        os.mkdir(args.csv_dir)

    # save to csv
    df_res_gt = pd.DataFrame(res_all)
    df_res_gt.to_csv(os.path.join(args.csv_dir, "ground_truth.csv"))

    # cacluate r value for remaining sets
    for i, (qc_set_id, qc_set_regex) in enumerate(
        zip(args.qc_set_ids, args.qc_set_regexs)
    ):
        # perform preprocessing on set that needs qc-ing
        qc_file_ids, qc_features = pipeline.process_features(
            ids, features, regex=qc_set_regex, build=False
        )

        # evalute metrics for set that needs qc-ing
        qc_min_mahal, qc_wmean_mahal, qc_median_mahal = eval_mahalanobis_metrics(
            qc_features, gt_features, gt_labels
        )

        # calculate r value for set that needs qc-ing
        res = {}
        res["file"] = qc_file_ids[::]
        res["r"] = []
        for i in range(len(qc_min_mahal)):
            min_ratio = qc_min_mahal[i] / min_baseline
            wmean_ratio = qc_wmean_mahal[i] / wmean_baseline
            median_ratio = qc_median_mahal[i] / median_baseline
            res["r"].append(
                np.arctan(
                    min_ratio * args.min_ratio
                    + wmean_ratio * args.wmean_ratio
                    + median_ratio * args.median_ratio
                )
                * 2
                / np.pi
            )

        res_all["file"].extend(res["file"])
        res_all["r"].extend(res["r"])

        # save to csv
        df_res = pd.DataFrame(res)
        df_res.to_csv(os.path.join(args.csv_dir, f"{qc_set_id}.csv"))

    # save cummilative csv
    df_all = pd.DataFrame(res_all)
    df_all.to_csv(os.path.join(args.csv_dir, "allmetrics.csv"))

    print("Finished.")
