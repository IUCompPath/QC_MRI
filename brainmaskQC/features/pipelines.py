import abc
from typing import Union, Dict, Tuple

import os
import ctypes
from multiprocessing import Process, Queue, Array
import re
import pickle

import numpy as np
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler

from .morphological import get_brainmask_features
from brainmaskQC.utils import csv_to_regex
from brainmaskQC.data import MaskLoader


class Pipeline(abc.ABC):
    def __init__(self) -> object:
        self.scaler = None

    @abc.abstractmethod
    def feature_method(self, x: sitk.Image) -> list:
        pass

    @abc.abstractmethod
    def processing_method(self, x: np.ndarray, rebuild: bool = False) -> np.ndarray:
        pass

    def get_features(
        self,
        loader: MaskLoader,
        save_as: Union[str, None] = None,
        load_from: Union[str, None] = None,
        override: bool = False,
        n_workers: int = 12,
    ) -> Tuple[list, np.ndarray]:
        # get shape of features
        temp_mask = sitk.ReadImage(loader.loader()[0], sitk.sitkUInt8)
        temp_features = self.feature_method(temp_mask)
        shape = (len(loader.loader()), len(temp_features))

        if load_from == None:
            load_from = save_as
        if override:
            load_from = None

        # load whatever features have already been saved
        if load_from != None:
            existing_features = self._load_features(load_from)
            # check if shapes match if dictionary isn't empty
            if bool(existing_features) and len(
                next(iter(existing_features.values()))
            ) != len(temp_features):
                # set dictionary as empty if shapes don't match
                existing_features = {}
        else:
            existing_features = {}

        # setup for multiprocessing
        shared_queue = Queue()
        shared_arr = Array(ctypes.c_double, shape[0] * shape[1])
        ids = []

        # initialize subprocesses
        workers = [
            FeatureWorker(
                shared_queue, shared_arr, existing_features, shape, self.feature_method
            )
            for _ in range(n_workers)
        ]

        for worker in workers:
            worker.start()

        # iterate over files and send them to queue
        for i, fname in enumerate(loader.loader()):
            # save fname as file identifier
            ids.append(fname)
            # queue up processing of file
            shared_queue.put((i, fname[::]))

        # send None as terminating flag for processes
        for _ in workers:
            shared_queue.put(None)

        # join
        for worker in workers:
            worker.join()

        # convert shared arr to np.ndarray
        features = np.frombuffer(shared_arr.get_obj())
        features = np.reshape(features, shape)

        # save features
        if save_as != None:
            self._save_features(ids, features, save_as)

        return (ids, features)

    def process_features(
        self, ids: list, features: np.ndarray, regex: str = ".*", build: bool = False
    ) -> Tuple[list, np.ndarray]:
        # in case regex argument is csv filename, convert all file names into regex
        regex = csv_to_regex(regex)

        # filter ids based on regex
        reg = re.compile(regex)
        file_filter = np.array([reg.match(id_) for id_ in ids], dtype=bool)
        filtered_ids = [ids[i] for i in range(len(ids)) if file_filter[i]]

        # filter features
        filtered_features = features[file_filter]

        # scale features
        if self.scaler == None or build:
            self.scaler = StandardScaler()
            self.scaler.fit(filtered_features)
        scaled_feats = self.scaler.transform(filtered_features)

        # perform additional processing
        processed_feats = self.processing_method(scaled_feats)

        return (filtered_ids, processed_feats)

    @staticmethod
    def _load_features(path: str) -> Dict:
        print(path)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"LOAD COMPLETE: loaded features from {path}")
            return data
        except:
            print(f"LOAD FAILED: {path} does not exist")
            return {}

    @staticmethod
    def _save_features(ids: list, feats: np.ndarray, path: str) -> None:
        # convert nd.array into dictionary
        feat_dict = {}
        for id_, feat in zip(ids, feats):
            feat_dict[id_] = feat.tolist()

        # save dictionary
        try:
            with open(path, "wb") as f:
                pickle.dump(feat_dict, f)
            print(f"SAVE COMPLETE: saved features to {path}")
        except:
            print(f"SAVE FAILED: saving to {path} failed")


class FeatureWorker(Process):
    def __init__(
        self,
        queue: Queue,
        shared_arr: Array,
        existing_feats: Dict,
        arr_shape: Tuple[int],
        method: callable,
    ) -> object:
        Process.__init__(self, name="FeatureWorker")
        self.queue = queue
        self.existing_feats = existing_feats
        self.arr_shape = arr_shape
        self.method = method

        # convert shared array into numpy for further processing
        arr = np.frombuffer(shared_arr.get_obj())
        self.shared_arr = np.reshape(arr, arr_shape)

    def run(self) -> None:
        while True:
            # get queue input
            inp = self.queue.get()
            # terminate if None
            if inp == None:
                break
            else:
                (i, fname) = inp

                # give feedback to terminal
                print(
                    "Running process --- {:.2f}%".format(
                        (i + 1) / self.arr_shape[0] * 100
                    )
                )

                # use loaded features if they were already created
                if fname in self.existing_feats:
                    self.shared_arr[i] = self.existing_feats[fname]

                # otherwise create features
                else:
                    x = sitk.ReadImage(fname, sitk.sitkUInt8)
                    self.shared_arr[i] = self.method(x)


class MorphPipe(Pipeline):
    def __init__(self) -> object:
        Pipeline.__init__(self)

    def feature_method(self, x: sitk.Image) -> list:
        return get_brainmask_features(x)

    def processing_method(self, x: np.ndarray, rebuild: bool = False) -> np.ndarray:
        return x
