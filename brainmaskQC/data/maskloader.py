from typing import List

import os
import re


class MaskLoader(object):
    def __init__(
        self, source_dir: str, mask_f_pattern: str, excluded_f_patterns: List[str] = []
    ) -> object:
        # combine excluded file pattern into single regex
        ex_combined = (
            f"({'|'.join(excluded_f_patterns)})" if excluded_f_patterns != [] else "a^"
        )

        # create regex
        self.mask_regex = re.compile(f"(?=({mask_f_pattern}))(?!({ex_combined}))")

        # get list of files
        self.mask_files = []
        for root, _, files in os.walk(source_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if self.mask_regex.match(full_path):
                    self.mask_files.append(full_path)

        # sort files by custom method
        self.mask_files.sort(key=self.__get_file_name)

    # sort by file name
    @staticmethod
    def __get_file_name(s: str) -> str:
        return s.split(os.sep)[-1]

    def loader(self, batch_size: int = -1) -> List:
        if batch_size < 0:
            return self.mask_files
        # otherwise return in batches
        else:
            batched_files = [
                self.mask_files[pos : pos + batch_size]
                for pos in range(0, len(self.mask_files), batch_size)
            ]
            return batched_files
