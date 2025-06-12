import os
import re

def csv_to_regex(string: str) -> str:

    # in case string matches a csv filename, convert all file names into one regex
    if os.path.exists(string) and os.path.splitext(string)[1] == ".csv":
        with open(string) as csv_file:
            func = lambda s: "(.*" + re.escape(s) + "$)"
            files_read = csv_file.read().splitlines()
            rgxs = map(func, files_read)
            regex = "(" + "|".join(rgxs) + ")"
        return regex
    # otherwise just return the string
    else:
        return string