import gzip
import json
from pathlib import Path
from typing import Union

import numpy as np


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def load_txt(filename: Union[str, Path]):
    return Path(filename).read_text(encoding="utf-8")


def dict_to_omegaconf_format(obj) -> dict:
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert int32/int64 to Python int
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert float32/float64 to Python float
    elif isinstance(obj, list):
        return [
            dict_to_omegaconf_format(item) for item in obj
        ]  # Recurse for list elements
    elif isinstance(obj, dict):
        return {
            k: dict_to_omegaconf_format(v) for k, v in obj.items()
        }  # Recurse for dict values
    else:
        return obj  # Return unchanged if type is already supported
