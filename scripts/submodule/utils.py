"""
get_dir.pyでimportされる
submodules.mychainerutils.utils
"""

from os import environ
from pathlib import Path
from pkgutil import get_data

def get_datasetroot():
    if "DATASET_ROOT" in environ:
        ret = Path(environ["DATASET_ROOT"])
    else:
        ret = Path("~", "dataset")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret


def get_saveroot():
    if "SAVE_ROOT" in environ:
        ret = Path(environ["SAVE_ROOT"])
    else:
        ret = Path("./results")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret


if __name__ == "__main__":
    y = get_datasetroot()
    print(y)    # /Users/minami/dataset