"""directory name utillity"""
import os

from submodule.utils import get_saveroot, get_datasetroot


def get_save_directory():
    """return SAVEDIR"""
    # os.path.join()：pathを結合
    return os.path.join(get_saveroot(), "lip2sp")


def get_data_directory():
    """return SAVEDIR"""
    return os.path.join(get_datasetroot(), "lip")

if __name__ == "__main__":
    out = get_save_directory()
    print(out)