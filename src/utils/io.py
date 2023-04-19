import errno
import json
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        f = json.load(f)
    return f


def dump_json(file: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(file, f, indent=4)
