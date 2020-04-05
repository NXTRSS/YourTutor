import json
import logging

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def setup_logging(filename=None, level=logging.INFO, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(level)

    # console
    # console_handler = logging.StreamHandler()
    # logger.addHandler(console_handler)

    # file
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode=mode)
        logger.addHandler(file_handler)
