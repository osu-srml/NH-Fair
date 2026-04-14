import sys
from datetime import datetime

import numpy as np
import pytz
import torch


def get_et_time():
    eastern_time = datetime.now().astimezone(pytz.timezone("America/New_York"))
    formatted_time = eastern_time.strftime("%Y%m%d-%H%M%S")
    return formatted_time


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")  # noqa: SIM115

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility with the standard stdout interface.
        self.terminal.flush()
        self.log.flush()
