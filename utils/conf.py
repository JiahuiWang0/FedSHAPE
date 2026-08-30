import random
import torch
import numpy as np


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def data_path() -> str:
    return '' # type your data path here.

def base_path() -> str:
    return ''

def log_path():
    return ''

def checkpoint_path() -> str:
    return ''


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
