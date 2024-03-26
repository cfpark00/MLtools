import os
import torch
import numpy as np
import warnings


def get_freer_device():
    if torch.cuda.is_available():
        i_gpu = get_freer_gpu(report=True)
        return f"cuda:{i_gpu}"
    else:
        return "cpu"


def get_freer_gpu(report=True):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Total >tmp_total")
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Reserved >tmp_reserved")
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp_used")
    memory_total = [int(x.split()[2]) for x in open("tmp_total", "r").readlines()]
    memory_reserved = [int(x.split()[2]) for x in open("tmp_reserved", "r").readlines()]
    memory_used = [int(x.split()[2]) for x in open("tmp_used", "r").readlines()]
    memory_available = [
        x - y - z for x, y, z in zip(memory_total, memory_reserved, memory_used)
    ]
    if len(memory_available) == 0:
        warnings.warn("No GPU is available.")
        return None
    i_gpu = np.argmax(memory_available)
    if report:
        print("memory_available", memory_available)
        print("best GPU:", i_gpu)
    return i_gpu
