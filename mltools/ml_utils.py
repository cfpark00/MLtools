import torch

def to_np(ten):
    return ten.detach().cpu().numpy()