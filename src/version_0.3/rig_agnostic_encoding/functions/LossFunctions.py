import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import combinations


def mse_loss(x,y):
    return (x-y).pow(2).mean(dim=(0,-1)).sum()


def mae_loss(x,y):
    return F.smooth_l1_loss(x,y)


def pos_rot_loss(px, py, rx, ry):
    px_norm, py_norm = torch.sum(px ** 2), torch.sum(py ** 2)
    pos_loss = (px - py) ** 2 / (px_norm * py_norm)

    rx %= 2 * np.pi
    ry %= 2 * np.pi
    rot_loss = mse_loss(rx, ry)
    return pos_loss, rot_loss
