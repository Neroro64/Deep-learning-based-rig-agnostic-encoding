import torch
import torch.nn as nn


def mse_loss(x,y):
    return nn.functional.mse_loss(x,y)


def mae_loss(x,y):
    return nn.functional.smooth_l1_loss(x,y)


def kl_loss(x,y):
    return nn.functional.kl_div(x,y)


def nll_loss(x,y):
    return nn.functional.nll_loss(x,y)


def mse_loss4(out, h, y):
    return nn.functional.mse_loss(torch.cat(out, dim=1), y)


def mae_loss4(out, h, y):
    return nn.functional.smooth_l1_loss(torch.cat(out, dim=1), y)


def mse_mse_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += mae_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    return output_loss + similarity_loss


def mae_mae_loss(out, h, y):
    output_loss = mae_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += mae_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_mae_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += mae_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_mae_loss2(out, h, y):
    output_loss = mse_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += mae_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    k1 = 0.5
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss

def mse_kl_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += kl_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_nll_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    similarity_loss = 0
    c = 0
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j: continue
            similarity_loss += nll_loss(h[i], h[j])
            c+=1

    similarity_loss /= c
    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss
