import torch
import torch.nn as nn
from itertools import combinations

def mse_loss(x,y):
    return (x-y).pow(2).mean(dim=(0,-1)).sum()


def mae_loss(x,y):
    return nn.functional.smooth_l1_loss(x,y)


def nll_loss(x,y):
    return nn.functional.nll_loss(x,y)


def mse_loss4(out, h, y):
    return mse_loss(torch.cat(out, dim=1), y)


def mae_loss4(out, h, y):
    return nn.functional.smooth_l1_loss(torch.cat(out, dim=1), y)

def kl_loss(recon_x, x, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum().clamp(max=0)
        kl_loss /= z_log_var.numel()

        recon_loss = (recon_x - x).pow(2).mean(dim=(0,-1)).sum()

        return kl_loss + recon_loss, kl_loss, recon_loss


def mse_mse_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mse_loss(*h_pairs[0])
    similarity_loss += mse_loss(*h_pairs[1])
    similarity_loss += mse_loss(*h_pairs[2])
    similarity_loss += mse_loss(*h_pairs[3])
    similarity_loss += mse_loss(*h_pairs[4])
    similarity_loss += mse_loss(*h_pairs[5])
    similarity_loss /= 6

    return output_loss + similarity_loss





def mae_mae_loss(out, h, y):
    output_loss = mae_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*(h_pairs[0]))
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6

    k1 = 1.0
    k2 = 1.0
    return k1 * output_loss + k2 * similarity_loss


def mae_mae_loss2(out, h, y):
    output_loss = mae_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*h_pairs[0])
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6

    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_mae_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*h_pairs[0])
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6

    k1 = 1.0
    k2 = 1.0
    return k1 * output_loss + k2 * similarity_loss


def mse_mae_loss2(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*h_pairs[0])
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6


    k1 = 0.5
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss

def mse_kl_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = kl_loss(*h_pairs[0])
    similarity_loss += kl_loss(*h_pairs[1])
    similarity_loss += kl_loss(*h_pairs[2])
    similarity_loss += kl_loss(*h_pairs[3])
    similarity_loss += kl_loss(*h_pairs[4])
    similarity_loss += kl_loss(*h_pairs[5])
    similarity_loss /= 6
    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_nll_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = nll_loss(*h_pairs[0])
    similarity_loss += nll_loss(*h_pairs[1])
    similarity_loss += nll_loss(*h_pairs[2])
    similarity_loss += nll_loss(*h_pairs[3])
    similarity_loss += nll_loss(*h_pairs[4])
    similarity_loss += nll_loss(*h_pairs[5])
    similarity_loss /= 6

    k1 = 1.0
    k2 = 2.0
    return k1 * output_loss + k2 * similarity_loss


def mse_mse_mse_loss(out, h, c, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mse_loss(*h_pairs[0])
    similarity_loss += mse_loss(*h_pairs[1])
    similarity_loss += mse_loss(*h_pairs[2])
    similarity_loss += mse_loss(*h_pairs[3])
    similarity_loss += mse_loss(*h_pairs[4])
    similarity_loss += mse_loss(*h_pairs[5])
    similarity_loss /= 6

    mu = [cc[0] for cc in c]
    sigma = [cc[1] for cc in c]

    mu_pairs = list(combinations(mu, r=2))
    mu_loss = mse_loss(*mu_pairs[0])
    mu_loss += mse_loss(*mu_pairs[1])
    mu_loss += mse_loss(*mu_pairs[2])
    mu_loss += mse_loss(*mu_pairs[3])
    mu_loss += mse_loss(*mu_pairs[4])
    mu_loss += mse_loss(*mu_pairs[5])
    mu_loss /= 6

    sigma_pairs = list(combinations(sigma, r=2))
    sigma_loss = mse_loss(*sigma_pairs[0])
    sigma_loss += mse_loss(*sigma_pairs[1])
    sigma_loss += mse_loss(*sigma_pairs[2])
    sigma_loss += mse_loss(*sigma_pairs[3])
    sigma_loss += mse_loss(*sigma_pairs[4])
    sigma_loss += mse_loss(*sigma_pairs[5])
    sigma_loss /= 6


    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0
    return k1 * output_loss + k2 * similarity_loss + k3 * mu_loss + k4 * sigma_loss


def mae_mae_mae_loss(out, h, c, y):
    output_loss = mae_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*h_pairs[0])
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6

    mu = [cc[0] for cc in c]
    sigma = [cc[1] for cc in c]

    mu_pairs = list(combinations(mu, r=2))
    mu_loss = mae_loss(*mu_pairs[0])
    mu_loss += mae_loss(*mu_pairs[1])
    mu_loss += mae_loss(*mu_pairs[2])
    mu_loss += mae_loss(*mu_pairs[3])
    mu_loss += mae_loss(*mu_pairs[4])
    mu_loss += mae_loss(*mu_pairs[5])
    mu_loss /= 6

    sigma_pairs = list(combinations(sigma, r=2))
    sigma_loss = mae_loss(*sigma_pairs[0])
    sigma_loss += mae_loss(*sigma_pairs[1])
    sigma_loss += mae_loss(*sigma_pairs[2])
    sigma_loss += mae_loss(*sigma_pairs[3])
    sigma_loss += mae_loss(*sigma_pairs[4])
    sigma_loss += mae_loss(*sigma_pairs[5])
    sigma_loss /= 6


    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0
    return k1 * output_loss + k2 * similarity_loss + k3 * mu_loss + k4 * sigma_loss


def mae_mae_mae_loss0(out, h, c, y):
    output_loss = mae_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mae_loss(*h_pairs[0])
    similarity_loss += mae_loss(*h_pairs[1])
    similarity_loss += mae_loss(*h_pairs[2])
    similarity_loss += mae_loss(*h_pairs[3])
    similarity_loss += mae_loss(*h_pairs[4])
    similarity_loss += mae_loss(*h_pairs[5])
    similarity_loss /= 6

    # mu = c[0]
    # sigma = c[1]
    #
    # mu_pairs = list(combinations(mu, r=2))
    # mu_loss = mae_loss(*mu_pairs[0])
    # mu_loss += mae_loss(*mu_pairs[1])
    # mu_loss += mae_loss(*mu_pairs[2])
    # mu_loss += mae_loss(*mu_pairs[3])
    # mu_loss += mae_loss(*mu_pairs[4])
    # mu_loss += mae_loss(*mu_pairs[5])
    # mu_loss /= 6
    #
    # sigma_pairs = list(combinations(sigma, r=2))
    # sigma_loss = mae_loss(*sigma_pairs[0])
    # sigma_loss += mae_loss(*sigma_pairs[1])
    # sigma_loss += mae_loss(*sigma_pairs[2])
    # sigma_loss += mae_loss(*sigma_pairs[3])
    # sigma_loss += mae_loss(*sigma_pairs[4])
    # sigma_loss += mae_loss(*sigma_pairs[5])
    # sigma_loss /= 6

    mu_loss = 0
    sigma_loss = 0

    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0
    return k1 * output_loss + k2 * similarity_loss + k3 * mu_loss + k4 * sigma_loss
def mae_kl_mae(out, h, c, y):
    output_loss = mae_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = kl_loss(*h_pairs[0])
    similarity_loss += kl_loss(*h_pairs[1])
    similarity_loss += kl_loss(*h_pairs[2])
    similarity_loss += kl_loss(*h_pairs[3])
    similarity_loss += kl_loss(*h_pairs[4])
    similarity_loss += kl_loss(*h_pairs[5])
    similarity_loss /= 6

    mu = [cc[0] for cc in c]
    sigma = [cc[1] for cc in c]

    mu_pairs = list(combinations(mu, r=2))
    mu_loss = mae_loss(*mu_pairs[0])
    mu_loss += mae_loss(*mu_pairs[1])
    mu_loss += mae_loss(*mu_pairs[2])
    mu_loss += mae_loss(*mu_pairs[3])
    mu_loss += mae_loss(*mu_pairs[4])
    mu_loss += mae_loss(*mu_pairs[5])
    mu_loss /= 6

    sigma_pairs = list(combinations(sigma, r=2))
    sigma_loss = mae_loss(*sigma_pairs[0])
    sigma_loss += mae_loss(*sigma_pairs[1])
    sigma_loss += mae_loss(*sigma_pairs[2])
    sigma_loss += mae_loss(*sigma_pairs[3])
    sigma_loss += mae_loss(*sigma_pairs[4])
    sigma_loss += mae_loss(*sigma_pairs[5])
    sigma_loss /= 6


    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0
    return k1 * output_loss + k2 * similarity_loss + k3 * mu_loss + k4 * sigma_loss

def mse_dec_kl_loss(out, h, y):
    output_loss = mse_loss4(out, h, y)

    h_pairs = list(combinations(h, r=2))
    similarity_loss = mse_loss(*h_pairs[0])
    similarity_loss += mse_loss(*h_pairs[1])
    similarity_loss += mse_loss(*h_pairs[2])
    similarity_loss += mse_loss(*h_pairs[3])
    similarity_loss += mse_loss(*h_pairs[4])
    similarity_loss += mse_loss(*h_pairs[5])
    similarity_loss /= 6

    kl1, kl2, kl3, kl4 = kl_q_p_loss(h[0]),  kl_q_p_loss(h[1]),  kl_q_p_loss(h[2]), kl_q_p_loss(h[3])

    kl = (kl1 + kl2 + kl3 + kl4) / 4.0

    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    return k1 * output_loss + k2 * similarity_loss + k3 * kl

def get_p(q:torch.Tensor) -> torch.Tensor:
    p = q**2 / q.sum(0)
    return torch.autograd.Variable((p.t()/p.sum(1)).data, requires_grad=True)

def kl_q_p_loss(q:torch.Tensor):
    p = q**2 / q.sum(dim=0)
    p = p.t() / p.sum(dim=1)


    return max(torch.sum(q * torch.log(p.t()/q)), 0)

