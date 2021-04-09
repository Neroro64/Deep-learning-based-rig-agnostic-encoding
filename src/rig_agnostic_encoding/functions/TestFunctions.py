import torch
import sys, os
import pandas as pd
import numpy as np
import functions.DataProcessingFunctions as Data
import functions.LossFunctions as Loss
from itertools import combinations_with_replacement

def test_recon_error_withLabel(mix_model, model, model_path, data_path,
                               save=True, save_path=""):
    M = mix_model.load_checkpoint(model, model_path)
    models = [M.model1, M.model2, M.model3, M.model4]
    test_data = Data.load(data_path)["data"]
    with torch.no_grad():
        recon_error_matrix = np.zeros((4,4))
        latent_error_matrix = np.zeros((4,4))

        pairs = list(combinations_with_replacement(np.arange(len(test_data)), r=2))
        for (i, j) in pairs:
            for x1, y1 in test_data[i]:
                for x2, y2 in test_data[j]:
                    h1, l1 = models[i].encode(x1)
                    h2, l2 = models[j].encode(x2)

                    out11 = models[i].decode(h1, l1)
                    out12 = models[i].decode(h2, l1)
                    out22 = models[j].decode(h2, l2)
                    out21 = models[j].decode(h1, l2)

                    recon_loss_xx = Loss.mse_loss(out11, y1)
                    recon_loss_yy = Loss.mse_loss(out22, y2)
                    recon_loss_xy = Loss.mse_loss(out12, y1)
                    recon_loss_yx = Loss.mse_loss(out21, y2)

                    latent_loss = Loss.mse_loss(h1, h2)

                    recon_error_matrix[i,i] = recon_loss_xx
                    recon_error_matrix[i,j] = recon_loss_xy
                    recon_error_matrix[j,j] = recon_loss_yy
                    recon_error_matrix[j,i] = recon_loss_yx
                    latent_error_matrix[i,j] = latent_error_matrix[j,i] = latent_loss
    if save:
        if not os.path.exists(save_path): os.mkdir(save_path)
        names = ["M1", "M2", "M3", "M4"]
        recon_df = pd.DataFrame(data=recon_error_matrix, index=names, columns=names)
        latent_df = pd.DataFrame(data=latent_error_matrix, index=names, columns=names)
        recon_df.to_csv(M.name + "_recon_error.csv")
        latent_df.to_csv(M.name + "_latent_error.csv")

def test_recon_error(mix_model, model, model_path, data_path,
                                save=True, save_path=""):
    M = mix_model.load_checkpoint(model, model_path)
    models = [M.model1, M.model2, M.model3, M.model4]
    test_data = Data.load(data_path)["data"]
    with torch.no_grad():
        recon_error_matrix = np.zeros((4,4))

        pairs = list(combinations_with_replacement(np.arange(len(test_data)), r=2))
        for (i, j) in pairs:
            for x1, y1 in test_data[i]:
                for x2, y2 in test_data[j]:
                    h1 = models[i].encode(x1)
                    h2 = models[j].encode(x2)

                    out11 = models[i].decode(h1)
                    out12 = models[i].decode(h2)
                    out22 = models[j].decode(h2)
                    out21 = models[j].decode(h1)

                    recon_loss_xx = Loss.mse_loss(out11, y1)
                    recon_loss_yy = Loss.mse_loss(out22, y2)
                    recon_loss_xy = Loss.mse_loss(out12, y1)
                    recon_loss_yx = Loss.mse_loss(out21, y2)

                    recon_error_matrix[i,i] = recon_loss_xx
                    recon_error_matrix[i,j] = recon_loss_xy
                    recon_error_matrix[j,j] = recon_loss_yy
                    recon_error_matrix[j,i] = recon_loss_yx
    if save:
        if not os.path.exists(save_path): os.mkdir(save_path)
        names = ["M1", "M2", "M3", "M4"]
        recon_df = pd.DataFrame(data=recon_error_matrix, index=names, columns=names)
        recon_df.to_csv(M.name + "_recon_error.csv")


def test_recon_error_withLabel_withProb(mix_model, model, model_path, data_path,
                               save=True, save_path=""):
    M = mix_model.load_checkpoint(model, model_path)
    models = [M.model1, M.model2, M.model3, M.model4]
    test_data = Data.load(data_path)["data"]
    with torch.no_grad():
        recon_error_matrix = np.zeros((4,4))
        latent_error_matrix = np.zeros((4,4))
        kl_error_matrix = np.zeros((4,4))
        error_matrix = np.zeros((4,4))
        pairs = list(combinations_with_replacement(np.arange(len(test_data)), r=2))
        for (i, j) in pairs:
            for x1, y1 in test_data[i]:
                for x2, y2 in test_data[j]:
                    h1, l1, mu1, logvar1 = models[i].encode(x1)
                    h2, l2, mu2, logvar2 = models[j].encode(x2)

                    out11, _, _, _ = models[i].decode(h1, l1, mu1, logvar1)
                    out12, _, _, _ = models[i].decode(h2, l1, mu2, logvar2)
                    out22, _, _, _ = models[j].decode(h2, l2, mu2, logvar2)
                    out21, _, _, _ = models[j].decode(h1, l2, mu1, logvar1)

                    loss_xx, kl_loss_xx, recon_loss_xx = Loss.kl_loss(out11, y1, mu1, logvar1)
                    loss_yy, kl_loss_yy, recon_loss_yy = Loss.kl_loss(out22, y2, mu2, logvar2)
                    loss_xy, kl_loss_xy, recon_loss_xy = Loss.kl_loss(out12, y1, mu1, logvar1)
                    loss_yx, kl_loss_yx, recon_loss_yx = Loss.kl_loss(out21, y2, mu2, logvar2)

                    latent_loss = Loss.mse_loss(h1, h2)

                    recon_error_matrix[i,i] = recon_loss_xx
                    recon_error_matrix[i,j] = recon_loss_xy
                    recon_error_matrix[j,j] = recon_loss_yy
                    recon_error_matrix[j,i] = recon_loss_yx

                    error_matrix[i, i] = loss_xx
                    error_matrix[i, j] = loss_xy
                    error_matrix[j, j] = loss_yy
                    error_matrix[j, i] = loss_yx

                    latent_error_matrix[i,j] = latent_error_matrix[j,i] = latent_loss
                    kl_error_matrix[i,j] =  kl_loss_xy
                    kl_error_matrix[j, i] = kl_loss_yx

    if save:
        if not os.path.exists(save_path): os.mkdir(save_path)
        names = ["M1", "M2", "M3", "M4"]
        recon_df = pd.DataFrame(data=recon_error_matrix, index=names, columns=names)
        latent_df = pd.DataFrame(data=latent_error_matrix, index=names, columns=names)
        error_df = pd.DataFrame(data=error_matrix, index=names, columns=names)
        kl_df = pd.DataFrame(data=kl_error_matrix, index=names, columns=names)
        recon_df.to_csv(M.name + "_recon_error.csv")
        latent_df.to_csv(M.name + "_latent_error.csv")
        error_df.to_csv(M.name + "_error.csv")
        kl_df.to_csv(M.name + "_kl_error.csv")



def test_recon_error_withProb(mix_model, model, model_path, data_path,
                               save=True, save_path=""):
    M = mix_model.load_checkpoint(model, model_path)
    models = [M.model1, M.model2, M.model3, M.model4]
    test_data = Data.load(data_path)["data"]
    with torch.no_grad():
        recon_error_matrix = np.zeros((4,4))
        latent_error_matrix = np.zeros((4,4))
        kl_error_matrix = np.zeros((4,4))
        error_matrix = np.zeros((4,4))

        pairs = list(combinations_with_replacement(np.arange(len(test_data)), r=2))
        for (i, j) in pairs:
            for x1, y1 in test_data[i]:
                for x2, y2 in test_data[j]:
                    h1, mu1, logvar1 = models[i].encode(x1)
                    h2, mu2, logvar2 = models[j].encode(x2)

                    out11 = models[i].decode(h1, mu1, logvar1)
                    out12 = models[i].decode(h2, mu2, logvar2)
                    out22 = models[j].decode(h2, mu2, logvar2)
                    out21 = models[j].decode(h1, mu1, logvar1)

                    loss_xx, kl_loss_xx, recon_loss_xx = Loss.kl_loss(out11, y1, mu1, logvar1)
                    loss_yy, kl_loss_yy, recon_loss_yy = Loss.kl_loss(out22, y2, mu2, logvar2)
                    loss_xy, kl_loss_xy, recon_loss_xy = Loss.kl_loss(out12, y1, mu1, logvar1)
                    loss_yx, kl_loss_yx, recon_loss_yx = Loss.kl_loss(out21, y2, mu2, logvar2)

                    latent_loss = Loss.mse_loss(h1, h2)

                    recon_error_matrix[i,i] = recon_loss_xx
                    recon_error_matrix[i,j] = recon_loss_xy
                    recon_error_matrix[j,j] = recon_loss_yy
                    recon_error_matrix[j,i] = recon_loss_yx

                    error_matrix[i, i] = loss_xx
                    error_matrix[i, j] = loss_xy
                    error_matrix[j, j] = loss_yy
                    error_matrix[j, i] = loss_yx

                    latent_error_matrix[i,j] = latent_error_matrix[j,i] = latent_loss
                    kl_error_matrix[i,j] =  kl_loss_xy
                    kl_error_matrix[j, i] = kl_loss_yx

    if save:
        if not os.path.exists(save_path): os.mkdir(save_path)
        names = ["M1", "M2", "M3", "M4"]
        recon_df = pd.DataFrame(data=recon_error_matrix, index=names, columns=names)
        latent_df = pd.DataFrame(data=latent_error_matrix, index=names, columns=names)
        error_df = pd.DataFrame(data=error_matrix, index=names, columns=names)
        kl_df = pd.DataFrame(data=kl_error_matrix, index=names, columns=names)
        recon_df.to_csv(M.name + "_recon_error.csv")
        latent_df.to_csv(M.name + "_latent_error.csv")
        error_df.to_csv(M.name + "_error.csv")
        kl_df.to_csv(M.name + "_kl_error.csv")
