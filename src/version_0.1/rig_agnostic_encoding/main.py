import sys, os
# sys.path.append("functions")
# sys.path.append("models")
# sys.path.append("settings")
import functions.DataProcessingFunctions as Data
import functions.LossFunctions as Loss
import functions.TrainingFunctions as Training

from models.MIX4 import MIX4
from models.MIX4_withLabel import MIX4_withLabel
from models.MIX4_withLabel_withClusterParam import MIX4_withLabel_withClusterParam
from models.MIX4_withLabel_prob import MIX4_withLabel_prob

from models.MLP import MLP
from models.MLP_withLabel import MLP_withLabel
from models.RBF_withLabel import RBF_withLabel
from models.DEC_withLabel import DEC_withLabel
from models.VaDE_withLabel import VaDE_withLabel
from models.VAE_withLabel import VAE_withLabel
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as f

def main(argv):
    n_samples = 10
    # model = [MIX4_withLabel, MLP_withLabel]
    # model_name = "MLP4_withLabel_best"
    # model = [MIX4_withLabel, RBF_withLabel]
    # model_name = "RBF4_withLabel"
    # model = [MIX4_withLabel, DEC_withLabel]
    # model_name = "DEC4_withLabel3"
    # model = [MIX4_withLabel_prob, VaDE_withLabel]
    # model_name = "VaDE4_withLabel2"
    # model = [MIX4_withLabel_prob, VAE_withLabel]
    # model_name = "VAE4_withLabel_best"

    model = [MLP_withLabel]
    model_name = "MLP_withLabel"







    datapath1 = ["LOCO_R1-default-locomotion.pbz2",
                 "LOCO_R1-default-locomotion-small.pbz2",
                 "LOCO_R1-default-locomotion-large.pbz2"]
    datapath2 = ["LOCO_R3-default-locomotion.pbz2",
                 "LOCO_R3-default-locomotion-small.pbz2",
                 "LOCO_R3-default-locomotion-large.pbz2"]
    datapath3 = ["LOCO_R2-default-locomotion.pbz2",
                 "LOCO_R2-default-locomotion-small.pbz2",
                 "LOCO_R2-default-locomotion-large.pbz2"]
    datapath4 = ["LOCO_R4-default-locomotion.pbz2",
                 "LOCO_R4-default-locomotion-small.pbz2",
                 "LOCO_R4-default-locomotion-large.pbz2"]

    featureList = ["pos", "rotMat", "velocity", "isLeft", "chainPos", "geoDistanceNormalised"]
    # featureList = ["pos", "rotMat", "velocity"]

    extra_feature_len = 21 * 3
    extra_feature_len2 = 31 * 3

    config = get_config_mlp()
    # config = get_config_rbf()
    # config = get_config_dec()
    # config = get_config_vade()
    # config = get_config_vae()

    # withLabel
    # Training.train_multi_model_withLabel(model=model, datapaths=[datapath1,datapath2,datapath3,datapath4],
    #                                     featureList=featureList, config=config,
    #                                     extra_feature_len=extra_feature_len, extra_feature_len2=extra_feature_len2,
    #                                     n_samples=n_samples, n_epochs=800, model_name=model_name)

    # without label
    # Training.train_multi_model(model=model, datapaths=[datapath1,datapath2,datapath3,datapath4],
    #                                     featureList=featureList, config=config,
    #                                     n_samples=n_samples, model_name=model_name)

    Training.train_single_model_withLabel(model=model, datapaths=[datapath3], extra_feature_len=extra_feature_len,
                                featureList=featureList, config=config, n_epochs=300, n_samples=30,
                                model_name=model_name)

def get_config_mlp():
    return {
        "k": tune.choice([256, 512]),
        "hidden_dim": tune.choice([256, 512]),
        "lr": tune.loguniform(9e-5, 1e-5),
        "batch_size": tune.choice([1, 3, 5, 10]),
        "activation" : tune.choice([nn.ELU]),
        "loss_fn" : tune.choice([Loss.mse_mse_loss]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
    }


def get_config_rbf():
    return  {
        "k": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-7),
        "batch_size": tune.choice([5, 15, 30, 60]),
        "loss_fn": tune.choice([Loss.mse_mse_loss]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
        "activation" : tune.choice([nn.ELU, nn.ReLU, nn.Tanh]),
        "basis_func" : tune.choice(["gaussian", "linear", "quadratic", "inverse quadratic", "spline"]),
    }


def get_config_dec():
    return  {
        "k": tune.choice([128, 256, 512]),
        "hidden_dim": tune.choice([128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-6),
        "batch_size": tune.choice([1, 5, 10]),
        "loss_fn": tune.choice([Loss.mse_dec_kl_loss]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
        "activation" : tune.choice([nn.ELU, nn.ReLU, nn.Tanh]),
        "n_centroids": tune.choice([8, 16, 32, 64, 128, 256])
    }


def get_config_vade():
    return  {
        "k": tune.choice([128, 256, 512]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-7),
        "batch_size": tune.choice([5, 10, 15]),
        "loss_fn": tune.choice([Loss.mse_mse_loss]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
        "activation" : tune.choice([nn.ELU]),
        "n_centroids": tune.choice([8, 16, 32, 64, 128, 256])
    }


def get_config_vae():
    return  {
        "k": tune.choice([256, 512]),
        "hidden_dim": tune.choice([512]),
        "lr": tune.loguniform(1e-4, 1e-6),
        "batch_size": tune.choice([1, 3, 5, 10]),
        "loss_fn": tune.choice([Loss.mse_mse_loss]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
        "activation" : tune.choice([nn.ELU]),
    }




if __name__ == "__main__":
    argv = sys.argv
    main(argv)
