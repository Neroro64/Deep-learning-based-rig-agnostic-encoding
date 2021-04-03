import sys, os
# sys.path.append("functions")
# sys.path.append("models")
# sys.path.append("settings")
import functions.DataProcessingFunctions as Data
import functions.LossFunctions as Loss
import functions.TrainingFunctions as Training
from models.MLP import MLP
from models.MLP_withLabel import MLP_withLabel
from models.MIX4 import MIX4
from models.MIX4_withLabel import MIX4_withLabel
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as f

def main(argv):

    model = [MIX4_withLabel, MLP_withLabel]

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

    extra_feature_len = 21 * 3
    extra_feature_len2 = 31 * 3

    config = {
        "k": tune.randint(3, 256),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(1e-2, 1e-7),
        "batch_size": tune.choice([5, 15, 30, 60]),
        "loss_fn": tune.choice([Loss.mse_mse_loss, Loss.mse_mae_loss, Loss.mae_mae_loss, Loss.mse_kl_loss, Loss.mse_mae_loss2]),
        "ae_loss_fn": tune.choice([Loss.mse_loss]),
        "activation" : tune.choice([nn.ELU, nn.ReLU, nn.Tanh])
    }


    Training.train_multi_model_withLabel(model=model, datapaths=[datapath1,datapath2,datapath3,datapath4],
                                        featureList=featureList, config=config,
                                        extra_feature_len=extra_feature_len, extra_feature_len2=extra_feature_len2,
                                        n_samples=1, model_name="MIX4_EXTRA")


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
