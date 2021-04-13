import sys, os
# sys.path.append("functions")
# sys.path.append("models")
# sys.path.append("settings")
import functions.DataProcessingFunctions as Data
import functions.LossFunctions as Loss
import functions.TrainingFunctions as Training
import functions.TestFunctions as Test

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
    # model = [MIX4_withLabel, MLP_withLabel]
    # model_name = "MLP4_withLabel"
    # model = [MIX4_withLabel, RBF_withLabel]
    # model_name = "RBF4_withLabel"
    # model = [MIX4_withLabel, DEC_withLabel]
    # model_name = "DEC4_withLabel3"
    # model = [MIX4_withLabel_prob, VaDE_withLabel]
    # model_name = "VaDE4_withLabel2"
    # model = [MIX4_withLabel_prob, VAE_withLabel]
    # model_name = "VAE4_withLabel_best"
    model = [MLP_withLabel]





    datapath = "/home/nuoc/Documents/MEX/models/MLP_withLabeltest_data.pbz2"
    modelPath = "/home/nuoc/Documents/MEX/models/MLP_withLabel/0.0013522337.512.pbz2"


    # Test.test_recon_error_withLabel(mix_model=model[0], model=model[1], model_path=modelPath,
    #                                 data_path=datapath, save=False, save_path="../../results")

    # Test.test_recon_error_withLabel_withProb(mix_model=model[0], model=model[1], model_path=modelPath,
    #                                          data_path=datapath, save=True, save_path="../../results")

    Test.test_single_model(model=model[0], model_path=modelPath, data_path=datapath, save=True, save_path="../../results")

if __name__ == "__main__":
    argv = sys.argv
    main(argv)
