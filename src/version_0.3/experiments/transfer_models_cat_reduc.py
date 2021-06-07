import os, sys
sys.path.append("motion_generation")
sys.path.append("rig_agnostic_encoding/functions")
sys.path.append("rig_agnostic_encoding/models")

from MoE import MoE
from MoE_Z import MoE as MoE_Z
import motion_generation
from GRU import GRU
from GRU_Z import GRU as GRU_Z
from LSTM import LSTM
from LSTM_Z import LSTM as LSTM_Z

from MotionGeneration import MotionGenerationModel as MoGen
from MotionGenerationR import MotionGenerationModel as MoGenR

from MotionGenerationEmbedd import MotionGenerationModel as MoGenZ
from MotionGenerationEmbeddR import MotionGenerationModel as MoGenZR

from MotionGenerationVAE import MotionGenerationModel as MoGenVAE
from MotionGenerationVAER import MotionGenerationModel as MoGenVAER

from MotionGenerationVAE_Embedd import MotionGenerationModel as MoGenVAE_Z
from MotionGenerationVAE_EmbeddR import MotionGenerationModel as MoGenVAE_ZR

from MLP import MLP
from MLP_Adversarial import MLP_ADV
from MLP_MIX import MLP_MIX
from MLP_MIX import MLP_layer
from RBF import RBF
from VAE import VAE
from DEC import DEC

from rig_agnostic_encoding.functions.DataProcessingFunctions import clean_checkpoints
from GlobalSettings import MODEL_PATH, RESULTS_PATH
import bz2
from cytoolz import concat, sliding_window, accumulate
from operator import add
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import func as F
import _pickle as pickle
import json as js
import importlib
import random
import traceback
import time

def fit(model, name, version="0.1", MIN_EPOCHS=20, MAX_EPOCHS=100, useEarlyStopping=False, patience=10):
    if useEarlyStopping:
        earlystopping = EarlyStopping(monitor="avg_val_loss",patience=patience)
        callbacks = [earlystopping]
    else:
        callbacks = []
    logger=TensorBoardLogger(save_dir="results/", name=name, version=version)

    trainer = pl.Trainer(logger=logger, gpus=1, precision=16)
    trainer.test(model)

    trainer = pl.Trainer(
        gpus=1, precision=16,
        min_epochs=MIN_EPOCHS,
        logger=logger,
        max_epochs=MAX_EPOCHS,
        stochastic_weight_avg=True,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                },
                on="validation_end")
            ]
    )

    trainer.fit(model)
    trainer.test(model)
    del model

def train_all_models(path, RESULTS):
    path = path.replace("\\", "/")
    tokens = path.split("/")
    tokens = tokens[-1].split("_")
    level = tokens[0]
    name = tokens[1] + "_" + tokens[2]
    name = name.replace(".pbz2", "")

    if level != "0":

        path2 = path.replace(level + "_F", "0_F")
        obj = F.load(path)
        obj2 = F.load(path2)

        data = obj["data"]
        data2 = obj2["data"]

        feature_dims = data[0][1]
        feature_dims2 = data2[0][1]
        clips = [np.copy(d[0]) for d in data]
        clips2 = [np.copy(d[0]) for d in data2]

        phase_dim = sum([feature_dims[feature] for feature in phase_features])
        pose_dim = sum([feature_dims[feature] for feature in pose_features])
        pose_dim2 = sum([feature_dims2[feature] for feature in pose_features])
        cost_dim = sum([feature_dims[feature] for feature in cost_features])
        target_dim = sum([feature_dims[feature] for feature in target_features])

        x_tensors = torch.stack([F.normaliseT(torch.from_numpy(clip[:-1])).float() for clip in clips])
        y_tensors = torch.stack([torch.from_numpy(clip[1:]).float() for clip in clips2])

        ae_path = "G:/MEX/models/AE_0_F_R1/0.0001.256.pbz2"
        ae_path1 = ae_path.replace("0_F_R1", level + "_" + name)
        ae_path2 = ae_path.replace("0_F_R1", "0_" + name)
        ae = MLP_ADV.load_checkpoint(ae_path1)
        ae2 = MLP_ADV.load_checkpoint(ae_path2)

        ae.decoder = ae2.decoder

        pose_label_name = "pose_label"+name[-1]+".pbz2"
        pose_label = F.load(pose_label_name)

        pose_label = np.concatenate([np.repeat(pose_label, 3), np.repeat(pose_label, 6), np.repeat(pose_label, 3)]).ravel()
        pose_label = np.where(pose_label > 0)[0]

        pose_label = torch.from_numpy(pose_label).to(config["device"])
        config["use_label"] = True
        config["pose_label"] = pose_label

    else:

        obj = F.load(path)
        data = obj["data"]

        feature_dims = data[0][1]
        clips = [np.copy(d[0]) for d in data]

        phase_dim = sum([feature_dims[feature] for feature in phase_features])
        pose_dim = sum([feature_dims[feature] for feature in pose_features])
        pose_dim2 = pose_dim
        cost_dim = sum([feature_dims[feature] for feature in cost_features])
        target_dim = sum([feature_dims[feature] for feature in target_features])
        x_tensors = torch.stack([F.normaliseT(torch.from_numpy(clip[:-1])).float() for clip in clips])
        y_tensors = torch.stack([torch.from_numpy(clip[1:]).float() for clip in clips])

        ae_path = "G:/MEX/models/AE_0_F_R1/0.0001.256.pbz2"
        ae_path = ae_path.replace("0_F_R1", level + "_" + name)
        ae = MLP_ADV.load_checkpoint(ae_path)

    dataset_p = TensorDataset(x_tensors, y_tensors)
    N = len(x_tensors)

    train_ratio = int(.2 * N)
    val_ratio = int((N - train_ratio) / 2.0)
    test_ratio = N - train_ratio - val_ratio
    train_set_p, val_set_p, test_set_p = random_split(dataset_p, [train_ratio, val_ratio, test_ratio],
                                                      generator=torch.Generator().manual_seed(2021))
    test_set_p += val_set_p

    C_models = ["RBF", "VAE", "DEC"]
    # C_models = ["None"]
    G_models = ["MoE_Z"]
    config["C"] = tune.grid_search(C_models)
    config["G"] = tune.grid_search(G_models)
    name = level + "_" + "R1_to_"+name + "_ZCAT_reduced_trainable"
    # G_models = [MoE, GRU, LSTM, MoE_Z, GRU_Z, LSTM_Z]
    # G_models = [MoE, LSTM]
    # G_models = [MoE_Z, GRU_Z, LSTM_Z]
    # G_models = [LSTM, GRU, LSTM, MoE_Z, GRU_Z, LSTM_Z]

    in_slice = [phase_dim, pose_dim, cost_dim, target_dim]
    out_slice = [phase_dim, config["k"], cost_dim]
    featureDim = {
        "phase_dim": phase_dim,
        "pose_dim": pose_dim2,
        "pose_dim2": pose_dim,
        "cost_dim": cost_dim,
        "target_dim": target_dim,
        "g_input_dim": config["z_dim"] + config["cost_hidden_dim"],
        "g_output_dim": phase_dim + config["k"] + cost_dim,
        "pos_dim": feature_dims["pos"],
        "rot_dim": feature_dims["rotMat2"],
        "vel_dim": feature_dims["velocity"],
        "posCost": feature_dims["posCost"],
        "rotCost": feature_dims["rotCost"]
    }


    reporter = CLIReporter(
        parameter_columns=["C", "G", "hidden_dim", "k", "z_dim"],
        metric_columns=["loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_model,
            name=name,
            featureDim=featureDim,
            feature_dims=feature_dims,
            train_set=train_set_p,
            val_set=val_set_p,
            test_set=test_set_p,
            RESULTS=RESULTS,
            ae=ae,
            pose_dim=pose_dim,
            in_slice=in_slice,
            out_slice=out_slice
        ),
        resources_per_trial={
            "cpu": 4,
            "gpu": 1
            },
        metric="loss",
        mode="min",
        config=config,
        num_samples=1,
        progress_reporter=reporter,
        name=name)
    del analysis, dataset_p, train_set_p, val_set_p, test_set_p

def getModel(model_name):
    if model_name=="None": return None
    elif model_name=="RBF": return RBF
    elif model_name=="VAE": return VAE
    elif model_name=="DEC": return DEC
    elif model_name=="MoE": return MoE
    elif model_name=="LSTM": return LSTM
    elif model_name=="MoE_Z": return MoE_Z
    elif model_name=="LSTM_Z": return LSTM_Z

def loadMoGen(path, Model, C_model):
    obj = F.load(path)

    cost_encoder = MLP.load_checkpoint(obj["cost_encoder_path"])
    generationModel = Model.load_checkpoint(obj["motionGenerationModelPath"])

    if C_model is not None:
        C_model.load_state_dict(obj["middle_layer_dict"])

    return cost_encoder, generationModel, C_model

def train_model(config, name,featureDim, feature_dims,
                train_set, val_set, test_set,
                RESULTS, ae, pose_dim, in_slice, out_slice):


    C = getModel(config["C"])
    G = getModel(config["G"])


    # if i > 0:
    ref_name = "_256_ZCAT_0.1"
    featureDim["g_input_dim"] = config["k"] + config["cost_hidden_dim"]
    if C == VAE:
        MoGenNet = MoGenVAE_ZR
    else:
        MoGenNet = MoGenZR
    # else:
    # ref_name = "_256_ZIN_0.1"
    # if C == VAE:
    #     MoGenNet = MoGenVAE
    # else:
    #     MoGenNet = MoGen

    if C is None:
        model_name = "AE_" + G.__name__ + ref_name + name
        featureDim["g_input_dim"] = config["k"] + config["cost_hidden_dim"]
    else:
        model_name = C.__name__ + "_" + G.__name__ + ref_name + name

    pretrained_model = "G:/MEX/models/RBF_MoE_256_ZCAT2_F_R1_ZCAT/final.pbz2"
    pretrained_model = pretrained_model.replace("RBF", config["C"])
    # h_dim = ae.h_dim
    # w_dim = ae.w_dim

    pose_auto_encoder = ae
    # pose_auto_encoder = MLP_ADV(config=config, dimensions=[pose_dim], h_dim=h_dim, w_dim=w_dim,
    #                             pos_dim=feature_dims["pos"], rot_dim=feature_dims["rotMat2"],
    #                             vel_dim=feature_dims["velocity"])
    #
    # pose_auto_encoder.encoder.load_state_dict(ae.encoder.state_dict())
    # pose_auto_encoder.decoder.load_state_dict(ae.decoder.state_dict())
    # pose_auto_encoder.convDiscriminator.load_state_dict(ae.convDiscriminator.state_dict())
    pose_dim = featureDim["pose_dim"]
    c_layer = C(config=config, input_dims=[pose_dim]).cluster_model if C is not None else MLP_layer()
    c_layer_2 = C(config=config, input_dims=[72]).cluster_model if C is not None else MLP_layer()

    cost_encoder, generationModel, C_model = loadMoGen(pretrained_model, Model=G, C_model=c_layer_2)
    model = MoGenNet(config=config, Model=G, pose_autoencoder=pose_auto_encoder, middle_layer=c_layer,
                     feature_dims=featureDim, use_advLoss=True,
                     input_slicers=in_slice, output_slicers=out_slice,
                     train_set=train_set, val_set=val_set, test_set=test_set,
                     name=model_name, workers=4
                     )

    # model.generationModel.load_state_dict(pretrained_model.generationModel.state_dict())
    # model.generationModel.gate.load_state_dict(pretrained_model.generationModel.gate.state_dict())
    # model.middle_layer.load_state_dict(pretrained_model.middle_layer.state_dict())
    model.generationModel = generationModel
    model.cost_encoder = cost_encoder
    model.middle_layer = C_model
    # model.middle_layer.requires_grad_(False)
    # model.cost_encoder.requires_grad_(False)
    # model.generationModel.gate.requires_grad_(False)
    # model.generationModel.requires_grad_(False)

    RESULTS["models"].append(model_name)
    RESULTS["parameters"].append(model.summarize())

    fit(model, name=model_name, version="Transferred", MAX_EPOCHS=30)

    clean_checkpoints(path=os.path.join(MODEL_PATH, model_name))
    model.save_checkpoint(best_val_loss="final")


config = {
    "hidden_dim": 256,
    "k": 256,
    "z_dim": 128,
    "lr": 1e-4,
    "batch_size": 32,
    "keep_prob": 0,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.AdamW,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":80, "gamma":.9},
    "basis_func":"gaussian",
    "n_centroid":64,
    "k_experts": 4,
    "gate_size": 128,
    "g_hidden_dim": 512,
    "num_layers": 4,
    "autoregress_prob":0,
    "autoregress_inc":.5,
    "autoregress_ep":10,
    "autoregress_max_prob":1,
    "cost_hidden_dim":128,
    "seq_len":13,
    "device":"cuda",
    "use_label":False
    }

phase_features = ["phase_vec_l2"]
pose_features = ["pos", "rotMat2", "velocity"]
cost_features = ["posCost", "rotCost"]
pose_label_feature = ["chainPos", "isLeft", "geoDistanceNormalised"]
target_features = ["targetPosition", "targetRotation"]
data_paths = []

# template_1 = js.load(open("/home/nuoc/Documents/MEX/src/version_0.3/R1_template.json"))
# template_2 = js.load(open("/home/nuoc/Documents/MEX/src/version_0.3/R2_template.json"))
# template_3 = js.load(open("/home/nuoc/Documents/MEX/src/version_0.3/R3_template.json"))
# template_4 = js.load(open("/home/nuoc/Documents/MEX/src/version_0.3/R4_template.json"))
# template_5 = js.load(open("/home/nuoc/Documents/MEX/src/version_0.3/R5_template.json"))
def main():
    try:
        for dname, dirs, files in os.walk("G:/MEX/datasets"):
            for i, file in enumerate(files):
                if "L" in file or "1_F" in file or "0_F" in file or "R1" in file:
                    continue
                data_paths.append(os.path.join(dname, file))

        RESULTS = {"models":[], "size":[], "parameters":[], "test_RESULTS":[], "model_path":[]}
        task_id = 0
        start_time = time.time()
        # try:
        for i in range(0, len(data_paths)):
            path = data_paths[i]
            train_all_models(path, RESULTS)

        RESULTS["total_time"] = time.time() - start_time
        F.save(RESULTS, "Reference_results", RESULTS_PATH)
        print("Done: ", RESULTS["total_time"])

    except Exception as err:
        F.save(data_paths[i:], "remaining_paths", RESULTS_PATH)
        F.save(RESULTS, "Reference_results2", RESULTS_PATH)
        print(err)
        traceback.print_exc()

if __name__ == '__main__':
    main()