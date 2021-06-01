import os, sys
sys.path.append("motion_generation")
sys.path.append("rig_agnostic_encoding/functions")
sys.path.append("rig_agnostic_encoding/models")

from motion_generation.MoE import MoE
from motion_generation.MoE_Z import MoE as MoE_Z
import motion_generation
from motion_generation.GRU import GRU
from motion_generation.GRU_Z import GRU as GRU_Z
from motion_generation.LSTM import LSTM
from motion_generation.LSTM_Z import LSTM as LSTM_Z

from motion_generation.MotionGeneration import MotionGenerationModel as MoGen
from motion_generation.MotionGenerationEmbedd import MotionGenerationModel as MoGenZ
from motion_generation.MotionGenerationVAE import MotionGenerationModel as MoGenVAE
from motion_generation.MotionGenerationVAE_Embedd import MotionGenerationModel as MoGenVAE_Z

from MLP import MLP
from MLP_Adversarial import MLP_ADV
from MLP_MIX import MLP_MIX
from RBF import RBF
from VAE import VAE
from DEC import DEC

from rig_agnostic_encoding.functions.DataProcessingFunctions import clean_checkpoints
from GlobalSettings import MODEL_PATH
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

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import func as F
import _pickle as pickle
import json as js
import importlib
import random

def fit(model, name, version="0.1", MIN_EPOCHS=20, MAX_EPOCHS=100, useEarlyStopping=False, patience=10):
    if useEarlyStopping:
        earlystopping = EarlyStopping(monitor="avg_val_loss",patience=patience)
        callbacks = [earlystopping]
    else:
        callbacks = []
    logger=TensorBoardLogger(save_dir="RESULTS/", name=name, version=version)

    trainer = pl.Trainer(logger=logger, gpus=1, precision=16)
    res1 = trainer.test(model)

    trainer = pl.Trainer(
        default_root_dir="/home/nuoc/Documents/MEX/src/motion_generation/checkpoints",
        gpus=1, precision=16,
        callbacks= callbacks,
        min_epochs=MIN_EPOCHS,
        logger=logger,
        max_epochs=MAX_EPOCHS,
        # stochastic_weight_avg=True
    )

    trainer.fit(model)
    res2 = trainer.test(model)
    return [res1, res2]


config = {
    "hidden_dim": 256,
    "k": 256,
    "z_dim": 256,
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
    "autoregress_inc":.3,
    "autoregress_ep":20,
    "autoregress_max_prob":1,
    "cost_hidden_dim":128,
    "seq_len":13,
    "device":"cuda"
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

for dname, dirs, files in os.walk("/home/nuoc/Documents/MEX/datasets/"):
    for i, file in enumerate(files):
        data_paths.append(os.path.join(dname, file))

RESULTS = {"models":[], "size":[], "parameters":[], "test_RESULTS":[], "model_path":[]}

for path in data_paths:
    tokens = path.split("/")
    tokens = tokens[-1].split("_")
    level = tokens[0]
    name = tokens[1] + "_" + tokens[2]
    name = name.replace(".pbz2","")

    obj = F.load(path)
    data = obj["data"]

    feature_dims = data[0][1]
    clips = [np.copy(d[0]) for d in data]

    phase_dim = sum([feature_dims[feature] for feature in phase_features])
    pose_dim = sum([feature_dims[feature] for feature in pose_features])
    cost_dim = sum([feature_dims[feature] for feature in cost_features])
    target_dim = sum([feature_dims[feature] for feature in target_features])
    x_tensors = torch.stack([F.normaliseT(torch.from_numpy(clip[:-1])).float() for clip in clips])
    y_tensors = torch.stack([torch.from_numpy(clip[1:]).float() for clip in clips])

    pose_data = x_tensors[:, :, phase_dim:phase_dim + pose_dim]
    dataset_p = TensorDataset(pose_data, pose_data)
    N = len(x_tensors)

    train_ratio = int(.8 * N)
    val_ratio = int((N - train_ratio) / 2.0)
    test_ratio = N - train_ratio - val_ratio
    train_set_p, val_set_p, test_set_p = random_split(dataset_p, [train_ratio, val_ratio, test_ratio],
                                                      generator=torch.Generator().manual_seed(2021))

    test_set_p += val_set_p
    h_dim = train_set_p[0][0].shape[0]
    w_dim = train_set_p[0][0].shape[1]

    sizes = [64, 128, 256]
    ae_name = "AE_"+level+"_"+name
    try:
        for size in sizes:
            config["hidden_dim"] = size
            config["k"] = size

            ae = MLP_ADV(config=config, dimensions=[pose_dim], h_dim=h_dim, w_dim=w_dim,
                         pos_dim=feature_dims["pos"], rot_dim=feature_dims["rotMat2"], vel_dim=feature_dims["velocity"],
                         train_set=train_set_p, val_set=val_set_p, test_set=test_set_p, name=ae_name)

            RESULTS["models"].append(ae_name)
            RESULTS["size"].append(size)
            RESULTS["parameters"].append(ae.summarize())

            res = fit(ae, name=ae_name, version=str(size), MAX_EPOCHS=200)
            p = ae.save_checkpoint(best_val_loss=0.0001)

            RESULTS["test_RESULTS"].append(res)
            RESULTS["model_path"].append(p)
    except:
        F.save(RESULTS, "AE", "/home/nuoc/Documents/MEX/RESULTS/")

    clean_checkpoints(path=os.path.join(MODEL_PATH,ae_name))
F.save(RESULTS, "AE", "/home/nuoc/Documents/MEX/results/")

