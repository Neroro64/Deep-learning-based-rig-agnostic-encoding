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

import ray

def getFilesNames(file_paths, data_paths, MAX_FILES=-1):
    for data_path in data_paths:
        for dname, dirs, files in os.walk(data_path):
            for i, file in enumerate(files):
                file_paths.append(os.path.join(dname, file))
                if MAX_FILES > 0 and i >= MAX_FILES:
                    break
    return file_paths

r1_f = [
    "../../data/Dataset_R1_Two_1",
    "../../data/Dataset_R1_Two_1_small",
    "../../data/Dataset_R1_Two_2",
    "../../data/Dataset_R1_Two_2_small",
]
r2_f = [
    "../../data/Dataset_R2_Two_1",
    "../../data/Dataset_R2_Two_1_small",
    "../../data/Dataset_R2_Two_2",
    "../../data/Dataset_R2_Two_2_small",
]
r3_f = [
    "../../data/Dataset_R3_Two_1",
    "../../data/Dataset_R3_Two_1_small",
    "../../data/Dataset_R3_Two_2",
    "../../data/Dataset_R3_Two_2_small",
]
r4_f = [
    "../../data/Dataset_R4_Two_1",
    "../../data/Dataset_R4_Two_1_small",
    "../../data/Dataset_R4_Two_2",
    "../../data/Dataset_R4_Two_2_small",
]

r5_f = [
    "../../data/Dataset2_R5_Two_1",
    "../../data/Dataset2_R5_Two_1_small",
    "../../data/Dataset2_R5_Two_2",
    "../../data/Dataset2_R5_Two_2_small",
]

r1_l = [
    "../../data/Locomotion_R1-Loco"
]
r2_l = [
    "../../data/Locomotion_R2_Loco"
]
r3_l = [
    "../../data/Locomotion_R3_Loco"
]
r4_l = [
    "../../data/Locomotion_R4_Loco"
]
r5_l = [
    "../../data/Locomotion_R5_Loco"
]

r1_f_data = getFilesNames([], r1_f)
r2_f_data = getFilesNames([], r2_f)
r3_f_data = getFilesNames([], r3_f)
r4_f_data = getFilesNames([], r4_f)
r5_f_data = getFilesNames([], r5_f)
r_data = [r1_f_data, r2_f_data, r3_f_data, r4_f_data, r5_f_data]
# r_data = [r5_f_data]

r1_l_data = getFilesNames([], r1_l)
r2_l_data = getFilesNames([], r2_l)
r3_l_data = getFilesNames([], r3_l)
r4_l_data = getFilesNames([], r4_l)
r5_l_data = getFilesNames([], r5_l)
r_l_data = [r1_l_data, r2_l_data, r3_l_data, r4_l_data, r5_l_data]
# r_l_data = [r5_l_data]

phase_features = ["phase_vec_l2"]
pose_features = ["currentValue"]
cost_features = ["posCost", "rotCost"]
pose_label_feature = ["chainPos", "isLeft", "geoDistanceNormalised"]
target_features = ["targetPosition", "targetRotation"]
features = phase_features + pose_features + cost_features + target_features

names = ["F_R1","F_R2","F_R3","F_R4","F_R5"]
# names = ["F_R5"]
names2 = ["L_R1","L_R2","L_R3","L_R4","L_R5"]
# names2 = ["L_R5"]
ray.init(num_cpus=12)
for level in range(1):
    for data_paths, name in zip(r_data, names):
        d = F.process_data_multithread(data_paths, features, level=level, shutdown=False)
        # pose_labels = F.process_data_multithread([data_paths[0]], pose_label_feature, shutdown=False)
        obj = {"data": d, "features":features}
        F.remote_save.remote(obj, filename="{}_{}".format(level, name), path="../../datasets/")

for level in range(1):
    for data_paths, name in zip(r_l_data, names2):
        d = F.process_data_multithread(data_paths, features, level=level, shutdown=False)
        # pose_labels = F.process_data_multithread([data_paths[0]], pose_label_feature, shutdown=False)
        obj = {"data": d, "features":features}
        F.remote_save.remote(obj, filename="{}_{}".format(level, name), path="../../datasets/")

ray.shutdown()
