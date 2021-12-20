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
import Extract as ext
import plotly.graph_objs as go
import plotly.express as ex
from plotly.subplots import make_subplots
import scipy.signal as signal
obj = F.load("G:/MEX/src/True_3_20.pbz2")
clip = pickle.loads(obj)

frames = clip["frames"]
keyJ = [i for i in range(len(frames[0])) if frames[0][i]["key"]]
contact_1 =  np.concatenate([frames[i][keyJ[-2]]["contact"] for i in range(len(frames))])
contact_2 =  np.concatenate([frames[i][keyJ[-1]]["contact"] for i in range(len(frames))])


block_fn_1 = contact_1.T
block_fn_2 = contact_2.T

def norm(block_fn):
    window_size_half = 15
    Frames = block_fn.shape[0]
    t = np.arange(Frames)
    normalised_block_fn = np.zeros_like(block_fn, dtype=np.float32)

    for ts in t:
        low = max(ts-window_size_half, 0)
        high = min(ts+window_size_half, Frames)
        window = np.arange(low, high)
        if len(window) < window_size_half*2:
            window = np.pad(window, (window_size_half*2-len(window),), mode='edge')
        slice = block_fn[window]
        mean = np.mean(slice)
        std = np.std(slice)
        std = 1 if std == 0 else std
        normalised_block_fn[ts] = (block_fn[ts]-mean) / std
    return normalised_block_fn

norm_block_fn_1 = norm(block_fn_1)
norm_block_fn_2 = norm(block_fn_2)

filter = signal.butter(3, .1, "low", analog=False, output="sos")
filtered_1 = signal.sosfilt(filter, norm_block_fn_1)
filtered_2 = signal.sosfilt(filter, norm_block_fn_2)

sin_norm_1 = np.sin(filtered_1)
sin_norm_2 = np.sin(filtered_2)

cos_norm_1 = np.cos(filtered_1)
cos_norm_2 = np.cos(filtered_2)

delta_sin_1 = np.diff(sin_norm_1, prepend=0)
delta_cos_1 = np.diff(cos_norm_1, prepend=0)
delta_sin_2 = np.diff(sin_norm_2, prepend=0)
delta_cos_2 = np.diff(cos_norm_2, prepend=0)
delta_cos_1[0]=0
delta_cos_2[0]=0

phase_vec_1 = (sin_norm_1*delta_sin_1, cos_norm_1*delta_cos_1)
phase_vec_2 = (sin_norm_2*delta_sin_2, cos_norm_2*delta_sin_2)

def plot_contact(original, normalised, filtered, local_phase, amp):
    fig = make_subplots(rows=4, cols=1, subplot_titles=["Original", "Normalised", "Filtered", "Local phase vector"])
    x = np.arange(len(original[0]))

    for i in range(1):
        fig.add_trace(go.Scatter(x=x, y=original[i], name="J"+str(i)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=normalised[i], name="J"+str(i)), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=filtered[i], name="J"+str(i)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=local_phase[i][0], name="J"+str(i)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=local_phase[i][1], name="J"+str(i)), row=4, col=1)
        fig.add_trace(go.Bar(x=x, y=amp[i][0], name="delta sin(filtered)"), row=3, col=1)
        fig.add_trace(go.Bar(x=x, y=amp[i][1], name="delta cos(filtered)"), row=3, col=1)


    fig.update_layout(
        title="Local motion phase computation",
        width=800,
        height=800,
        showlegend=False
    )
    fig.update_xaxes(title_text="frames", row=4, col=1)
    return fig


block_fn = [block_fn_1, block_fn_2]
normalised_block_fn = [norm_block_fn_1, norm_block_fn_2]
filtered = [filtered_1, filtered_2]
phase_vec = [phase_vec_1, phase_vec_2]
amp = [(delta_sin_1, delta_cos_1), (delta_sin_2, delta_cos_2)]

fig = plot_contact(block_fn, normalised_block_fn, filtered, phase_vec, amp)

fig.write_image("lmp_computation.png")
