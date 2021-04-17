


import os, sys, math, time
import numpy as np
import numpy.linalg as la
import plotly.graph_objects as go
import plotly.express as ex
from plotly.subplots import make_subplots
import pandas as pd

import json as js
import _pickle as pickle
import bz2
import ray

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from collections import OrderedDict
from cytoolz import sliding_window, accumulate
import pytorch_lightning as pl
from operator import add
from tabulate import tabulate

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

import ray
import ray.tune as tune


sys.path.append("../")
sys.path.append("../rig_agnostic_encoding")
sys.path.append("../rig_agnostic_encoding/models")
sys.path.append("../rig_agnostic_encoding/functions")
import func
# from MLP_withLabel import MLP_withLabel
# from MLP import MLP
from DataProcessingFunctions import clean_checkpoints


DATA_PATH = "/home/nuoc/Documents/MEX/data"
MODEL_PATH = "/home/nuoc/Documents/MEX/models"
RESULTS_PATH = "/home/nuoc/Documents/MEX/results"




class MLP_withLabel(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, extra_feature_len:int=0,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(MLP_withLabel, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module
        self.extra_feature_len = extra_feature_len
        self.act = nn.ELU
        self.k = 0
        if load:
            self.build()
        else:
            self.hidden_dim = config["hidden_dim"]
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.act = config["activation"]
            self.loss_fn = config["ae_loss_fn"]
            self.batch_size = config["batch_size"]

            self.dimensions = [self.dimensions[0]-extra_feature_len, self.hidden_dim, self.k]
            self.train_set = train_set
            self.val_set = val_set
            self.test_set = test_set

            self.best_val_loss = np.inf

            self.build()
            self.encoder.apply(self.init_params)
            self.decoder.apply(self.init_params)


    def build(self):
        layer_sizes = list(sliding_window(2, self.dimensions))
        if self.single_module == -1 or self.single_module == 0:
            layers = []
            for i, size in enumerate(layer_sizes):
                layers.append(("fc"+str(i), nn.Linear(size[0], size[1])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.encoder = nn.Sequential(OrderedDict(layers))
        else:
            self.encoder = nn.Sequential()

        if self.single_module == 0 or self.single_module == 1:
            layers = []
            layer_sizes[-1] = (layer_sizes[-1][0], layer_sizes[-1][1] + self.extra_feature_len)
            for i, size in enumerate(layer_sizes[-1::-1]):
                layers.append(("fc"+str(i), nn.Linear(size[1], size[0])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(x))

    def encode(self, x):
        _x, label = x[:, :-self.extra_feature_len], x[:, -self.extra_feature_len:]
        h = self.encoder(_x)
        return h, label

    def decode(self, h, label):
        hr = torch.cat((h, label), dim=1)
        return self.decoder(hr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log('ptl/val_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log('ptl/test_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best_val_loss=self.best_val_loss.cpu().numpy())

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):

        model = {"k":self.k, "dimensions":self.dimensions,"keep_prob":self.keep_prob, "name":self.name,
                 "extra_feature_len" : self.extra_feature_len,
                 "encoder":self.encoder.state_dict(),
                 "decoder":self.decoder.state_dict()}

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, str(best_val_loss)+"."+str(self.k)+".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath

    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)

        model = MLP_withLabel(name=obj["name"], dimensions=obj["dimensions"], extra_feature_len=obj["extra_feature_len"], keep_prob=obj["keep_prob"], load=True)
        model.encoder.load_state_dict(obj["encoder"])
        model.decoder.load_state_dict(obj["decoder"])
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup_data(self):
        pass
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True)

    @staticmethod
    def init_params(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(.01)



class MLP(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(MLP, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module
        self.act = nn.ELU
        self.k = 0
        if load:
            self.build()
        else:
            self.hidden_dim = config["hidden_dim"]
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.act = config["activation"]
            self.loss_fn = config["loss_fn"]
            self.batch_size = config["batch_size"]

            self.dimensions = dimensions + [self.hidden_dim, self.k]
            self.train_set = train_set
            self.val_set = val_set
            self.test_set = test_set

            self.best_val_loss = np.inf

            self.build()
        self.encoder.apply(self.init_params)
        self.decoder.apply(self.init_params)


    def build(self):
        layer_sizes = list(sliding_window(2, self.dimensions))
        if self.single_module == -1 or self.single_module == 0:
            layers = []
            for i, size in enumerate(layer_sizes):
                layers.append(("fc"+str(i), nn.Linear(size[0], size[1])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.encoder = nn.Sequential(OrderedDict(layers))
        else:
            self.encoder = nn.Sequential()

        if self.single_module == 0 or self.single_module == 1:
            layers = []
            for i, size in enumerate(layer_sizes[-1::-1]):
                layers.append(("fc"+str(i), nn.Linear(size[1], size[0])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log('ptl/val_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log('ptl/test_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best_val_loss=self.best_val_loss.cpu().numpy())

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):

        model = {"k":self.k, "dimensions":self.dimensions,"keep_prob":self.keep_prob, "name":self.name,
                 "encoder":self.encoder.state_dict(),
                 "decoder":self.decoder.state_dict()}

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, str(best_val_loss)+"."+str(self.k)+".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath

    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)

        model = MLP(name=obj["name"], dimensions=obj["dimensions"], keep_prob=obj["keep_prob"], load=True)
        model.encoder.load_state_dict(obj["encoder"])
        model.decoder.load_state_dict(obj["decoder"])
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup_data(self):
        pass
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True)

    @staticmethod
    def init_params(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(.01)



class MoE(nn.Module):
    def __init__(self, config=None, dimensions=None, phase_input_dim:int=0, gate_size=0, name="model", load=False):
        super().__init__()

        self.phase_input_dim = phase_input_dim
        self.dimensions = dimensions
        self.act_fn = nn.ELU
        self.name = name
        self.config=config
        self.gate_size=gate_size
        if not load:
            self.k_experts = config["k_experts"]
            self.gate_size = config["gate_size"]
            self.keep_prob = config["keep_prob"]
            self.dimensions = [self.dimensions[0], config["hidden_dim"], config["hidden_dim"], self.dimensions[-1]]

        self.layers = []

        self.build()
        self.gate = nn.Sequential(
            nn.Linear(phase_input_dim, self.gate_size),
            self.act_fn(),
            nn.Linear(self.gate_size, self.gate_size),
            self.act_fn(),
            nn.Linear(self.gate_size, self.k_experts)
        )
        self.init_params()



    def forward(self, x:torch.Tensor, phase) -> torch.Tensor:
        coefficients = F.softmax(self.gate(phase), dim=1)
        layer_out = x
        for (weight, bias, activation) in self.layers:
            if weight is None:
                layer_out = activation(layer_out, p=self.keep_prob)
            else:
                flat_weight = weight.flatten(start_dim=1, end_dim=2)
                mixed_weight = torch.matmul(coefficients, flat_weight).view(
                    coefficients.shape[0], *weight.shape[1:3]
                )
                input = layer_out.unsqueeze(1)
                mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
                out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
                layer_out = activation(out) if activation is not None else out
        return layer_out

    def build(self):
        layers = []
        for i, size in enumerate(zip(self.dimensions[0:], self.dimensions[1:])):
            if i < len(self.dimensions) - 2:
                layers.append(
                    (
                        nn.Parameter(torch.empty(self.k_experts, size[0], size[1])),
                        nn.Parameter(torch.empty(self.k_experts, size[1])),
                        self.act_fn()
                    )
                )
                if self.keep_prob > 0:
                    layers.append((None, None, F.dropout))
            else:
                layers.append(
                    (
                        nn.Parameter(torch.empty(self.k_experts, size[0], size[1])),
                        nn.Parameter(torch.empty(self.k_experts, size[1])),
                        None
                    )
                )


        self.layers = layers

    def init_params(self):
        for i, (w, b, _) in enumerate(self.layers):
            if w is None:
                continue

            i = str(i)
            torch.nn.init.kaiming_uniform_(w)
            b.data.fill_(0.01)
            self.register_parameter("w" + i, w)
            self.register_parameter("b" + i, b)

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):

        model = {"dimensions":self.dimensions,
                 "name":self.name,
                 "gate":self.gate.state_dict(), "phase_input_dim":self.phase_input_dim,
                 "generationNetwork":self.state_dict(),
                 "gate_size":self.gate_size,
                 }

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, str(best_val_loss)+".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath

    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)

        model = MoE(name=obj["name"], dimensions=obj["dimensions"], gate_size=obj["gate_size"],
                    phase_input_dim=obj["phase_input_dim"])
        model.gate.load_state_dict(obj["gate"])
        model.load_state_dict(obj["generationNetwork"])
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return [optimizer], [scheduler]


class MotionGenerationModel(pl.LightningModule):
    def __init__(self, config:dict=None, pose_autoencoder=None, cost_input_dimension=None, phase_dim=0,
                 input_slicers:list=None, output_slicers:list=None, train_set=None, val_set=None, name="model", load=False):
        super().__init__()

        if not load:
            self.pose_autoencoder = pose_autoencoder # start with 3
            cost_hidden_dim = config["cost_hidden_dim"]
            self.cost_encoder = MLP(dimensions=[cost_input_dimension, cost_hidden_dim, cost_hidden_dim],
                                    name="CostEncoder", load=True, single_module=-1)


            phase_dim = input_slicers[0]
            moe_input_dim = pose_autoencoder.dimensions[-1] + cost_hidden_dim
            moe_output_dim = pose_autoencoder.dimensions[-1] + phase_dim*2
            self.generationModel =  MoE(config=config, dimensions=[moe_input_dim, moe_output_dim], phase_input_dim=phase_dim,
                                        name="MixtureOfExperts")
            # self.batch_norm = nn.BatchNorm1d(np.sum())

            # self.in_slices = [0] + list(accumulate(add, input_slicers))
            self.in_slices = input_slicers
            # self.out_slices = [0] + list(accumulate(add, output_slicers))
            self.out_slices = output_slicers

            self.config=config
            self.batch_size = config["batch_size"]
            self.learning_rate = config["lr"]
            self.loss_fn = config["loss_fn"]
            self.window_size = config["window_size"]
            self.autoregress_prob = config["autoregress_prob"]
            self.autoregress_inc = config["autoregress_inc"]
            self.best_val_loss = np.inf
            self.phase_smooth_factor = 0.9

        self.train_set = train_set
        self.val_set = val_set
        self.name = name



    def forward(self, x):
        x=torch.squeeze(x,dim=0)
        # x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.in_slices[:-1], self.in_slices[1:])]
        x_tensors = torch.split(x, self.in_slices, dim=1)
        pose_h, pose_label = self.pose_autoencoder.encode(x_tensors[1])
        cost_out =self.cost_encoder(x_tensors[2])
        embedding = torch.cat((pose_h, cost_out), dim=1)
        out = self.generationModel(embedding, x_tensors[0])
        # out_tensors = [out[:, d0:d1] for d0, d1 in zip(self.out_slices[:-1], self.out_slices[1:])] # phase, phase_update, pose
        out_tensors = torch.split(out, self.out_slices, dim=1)

        phase = self.update_phase(x_tensors[0], out_tensors[0], out_tensors[1]) # phase_0, phase_1, phase_update
        new_pose = self.pose_autoencoder.decode(out_tensors[2], pose_label)

        out2 = torch.cat((phase, new_pose), dim=1)
        # out = self.batch_norm(out)
        return out2

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("ptl/val_loss", loss, prog_bar=True)
        return {"val_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint()

    def save_checkpoint(self, checkpoint_dir=MODEL_PATH):
        path = os.path.join(checkpoint_dir, self.name)
        loss = self.best_val_loss.cpu().numpy()

        pose_autoencoder_path = self.pose_autoencoder.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)
        cost_encoder_path = self.cost_encoder.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)
        generationModel_path = self.generationModel.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)

        model = {"name":self.name,
                 "pose_autoencoder_path":pose_autoencoder_path,
                 "cost_encoder_path": cost_encoder_path,
                 "motionGenerationModelPath":generationModel_path,
                 "in_slices":self.in_slices,
                 "out_slices":self.out_slices,
                 }

        if not os.path.exists(path):
            os.mkdir(path)
        with bz2.BZ2File(os.path.join(path,
                                      str(loss)+".pbz2"), "w") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_checkpoint(filename, pose_ae_model, cost_encoder_model, generation_model):
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        pose_autoencoder = pose_ae_model.load_checkpoint(obj["pose_autoencoder_path"])
        cost_encoder = cost_encoder_model.load_checkpoint(obj["cost_encoder_path"])
        generationModel = generation_model.load_checkpoint(obj["motionGenerationModelPath"])
        model = MotionGenerationModel(name=obj["name"])
        model.pose_autoencoder = pose_autoencoder
        model.cost_encoder = cost_encoder
        model.generationModel = generationModel
        model.in_slices = obj["in_slices"]
        model.out_slices = obj["out_slices"]

        return model

    def update_phase(self, p1, p2, p_delta):
        return self.phase_smooth_factor * p2 + (1-self.phase_smooth_factor)*(p1+p_delta)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return [optimizer],[scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True)





data_path = ["/home/nuoc/Documents/MEX/data/ONE_R2-default-One.pbz2",
             "/home/nuoc/Documents/MEX/data/ONE_R2-default-One-large.pbz2",
             "/home/nuoc/Documents/MEX/data/ONE_R2-default-One-small.pbz2"
             ]



pose_features = ["pos", "rotMat", "velocity", "isLeft", "chainPos", "geoDistanceNormalised"]
cost_features = ["posCost", "rotCost", "targetPosition", "targetRotation","contact"]
phase_features = ["phase_vec"]




def load(file_path):
    with bz2.BZ2File(file_path, "rb") as f:
        obj = pickle.load(f)
        return obj

data = [load(path) for path in data_path]




window_size = 6
frame_window = 30
sampling_step = frame_window / window_size

data_tensors = []

data_dims = []
feature_list = []

data_nan = []
first_row = True
for Data in data:
    for clip in Data:
        d = pickle.loads(clip)
        sequence = []
        n_frames = len(d["frames"])
        for f, frame in enumerate(d["frames"]):
            row_vec = []
            f_idx = np.arange(f-frame_window, f+frame_window, sampling_step, dtype=int)
            f_idx[f_idx < 0] = 0
            f_idx[f_idx >= n_frames] = n_frames-1
            f_idx = f_idx.tolist()
            for feature in phase_features:
                if feature == "phase_vec":

                    sin = np.asarray([jo["phase_vec"] for idx in f_idx for jo in d["frames"][idx] if jo["key"]])
                    vel = np.concatenate([jo["velocity"] for idx in f_idx for jo in d["frames"][idx]  if jo["key"]])
                    vel = np.reshape(vel, (3,-1))
                    vel = np.sqrt(np.sum(vel**2, axis=0))
                    vel[vel==0] = 1
                    cos = np.cos(np.arcsin(sin/vel)) * vel
                    row_vec.append(np.concatenate([np.asarray([sin[i], cos[i]]) for i in range(len(sin))]))
                else:
                    row_vec.append(np.asarray([jo[feature] for jo in frame if jo["key"]]))

                if first_row:
                    data_dims.append(row_vec[-1].shape)
                    feature_list.append(feature)
            for feature in pose_features:
                if feature=="rotMat":
                    row_vec.append(np.concatenate([jo["rotMat"].ravel() for jo in frame]))
                elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                    row_vec.append(np.concatenate([[jo[feature]] for jo in frame]))
                else:
                    row_vec.append(np.concatenate([jo[feature] for jo in frame]))

                if first_row:
                    data_dims.append(row_vec[-1].shape)
                    feature_list.append(feature)
            for feature in cost_features:
                if feature == "contact":
                    row_vec.append(np.asarray([jo["contact"] for idx in f_idx for jo in d["frames"][idx]  if jo["key"]], dtype=np.float32))
                elif feature == "targetRotation":
                    row_vec.append(np.concatenate([jo[feature].ravel() for jo in frame if jo["key"]]))
                elif feature == "posCost" or feature == "rotCost":
                    row_vec.append(np.concatenate([jo[feature] for idx in f_idx for jo in d["frames"][idx] if jo["key"]]))
                else:
                    row_vec.append(np.concatenate([jo[feature] for jo in frame if jo["key"]]))

                if first_row:
                    data_dims.append(row_vec[-1].shape)
                    feature_list.append(feature)
            if first_row: first_row = False
            sequence.append(np.concatenate(row_vec))
            data_nan.append(np.isnan(sequence[-1]))
        data_tensors.append(np.vstack(sequence))


def loss_fn(x, y):
    return nn.functional.mse_loss(x,y, reduction="mean")
def normalise(x):
    std = torch.std(x, dim=0)
    std[std==0] = 1
    return (x-torch.mean(x,dim=0)) / std

extra_feature_len = 21 * 3
n_phase_features = len(phase_features)
n_pose_features = len(pose_features)
phase_dim = np.sum(data_dims[0:n_phase_features])
pose_dim = np.sum(data_dims[n_phase_features:n_phase_features+n_pose_features])
cost_dim = np.sum(data_dims[n_phase_features+n_pose_features:])

table = [feature_list, data_dims]
print(tabulate(table))
print("phase dim: ",phase_dim)
print("pose dim: ", pose_dim)
print("cost dim: ", cost_dim)

x_tensors = torch.stack([normalise(torch.from_numpy(clip[:-1])).float() for clip in data_tensors])
y_tensors = torch.stack([torch.from_numpy(clip[1:][:, :-(cost_dim+extra_feature_len)]).float() for clip in data_tensors])


print(len(x_tensors), x_tensors[0].shape)
print(len(y_tensors), y_tensors[0].shape)



dataset = TensorDataset(x_tensors, y_tensors)
N = len(x_tensors)

train_ratio = int(.7*N)
val_ratio = int((N-train_ratio) / 2.0)
train_set, val_set, test_set = random_split(dataset, [train_ratio, val_ratio, val_ratio], generator=torch.Generator().manual_seed(2021))
print(len(train_set), len(val_set), len(test_set))


input_dim = phase_dim + pose_dim + cost_dim
output_dim = phase_dim + pose_dim-extra_feature_len
print(input_dim)
print(output_dim)

config = {
    "k_experts":tune.choice([1, 2, 4, 8, 10]),
    "gate_size":tune.choice([16, 32, 64, 128]),
    "keep_prob":tune.choice([.2, .25, .3]),
    "hidden_dim":tune.choice([16, 32, 64, 128, 256, 512]),
    "cost_hidden_dim" : tune.choice([16, 32, 64, 128]),
    "batch_size":tune.choice([1]),
    "lr":tune.loguniform(1e-3, 1e-8),
    "loss_fn":tune.choice([loss_fn]),
    "window_size":tune.choice([1]),
    "autoregress_prob" : tune.choice([.2]),
    "autoregress_inc" : tune.choice([.1])
}

def tuning(config=None, MODEL=None, pose_autoencoder=None, cost_dim=None, phase_dim=None,
          input_slices=None, output_slices=None,
          train_set=None, val_set=None, num_epochs=300, model_name="model"):
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=1,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name, version="0.0"),
        progress_bar_refresh_rate=5,
        callbacks=[
            TuneReportCallback({"loss": "avg_val_loss", }, on="validation_end"),
            EarlyStopping(monitor="avg_val_loss")
        ],
    )
    model = MODEL(config=config, pose_autoencoder=pose_autoencoder, cost_input_dimension=cost_dim, phase_dim=phase_dim,
                              input_slicers=input_slices, output_slicers=output_slices,
                              train_set=train_set, val_set=val_set, name=model_name)

    trainer.fit(model)


def start_training(name):
    Epochs = 1000
    Samples = 50
    ModelName=name

    pose_autoencoder = MLP_withLabel.load_checkpoint("/home/nuoc/Documents/MEX/models/MLP4_withLabel_best/M3/0.00324857.512.pbz2")
    # pose_autoencoder = MLP_withLabel.load_checkpoint("/home/nuoc/Documents/MEX/models/MLP_withLabel/0.0013522337.512.pbz2")

    pose_encoder_out_dim = pose_autoencoder.dimensions[-1]

    scheduler = ASHAScheduler(max_t = Epochs, grace_period=15, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["k", "lr", "batch_size", "loss_fn"],
        metric_columns=["loss", "training_iteration"],
        max_error_rows=5,
        max_progress_rows=5,
        max_report_frequency=1)

    analysis = tune.run(
        tune.with_parameters(
            tuning,
            MODEL=MotionGenerationModel,
            pose_autoencoder=pose_autoencoder,
            cost_dim = cost_dim,
            phase_dim=phase_dim,
            input_slices=[phase_dim, pose_dim, cost_dim],
            output_slices=[phase_dim, phase_dim, pose_encoder_out_dim],
            train_set=train_set, val_set=val_set,
            num_epochs=Epochs,
            model_name=ModelName
        ),
        resources_per_trial= {"cpu":2, "gpu":1},
        metric="loss",
        mode="min",
        config=config,
        num_samples=Samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=ModelName,
        verbose=False
    )

    print("-"*70)
    print("Done")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best achieved loss was: ", analysis.best_result)
    print("-"*70)

    ray.shutdown()


if __name__ == "__main__":
    model_name = "MoE_Test_2"
    start_training(model_name)
    clean_checkpoints("/home/nuoc/Documents/MEX/models/"+model_name)