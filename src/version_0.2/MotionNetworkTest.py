#%%

import os, sys
sys.path.append("motion_generation")
sys.path.append("rig_agnostic_encoding/functions")
sys.path.append("rig_agnostic_encoding/models")

from rig_agnostic_encoding.functions.DataProcessingFunctions import clean_checkpoints
from GlobalSettings import MODEL_PATH

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
import json as js
import bz2
import _pickle as pickle
import ray
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from cytoolz import sliding_window, accumulate, get
from operator import add
from collections import OrderedDict

class MoE(nn.Module):
    def __init__(self, config=None, dimensions=None,
                 phase_input_dim:int=0, name="MoE", device="cuda"):
        super(MoE, self).__init__()

        self.phase_input_dim = phase_input_dim
        self.dimensions = dimensions
        self.act_fn = nn.ELU

        self.name = name
        self.config = config
        self.device = device

        self.k_experts = config["k_experts"]
        self.gate_size = config["gate_size"]
        self.keep_prob = config["keep_prob"]
        self.hidden_dim = config["g_hidden_dim"]

        self.dimensions = dimensions if len(dimensions) > 2 else \
            [dimensions[0], self.hidden_dim, self.hidden_dim, dimensions[-1]]

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
        coefficients = nn.functional.softmax(self.gate(phase), dim=1)
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
                    layers.append((None, None, nn.functional.dropout))
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
        config = dict(
                k_experts=self.k_experts,
                gate_size=self.gate_size,
                keep_prob=self.keep_prob, g_hidden_dim=self.hidden_dim
                      )

        model = {
            "config":config,
            "dimensions":self.dimensions,
            "name":self.name,
            "phase_input_dim":self.phase_input_dim,
            "generationNetwork":self.state_dict(),
            "gate":self.gate.state_dict(),
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

        model = MoE(config=obj["config"], name=obj["name"], dimensions=obj["dimensions"],
                    phase_input_dim=obj["phase_input_dim"])

        model.gate.load_state_dict(obj["gate"])
        model.load_state_dict(obj["generationNetwork"])
        return model

    def freeze(self):
        self.gate.requires_grad_(False)
        for (weight, bias, _) in self.layers:
            if weight == None: continue
            weight.requires_grad_(False)
            bias.requires_grad_(False)

class MLP(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, pose_labels=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model", single_module:int=0, save_period=5,
                 workers=6):

        super(MLP, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.single_module = single_module
        self.act = nn.ELU
        self.save_period = save_period
        self.workers = workers
        self.config=config

        self.hidden_dim = config["hidden_dim"]
        self.keep_prob = config["keep_prob"]
        self.k = config["k"]
        self.learning_rate = config["lr"]
        self.batch_size = config["batch_size"]

        self.dimensions = dimensions if len(dimensions) > 1 else \
            [dimensions[0], self.hidden_dim, self.hidden_dim, self.k]

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.Adam
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None

        self.pose_labels = pose_labels  # should be Tensor(1,63) for example
        self.use_label = pose_labels is not None

        self.train_set, self.val_set, self.test_set = train_set, val_set, test_set
        self.best_val_loss = np.inf

        self.encoder, self.decoder = nn.Module(), nn.Module()
        self.build()
        if "device" not in config:
            config["device"] = "cuda"

        self.encoder.to(config["device"])
        self.decoder.to(config["device"])

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
                if i == 0:
                    layers.append(("drop", nn.Dropout(self.keep_prob)))
            self.encoder = nn.Sequential(OrderedDict(layers))
        else:
            self.encoder = nn.Sequential()

        if self.single_module == 0 or self.single_module == 1:
            layers = []
            if self.pose_labels is not None:
                layer_sizes[-1] = (layer_sizes[-1][0], layer_sizes[-1][1] + self.pose_labels.numel())
            for i, size in enumerate(layer_sizes[-1::-1]):
                layers.append(("fc"+str(i), nn.Linear(size[1], size[0])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                if i == 0:
                    layers.append(("drop", nn.Dropout(self.keep_prob)))
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        if self.use_label:
            h = torch.cat((h, self.pose_labels.expand(h.shape[0],-1)), dim=1)
        return self.decoder(h)

    def decode_label(self, h):
        h = torch.cat((h, self.pose_labels.expand(h.shape[0],-1)), dim=1)
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
        return {"test_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if self.current_epoch % self.save_period == 0:
                self.save_checkpoint(best_val_loss=self.best_val_loss.item())

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):
        config = {
            "hidden_dim":self.hidden_dim,
            "k":self.k,
            "lr":self.learning_rate,
            "batch_size":self.batch_size,
            "keep_prob":self.keep_prob,
            "optimizer":self.opt,
            "scheduler":self.scheduler,
            "scheduler_param":self.scheduler_param,
            "device":self.config["device"],
        }
        model = {"config":config, "name":self.name,"dimensions":self.dimensions,
                 "pose_labels": self.pose_labels,
                 "single_module":self.single_module,
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
        model = MLP(config=obj["config"], single_module=obj["single_module"], pose_labels=obj["pose_labels"],
                    name=obj["name"], dimensions=obj["dimensions"])

        model.encoder.load_state_dict(obj["encoder"])
        model.decoder.load_state_dict(obj["decoder"])
        return model

    def freeze(self, flag=False):
        self.encoder.requires_grad_(flag)
        self.decoder.requires_grad_(flag)

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_param)
            return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    @staticmethod
    def init_params(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(.01)

class MotionGenerationModel(pl.LightningModule):
    def __init__(self, config:dict=None, Model=None, pose_autoencoder=None, feature_dims=None,
                 input_slicers:list=None, output_slicers:list=None,
                 train_set=None, val_set=None, test_set=None, name="MotionGeneration", workers=8):
        super().__init__()

        self.feature_dims = feature_dims
        self.config=config
        self.workers = workers

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.Adam
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None
        self.batch_size = config["batch_size"]
        self.learning_rate = config["lr"]
        self.seq_len = config["seq_len"]

        self.autoregress_prob = config["autoregress_prob"]
        self.autoregress_inc = config["autoregress_inc"]
        self.autoregress_ep = config["autoregress_ep"]
        self.autoregress_max_prob = config["autoregress_max_prob"]

        self.best_val_loss = np.inf
        self.phase_smooth_factor = 0.9

        self.pose_autoencoder = pose_autoencoder if pose_autoencoder is not None else \
            MLP(config=config, dimensions=[feature_dims["pose_dim"]], name="PoseAE")
        self.use_label = pose_autoencoder is not None and pose_autoencoder.use_label

        cost_hidden_dim = config["cost_hidden_dim"]
        self.cost_encoder = MLP(config=config,
                            dimensions=[feature_dims["cost_dim"], cost_hidden_dim, cost_hidden_dim, cost_hidden_dim],
                            name="CostEncoder", single_module=-1)

        self.generationModel =  Model(config=config,
                                      dimensions=[feature_dims["g_input_dim"], feature_dims["g_output_dim"]],
                                      phase_input_dim = feature_dims["phase_dim"])

        self.input_dims = input_slicers
        self.output_dims = output_slicers
        self.in_slices = [0] + list(accumulate(add, input_slicers))
        self.out_slices = [0] + list(accumulate(add, output_slicers))

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.name = name

        self.automatic_optimization = False

    def forward(self, x):
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.in_slices[:-1], self.in_slices[1:])]

        pose_h = self.pose_autoencoder.encode(x_tensors[1])
        embedding = torch.cat([pose_h, self.cost_encoder(x_tensors[2])], dim=1)
        out = self.generationModel(embedding, x_tensors[0])
        out_tensors = [out[:, d0:d1] for d0, d1 in zip(self.out_slices[:-1], self.out_slices[1:])]
        phase = self.update_phase(x_tensors[0], out_tensors[0])
        new_pose = self.pose_autoencoder.decode(out_tensors[1])

        return [phase, new_pose, out_tensors[-1]]

    def computeCost(self, targets, trajectory):
        # targetPos = targets[:, :self.feature_dims["targetPosition"]]
        # targetRot = targets[:, self.feature_dims["targetPosition"]:]
        # posT = trajectory[:, :self.feature_dims["tPos"]]
        # rotT = trajectory[:, self.feature_dims["tPos"]:]
        #
        # targetPos = targetPos.reshape((-1, 12, 3))
        # posT = posT.reshape((-1, 12, 3))
        # # targetRot = targetRot.reshape((-1, 12, 3,3))
        # # rotT = rotT.reshape((-1, 12, 3, 3))
        #
        # posCost = torch.sum(((targetPos - posT)**2), axis=2).reshape((-1, self.feature_dims["posCost"]))
        # colLength = torch.sqrt(torch.clip(torch.sum(rotT**2, axis=2), 0))
        # rotT = rotT / colLength[:, :, :, None]

        # rotT = torch.transpose(rotT, dim0=2, dim1=3)
        # trace =torch.diagonal(targetRot @ rotT, offset=0, dim1=2, dim2=3).sum(dim=2)
        # rotCost = torch.abs(torch.arccos((torch.clamp( (trace - 1) / 2.0, -1, 1))))
        # torch.nan_to_num_(rotCost, 0)
        # rotCost = rotCost.reshape((-1, self.feature_dims["rotCost"]))

        return 0, 0

    def step(self, x, y):
        opt = self.optimizers()

        n = x.size()[1]
        tot_loss = 0

        x_c = x[:,0,:]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:,i-1,:]

            out= self(x_c)
            recon = torch.cat(out, dim=1)
            loss = self.loss_fn(recon, y_c)
            tot_loss += loss.detach()

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            if autoregress_bools[i]:
                x_c = torch.cat(out,dim=1).detach()
            else:
                x_c = x[:,i,:]

        tot_loss /= float(n)
        return tot_loss

    def step_eval(self, x, y):
        n = x.size()[1]
        tot_loss = 0

        x_c = x[:,0,:]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:,i-1,:]

            out= self(x_c)
            recon = torch.cat(out, dim=1)
            loss = self.loss_fn(recon, y_c)
            tot_loss += loss.detach()

            if autoregress_bools[i]:
                x_c = torch.cat(out,dim=1).detach()
            else:
                x_c = x[:,i,:]

        tot_loss /= float(n)
        return tot_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step(x,y)
        self.log("ptl/train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step_eval(x,y)
        self.log("ptl/val_loss", loss, prog_bar=True)

        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step_eval(x,y)
        self.log("ptl/test_loss", loss, prog_bar=True)

        return {"test_loss":loss}

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0 and self.current_epoch % self.autoregress_ep == 0:
            self.autoregress_prob = min(self.autoregress_max_prob, self.autoregress_prob+self.autoregress_inc)

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best_val_loss=self.best_val_loss.item())

    def save_checkpoint(self, best_val_loss=np.inf, checkpoint_dir=MODEL_PATH):
        path = os.path.join(checkpoint_dir, self.name)

        config = {}
        for key, val in self.config.items():
            config[key] = val

        pose_autoencoder_path = self.pose_autoencoder.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)
        cost_encoder_path = self.cost_encoder.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)
        generationModel_path = self.generationModel.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)

        model = {"name":self.name,
                 "config":config,
                 "pose_autoencoder_path":pose_autoencoder_path,
                 "cost_encoder_path": cost_encoder_path,
                 "motionGenerationModelPath":generationModel_path,
                 "in_slices":self.in_slices,
                 "out_slices":self.out_slices,
                 "feature_dims":self.feature_dims,
                 }

        if not os.path.exists(path):
            os.mkdir(path)
        with bz2.BZ2File(os.path.join(path,
                                      str(best_val_loss)+".pbz2"), "w") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_checkpoint(filename, Model):
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        pose_autoencoder = MLP.load_checkpoint(obj["pose_autoencoder_path"])
        cost_encoder = MLP.load_checkpoint(obj["cost_encoder_path"])
        generationModel = Model.load_checkpoint(obj["motionGenerationModelPath"])

        model = MotionGenerationModel(config=obj["config"], feature_dims=obj["feature_dims"],
                                      input_slicers=obj["in_slices"], output_slicers=obj["out_slices"],
                                      name=obj["name"])

        model.pose_autoencoder = pose_autoencoder
        model.cost_encoder = cost_encoder
        model.generationModel = generationModel

        return model

    def update_phase(self, p1, p2):
        return self.phase_smooth_factor * p2 + (1-self.phase_smooth_factor)*p1

    def configure_optimizers(self):
        optimizer = self.opt(self.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_param)
            return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True, num_workers=self.workers)

    def swap_pose_encoder(self, pose_encoder=None,
                          input_dim=None, output_dim=None,
                          feature_dims=None, freeze=False):
        self.pose_autoencoder = pose_encoder
        self.input_dims = input_dim
        self.output_dims = output_dim
        self.feature_dims = feature_dims
        self.in_slices = [0] + list(accumulate(add, self.input_dims))
        self.out_slices = [0] + list(accumulate(add, self.output_dims))

        if freeze:
            self.generationModel.freeze()
            self.cost_encoder.freeze()



config = {
    "hidden_dim": 512,
    "k": tune.choice([256, 512]),
    "lr": tune.loguniform(1e-4, 1e-7),
    "batch_size": 32,
    "keep_prob": .2,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.AdamW,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": tune.choice([{"step_size":40, "gamma":.9},{"step_size":80, "gamma":.9}, {"step_size":160, "gamma":.4}]),
    "basis_func":"gaussian",
    "n_centroid":128,
    "k_experts": tune.choice([4, 8, 10]),
    "gate_size": tune.choice([128, 256]),
    "g_hidden_dim": tune.choice([128, 256, 512]),
    "num_layers": 4,
    "autoregress_prob":0,
    "autoregress_inc":tune.choice([.2, .3]),
    "autoregress_ep":tune.choice([5, 10, 20, 50]),
    "autoregress_max_prob":.5,
    "cost_hidden_dim":tune.choice([128, 256, 512]),
    "seq_len":13,
    "device":"cuda"
    }

MAX_FILES = -1
data_path = "/home/nuoc/Documents/MEX/data/Dataset_R1_One_1"
file_paths = []
for dname, dirs, files in os.walk(data_path):
    for i, file in enumerate(files):
        file_paths.append(os.path.join(dname, file))
        if MAX_FILES > 0 and i >= MAX_FILES:
            break


phase_features = ["phase_vec_l2"]
pose_features = ["pos", "rotMat2", "velocity"]
cost_features = ["posCost", "rotCost"]
features = phase_features + pose_features + cost_features
clips = []
feature_dims = {}


data = F.process_data_multithread(file_paths, features, use_window=True)
feature_dims = data[0][1]
clips = [np.copy(d[0]) for d in data]

phase_dim = sum([feature_dims[feature] for feature in phase_features])
pose_dim = sum([feature_dims[feature] for feature in pose_features])
cost_dim = sum([feature_dims[feature] for feature in cost_features])
print(phase_dim, " ", pose_dim, " ", cost_dim)


x_tensors = torch.stack([F.normaliseT(torch.from_numpy(clip[:-1])).float() for clip in clips])
y_tensors = torch.stack([F.normaliseT(torch.from_numpy(clip[1:])).float() for clip in clips])
y_tensors2 = torch.stack([torch.from_numpy(clip[1:]).float() for clip in clips])

dataset = TensorDataset(torch.Tensor(x_tensors), torch.Tensor(y_tensors))
N = len(x_tensors)

train_ratio = int(.7*N)
val_ratio = int((N-train_ratio) / 2.0)
test_ratio = N - train_ratio - val_ratio
train_set, val_set, test_set = random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=torch.Generator().manual_seed(2021))
print(len(train_set), len(val_set), len(test_set))


F.save([train_set, val_set, test_set], "R1_MoGenData_window", "/home/nuoc/Documents/MEX/data")


# obj = F.load("/home/nuoc/Documents/MEX/data/Test1.pbz2")
# train_set, val_set, test_set = obj

#%%

print(len(train_set), train_set[0][0].shape)
print(len(val_set), val_set[0][0].shape)
print(len(test_set), test_set[0][0].shape)

#%%

model_name = "MLP_MoE_R1_One_1_Full_Window_TUNE"
MAX_EPOCHS = 300

def train_model(config=None):
    trainer = pl.Trainer(
        default_root_dir="/home/nuoc/Documents/MEX/src/version_0.2/checkpoints",
        gpus=1, precision=16,
        min_epochs=20,
        max_epochs=MAX_EPOCHS,
        callbacks=[TuneReportCallback({"loss": "avg_val_loss", }, on="validation_end")],
        logger=TensorBoardLogger(save_dir="logs/", name=model_name, version="0.0"),
        stochastic_weight_avg=True
    )
    feature_dims2 = {
        "phase_dim": phase_dim,
        "pose_dim": pose_dim,
        "cost_dim": cost_dim,
        "g_input_dim": config["k"] + config["cost_hidden_dim"],
        "g_output_dim": phase_dim + config["k"] + cost_dim
    }

    in_slice = [phase_dim, pose_dim, cost_dim]
    out_slice = [phase_dim, config["k"], cost_dim]

    pose_encoder = MLP(config=config, dimensions=[pose_dim])
    model = MotionGenerationModel(config=config, Model=MoE, pose_autoencoder=pose_encoder,
                                     feature_dims=feature_dims2,
                                     input_slicers=in_slice, output_slicers=out_slice,
                                     train_set=train_set, val_set=val_set, test_set=test_set,
                                     name=model_name
                                     )

    trainer.fit(model)


n_samples = 30

scheduler = ASHAScheduler(max_t=MAX_EPOCHS, grace_period=1, reduction_factor=2)
reporter = CLIReporter(
    parameter_columns=["k", "lr", "batch_size", "hidden_dim"],
    metric_columns=["loss", "training_iteration"],
    max_error_rows=5,
    max_progress_rows=5,
    max_report_frequency=10)

analysis = tune.run(
    tune.with_parameters(
        train_model,
    ),
    resources_per_trial={"cpu": 6, "gpu": 1},
    metric="loss",
    mode="min",
    config=config,
    num_samples=n_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    name=model_name,
    verbose=False
)

print("-" * 70)
print("Done")
print("Best hyperparameters found were: ", analysis.best_config)
print("Best achieved loss was: ", analysis.best_result)
print("-" * 70)

clean_checkpoints(path=os.path.join(MODEL_PATH,model_name))

