#%%

import os, sys
sys.path.append("motion_generation")
sys.path.append("rig_agnostic_encoding/functions")
sys.path.append("rig_agnostic_encoding/models")


from rig_agnostic_encoding.functions.DataProcessingFunctions import clean_checkpoints
from GlobalSettings import MODEL_PATH

import torch
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
import ray
from ray import tune
import importlib
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from cytoolz import sliding_window
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
import bz2

from cytoolz import sliding_window, accumulate, get
from operator import add

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

class VAE_Layer(nn.Module):
    def __init__(self, layer_size:list=None):
        super(self.__class__, self).__init__()
        self._enc_mu = nn.Linear(*layer_size)
        self._enc_log_sigma = nn.Linear(*layer_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum().clamp(max=0)
        kl_loss /= z_log_var.numel()

        return kl_loss

    def forward(self, h):
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAE(pl.LightningModule):
    def __init__(self, config: dict = None, input_dims: list = None, pose_labels=None,
                 train_set=None, val_set=None, test_set=None,
                 name: str = "model", save_period=5, workers=6):

        super(VAE, self).__init__()

        M = len(input_dims)

        self.name = name
        self.input_dims = input_dims
        self.input_slice = [0] + list(accumulate(add, input_dims))

        self.act = nn.ELU
        self.save_period = save_period
        self.workers = workers
        self.pose_labels = pose_labels if pose_labels is not None else [None for _ in range(M)]
        self.use_label = pose_labels is not None and pose_labels[0] is not None

        self.config = config
        self.hidden_dim = config["hidden_dim"]
        self.keep_prob = config["keep_prob"]
        self.k = config["k"]
        self.learning_rate = config["lr"]
        self.batch_size = config["batch_size"]

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.Adam
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None

        self.models = [MLP(config=config, dimensions=[input_dims[i]], pose_labels=self.pose_labels[i],
                           name="M" + str(i), single_module=0) for i in range(M)]
        self.active_models = self.models

        self.cluster_model = VAE_Layer(layer_size=[self.k, self.k])

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.best_val_loss = np.inf


    def forward(self, x: torch.Tensor):
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]

        encoded = [m.encode(x_tensors[i]) for i, m in enumerate(self.active_models)]
        embeddings = [self.cluster_model(vec) for vec in encoded]
        z = [vec[0] for vec in embeddings]
        mu = [vec[1] for vec in embeddings]
        logvar = [vec[2] for vec in embeddings]
        if self.use_label:
            decoded = [m.decode_label(z[i]) for i, m in enumerate(self.active_models)]
        else:
            decoded = [m.decode(z[i]) for i, m in enumerate(self.active_models)]

        return torch.cat(decoded, dim=1), z, mu, logvar

    def training_step(self, batch, batch_idx):
        x, y = batch

        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum(loss) / float(len(loss))
        loss = recon_loss+kl_loss

        self.log("ptl/train_loss", loss, prog_bar=True)
        self.log("ptl/train_loss_kl", kl_loss)
        self.log("ptl/train_loss_recon", recon_loss)
        return loss + recon_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum(loss) / float(len(loss))
        loss = recon_loss + kl_loss

        self.log("ptl/val_loss", loss, prog_bar=True)
        self.log("ptl/val_loss_kl", kl_loss, prog_bar=True)
        self.log("ptl/val_loss_recon", recon_loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum(loss) / float(len(loss))
        loss = recon_loss + kl_loss

        self.log("ptl/test_loss", loss, prog_bar=True)
        self.log("ptl/test_loss_kl", kl_loss, prog_bar=True)
        self.log("ptl/test_loss_recon", recon_loss, prog_bar=True)
        return {"test_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if self.current_epoch % self.save_period == 0:
                self.save_checkpoint(best_val_loss=self.best_val_loss.item())

    def save_checkpoint(self, best_val_loss: float = np.inf, checkpoint_dir=MODEL_PATH):
        path = os.path.join(checkpoint_dir, self.name)

        config = {
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "keep_prob": self.keep_prob,
            "optimizer": self.opt,
            "scheduler": self.scheduler,
            "scheduler_param": self.scheduler_param,
        }

        model_paths = [m.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path) for m in self.models]
        model = {"config": config, "name": self.name, "model_paths": model_paths,
                 "input_dims": self.input_dims, "pose_labels": self.pose_labels,
                 "cluster_model": self.cluster_model.state_dict()
                 }

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, str(best_val_loss) + "." + str(self.k) + ".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath

    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)
        model = VAE(config=obj["config"], name=obj["name"],
                     input_dims=obj["input_dims"], pose_labels=obj["pose_labels"])

        models = [MLP.load_checkpoint(path) for path in obj["model_paths"]]
        model.models = models
        model.cluster_model.load_state_dict(obj["cluster_model"])
        return model

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

    def add_models(self, input_dims: list = None, pose_labels: list = None, freeze=False):
        n = len(self.models) + 1
        if pose_labels is not None:
            self.models += [MLP(config=self.config, dimensions=[input_dims[i]], pose_labels=pose_labels[i],
                                name="M" + str(i + n), single_module=0) for i in range(len(input_dims))]
        else:
            self.models += [MLP(config=self.config, dimensions=[input_dims[i]],
                                name="M" + str(i + n), single_module=0) for i in range(len(input_dims))]
        if freeze:
            for model in self.active_models:
                model.freeze(True)
            self.active_models = self.models[n - 1:]
            self.input_dims = input_dims
        else:
            self.active_models = self.models
            self.input_dims += input_dims

        self.input_slice = [0] + list(accumulate(add, self.input_dims))

config_old = {
    "hidden_dim": 256,
    "k": 256,
    "lr": 1e-3,
    "batch_size": 16,
    "keep_prob": .3,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.Adam,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":80, "gamma":.7},
    "basis_func":"gaussian",
    "n_centroid":128,
    "k_experts": 4,
    "gate_size": 64,
    "g_hidden_dim": 256,
    "num_layers": 4,
    "autoregress_prob":0,
    "autoregress_inc":.2,
    "autoregress_ep":10,
    "autoregress_max_prob":.5,
    "cost_hidden_dim":128,
    "seq_len":13,
    "device":"cuda"
    }

config = {
    "hidden_dim": tune.choice([128, 256, 512]),
    "k": tune.choice([128, 256, 512]),
    "lr": tune.loguniform(1e-3, 1e-7),
    "batch_size": tune.choice([16, 32, 64]),
    "keep_prob": tune.choice([.2, .3]),
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":tune.choice([torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD]),
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": tune.choice([{"step_size":80, "gamma":.7},{"step_size":80, "gamma":.9}, {"step_size":80, "gamma":.8}]),
    "basis_func":"gaussian",
    "n_centroid":128,
    "k_experts": 4,
    "gate_size": 64,
    "g_hidden_dim": 256,
    "num_layers": 4,
    "autoregress_prob":0,
    "autoregress_inc":.2,
    "autoregress_ep":10,
    "autoregress_max_prob":.5,
    "cost_hidden_dim":128,
    "seq_len":13,
    "device":"cuda"
    }
dataset_name = "R1_Pose_data"
obj = F.load("/home/nuoc/Documents/MEX/data/"+dataset_name+".pbz2")
train_set, val_set, test_set = obj["data"]
feature_dims = obj["feature_dims"]
x_tensors, y_tensors2 = obj["original"]

#%%
pose_dim = x_tensors.shape[-1]
print(len(train_set), train_set[0][0].shape)
print(len(val_set), val_set[0][0].shape)
print(len(test_set), test_set[0][0].shape)

#%%
model_name = "VAE_TUNE"

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
    model = VAE(config=config, input_dims=[pose_dim], name=model_name,
                    train_set=train_set, val_set=val_set, test_set=test_set)

    trainer.fit(model)


MAX_EPOCHS = 300
n_samples = 50

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

