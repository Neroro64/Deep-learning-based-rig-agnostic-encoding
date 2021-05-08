import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
import bz2
import _pickle as pickle
from cytoolz import sliding_window, accumulate, get
from operator import add
sys.path.append("../../")
from GlobalSettings import DATA_PATH, MODEL_PATH
from MLP import MLP


class MLP_MIX(pl.LightningModule):
    def __init__(self, config:dict=None, input_dims:list=None, pose_labels=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model", save_period=5, workers=6):

        super().__init__()

        M = len(input_dims)

        self.name = name
        self.input_dims = input_dims
        self.input_slice = [0] + list(accumulate(add, input_dims))

        self.act = nn.ELU
        self.save_period = save_period
        self.workers = workers
        self.pose_labels = pose_labels if pose_labels is not None else [None for _ in range(M)]

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

        self.models = []
        self.active_models = []
        self.cluster_model = nn.Sequential()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.best_val_loss = np.inf

        self.models = [MLP(config=self.config, dimensions=[self.input_dims[i]], pose_labels=self.pose_labels[i],
                           name="M"+str(i), single_module=0) for i in range(M)]
        self.active_models = self.models

        self.cluster_model = nn.Sequential(
            nn.Linear(in_features=self.k, out_features=self.k))
        self.init_params(self.cluster_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]

        encoded = [m.encode(x_tensors[i]) for i, m in enumerate(self.active_models)]
        embeddings = [self.cluster_model(vec) for vec in encoded]
        decoded = [m.decode(embeddings[i]) for i, m in enumerate(self.active_models)]

        return torch.cat(decoded, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        loss = self.loss_fn(prediction, y)

        self.log('ptl/val_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
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
        path = os.path.join(checkpoint_dir, self.name)

        config = {
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "keep_prob":self.keep_prob,
            "optimizer": self.opt,
            "scheduler": self.scheduler,
            "scheduler_param": self.scheduler_param,
        }

        model_paths = [m.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path) for m in self.models]
        model = {"config":config, "name":self.name, "model_paths":model_paths,
                 "input_dims":self.input_dims, "pose_labels":self.pose_labels,
                 "cluster_model":self.cluster_model.state_dict()
                 }

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
        model = MLP_MIX(config=obj["config"], name=obj["name"],
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

    def add_models(self, input_dims:list=None, pose_labels:list=None, freeze=False):
        n = len(self.models)+1
        if pose_labels is not None:
            self.models += [MLP(config=self.config, dimensions=[input_dims[i]], pose_labels=pose_labels[i],
                            name="M" + str(i+n), single_module=0) for i in range(len(input_dims))]
        else:
            self.models += [MLP(config=self.config, dimensions=[input_dims[i]],
                            name="M" + str(i+n), single_module=0) for i in range(len(input_dims))]

        if freeze:
            for model in self.active_models:
                model.freeze(True)
            self.active_models = self.models[n-1:]
            self.input_dims = input_dims
        else:
            self.active_models = self.models
            self.input_dims += input_dims

        self.input_slice = [0] + list(accumulate(add, self.input_dims))
