import pytorch_lightning as pl
import numpy as np
from cytoolz import sliding_window
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
import bz2
import _pickle as pickle
sys.path.append("../../")
from GlobalSettings import DATA_PATH, MODEL_PATH

class MLP(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, pose_labels=None,
                 train_set=None, val_set=None, test_set=None, pos_dim=0, rot_dim=0, vel_dim=0,
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

        self.pos_dim = pos_dim
        self.rot_dim = pos_dim + rot_dim
        self.vel_dim = self.rot_dim + vel_dim

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
                # if i == 0:
                    # layers.append(("drop", nn.Dropout(self.keep_prob)))
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

    def loss(self, x, y):
        px, py =x[:, :self.pos_dim].detach(), y[:, :self.pos_dim].detach()
        px_norm, py_norm = torch.sum(px**2), torch.sum(py**2)
        pos_loss = (px-py)**2 / (px_norm*py_norm)

        rx, ry = x[:, self.pos_dim:self.rot_dim].detach(), y[:, self.pos_dim:self.rot_dim].detach()
        rx %= 2 * np.pi
        ry %= 2 * np.pi
        rot_loss = self.loss_fn(rx, ry)
        recon_loss = self.loss_fn(x, y)

        return recon_loss, pos_loss, rot_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        self.log("ptl/train_loss", recon_loss)
        self.log("ptl/train_pos_loss", pos_loss)
        self.log("ptl/train_rot_loss", rot_loss)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        self.log("ptl/val_loss", recon_loss, prog_bar=True)
        self.log("ptl/val_pos_loss", pos_loss, prog_bar=True)
        self.log("ptl/val_rot_loss", rot_loss, prog_bar=True)
        return {"val_loss":recon_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        self.log("ptl/test_loss", recon_loss)
        self.log("ptl/test_pos_loss", pos_loss)
        self.log("ptl/test_rot_loss", rot_loss)
        return {"test_loss":recon_loss}

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
            "dims": [self.pos_dim, self.rot_dim, self.vel_dim]
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
        model.pos_dim = obj["config"]["dims"][0]
        model.rot_dim = obj["config"]["dims"][1]
        model.vel_dim = obj["config"]["dims"][2]
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
