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
sys.path.append("..")
from settings.GlobalSettings import DATA_PATH, MODEL_PATH
from RBF_withLabel import RBF_Layer


class RBF(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, basis_func=None,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(RBF, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module
        self.basis_func = basis_func
        self.act = nn.ReLU

        if load:
            self.build()
        else:
            self.basis_func = basis_func_dict()[config["basis_func"]]
            self.hidden_dim = config["hidden_dim"]
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.act = config["activation"]
            self.loss_fn = config["loss_fn"]
            self.batch_size = config["batch_size"]

            self.dimensions = [self.dimensions[0], self.hidden_dim, self.k]
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
                if i == len(layer_sizes)-1:
                    self.cluster_layer = RBF_Layer(in_features=size[0], out_features=size[1], basis_func=self.basis_func)
                else:
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
        h, label, _ = self.encode(x)
        return self.decode(h, label)

    def encode(self, x):
        h = self.encoder(x)
        h2 = self.cluster_layer(h)
        return h2

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
                 "basis_func":self.basis_func,
                 "encoder":self.encoder.state_dict(),
                 "cluster_layer":self.cluster_layer.state_dict(),
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

        model = RBF(name=obj["name"], dimensions=obj["dimensions"], basis_func=obj["basis_func"],
                              keep_prob=obj["keep_prob"], load=True)
        model.encoder.load_state_dict(obj["encoder"])
        model.cluster_layer.load_state_dict(obj["cluster_layer"])
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


def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases