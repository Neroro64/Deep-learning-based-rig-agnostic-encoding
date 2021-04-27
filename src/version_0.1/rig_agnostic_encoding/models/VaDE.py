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
import math
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
sys.path.append("..")
from settings.GlobalSettings import DATA_PATH, MODEL_PATH
from VaDE_withLabel import VaDE_Layer

"""
from eelxpeng {https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py}
"""

class VaDE(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, n_centroids:int=10,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(VaDE, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module
        self.act = nn.ReLU
        self.n_centroids = n_centroids

        if load:
            self.build()
        else:
            self.hidden_dim = config["hidden_dim"]
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.act = config["activation"]
            self.loss_fn = config["loss_fn"]
            self.batch_size = config["batch_size"]
            self.n_centroids = config["n_centroids"]

            self.dimensions = [self.dimensions[0], self.hidden_dim, self.k]
            self.train_set = train_set
            self.val_set = val_set
            self.test_set = test_set

            self.best_val_loss = np.inf

            self.build()
            self.encoder.apply(self.init_params)
            self.decoder.apply(self.init_params)
            self.initialize_gmm(train_set)



    def build(self):
        layer_sizes = list(sliding_window(2, self.dimensions))
        if self.single_module == -1 or self.single_module == 0:
            layers = []
            for i, size in enumerate(layer_sizes):
                if i == len(layer_sizes)-1:
                    self.cluster_layer = VaDE_Layer(layer_size=size, n_centroids=self.n_centroids)
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

    def forward(self, x:torch.Tensor):
        z,mu, logvar = self.encode(x)
        return self.decode(z, mu, logvar)

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.cluster_layer(h)
        return z, mu, logvar

    def decode(self, h, mu, logvar):
        return self.decoder(h), h, mu, logvar

    def initialize_gmm(self, train_data):
        self.eval()
        data = []
        for batch_idx, inputs in enumerate(train_data):
            inputs = torch.unsqueeze(inputs, 0)
            inputs = Variable(inputs)

            outputs, z,  mu, logvar = self.forward(inputs)
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        gmm.fit(data)
        self.cluster_layer.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.cluster_layer.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction, z, mu, logvar = self(x)
        # loss = self.loss_fn(prediction, y)
        loss, kl_loss,recon_loss = self.cluster_layer.loss_function(prediction, x, z, mu, logvar)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_kl_loss", kl_loss)
        self.log("ptl/train_recon_loss", recon_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction, z, mu, logvar = self(x)
        # loss = self.loss_fn(prediction, y)
        loss, kl_loss, recon_loss = self.cluster_layer.loss_function(prediction, x, z, mu, logvar)

        self.log('ptl/val_loss', loss, prog_bar=True)
        self.log('ptl/val_kl_loss', kl_loss, prog_bar=True)
        self.log('ptl/val_recon_loss', recon_loss, prog_bar=True)
        return {"val_loss":loss, "kl_loss":kl_loss, "recon_loss":recon_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction, z, mu, logvar = self(x)
        # loss = self.loss_fn(prediction, y)
        loss, kl_loss, recon_loss = self.cluster_layer.loss_function(prediction, x, z, mu, logvar)

        self.log('ptl/test_loss', loss, prog_bar=True)
        self.log('ptl/test_kl_loss', kl_loss, prog_bar=True)
        self.log('ptl/test_recon_loss', recon_loss, prog_bar=True)
        return {"val_loss":loss, "kl_loss":kl_loss, "recon_loss":recon_loss}


    def validation_epoch_end(self, outputs):
        avg_kl_loss = torch.stack([x["kl_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("avg_val_kl_loss", avg_kl_loss)
        self.log("avg_val_recon_loss", avg_recon_loss)
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best_val_loss=self.best_val_loss.cpu().numpy())

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):

        model = {"k":self.k, "dimensions":self.dimensions,"keep_prob":self.keep_prob, "name":self.name,
                 "n_centroids" : self.n_centroids,
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

        model = VaDE(name=obj["name"], dimensions=obj["dimensions"], n_centroids=obj["n_centroids"],
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
