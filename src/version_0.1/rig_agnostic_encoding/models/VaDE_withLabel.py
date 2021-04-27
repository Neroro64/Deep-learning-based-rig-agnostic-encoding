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


"""
from eelxpeng {https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py}
"""


class VaDE_Layer(nn.Module):
    def __init__(self, layer_size:list=None, n_centroids:int=10):
        super(self.__class__, self).__init__()
        self.n_centroids = n_centroids
        self._enc_mu = nn.Linear(*layer_size)
        self._enc_log_sigma = nn.Linear(*layer_size)

        self.create_gmmparam(n_centroids, layer_size[-1])

    def create_gmmparam(self, n_centroids, z_dim):
        self.theta_p = nn.Parameter(torch.ones(n_centroids)/n_centroids)
        self.u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
        self.lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def loss_function(self, recon_x, x, z, z_mean, z_log_var):

        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK

        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
            torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
        qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
        logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        recon_loss = (recon_x - x).pow(2).mean(dim=(0,-1))
        kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum().clamp(max=0)
        kl_loss /= z_log_var.numel()

        # kl_loss = kl_loss.clamp(min=0)
        # logpzc = logpzc.clamp(min=0)
        # qentropy = qentropy.clamp(min=0)
        # logpc = logpc.clamp(min=0)
        # logqcx = logqcx.clamp(min=0)

        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(recon_loss + kl_loss + logpzc + qentropy + logpc + logqcx)
        return loss, kl_loss, recon_loss

    def forward(self, h):
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar



class VaDE_withLabel(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None, n_centroids:int=10,
                 extra_feature_len:int=0,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(VaDE_withLabel, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module
        self.extra_feature_len = extra_feature_len
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

            self.dimensions = [self.dimensions[0]-extra_feature_len, self.hidden_dim, self.k]
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
            layer_sizes[-1] = (layer_sizes[-1][0], layer_sizes[-1][1] + self.extra_feature_len)
            for i, size in enumerate(layer_sizes[-1::-1]):
                layers.append(("fc"+str(i), nn.Linear(size[1], size[0])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()

    def forward(self, x:torch.Tensor):
        z, label, mu, logvar = self.encode(x)
        return self.decode(z, label, mu, logvar)

    def encode(self, x):
        _x, label = x[:, :-self.extra_feature_len], x[:, -self.extra_feature_len:]
        h = self.encoder(_x)
        z, mu, logvar = self.cluster_layer(h)
        return z, label, mu, logvar

    def decode(self, h, label, mu, logvar):
        hr = torch.cat((h, label), dim=1)
        return self.decoder(hr), h, mu, logvar

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
                 "extra_feature_len" : self.extra_feature_len,
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

        model = VaDE_withLabel(name=obj["name"], dimensions=obj["dimensions"], n_centroids=obj["n_centroids"],
                              extra_feature_len=obj["extra_feature_len"], keep_prob=obj["keep_prob"], load=True)
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
