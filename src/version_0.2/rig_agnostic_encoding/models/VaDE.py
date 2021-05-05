import pytorch_lightning as pl
import numpy as np
from cytoolz import sliding_window, accumulate
from operator import add
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
sys.path.append("../../")

from GlobalSettings import DATA_PATH, MODEL_PATH
from MLP import MLP

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

    def loss_function(self, z, z_mean, z_log_var):

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

        kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum().clamp(max=0)
        kl_loss /= z_log_var.numel()

        loss = torch.mean(kl_loss + logpzc + qentropy + logpc + logqcx)
        return loss, kl_loss

    def forward(self, h):
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar



class VaDE(pl.LightningModule):
    def __init__(self, config: dict = None, input_dims: list = None, pose_labels=None,
                 train_set=None, val_set=None, test_set=None,
                 name: str = "model", save_period=5, workers=6):

        super(VaDE, self).__init__()

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
        self.n_centroid = config["n_centroid"]

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.Adam
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None

        self.models = [MLP(config=config, dimensions=[input_dims[i]], pose_labels=self.pose_labels[i],
                           name="M" + str(i), single_module=0) for i in range(M)]
        self.active_models = self.models

        self.cluster_model = VaDE_Layer(layer_size=[self.k, self.k], n_centroids=self.n_centroid)

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
        decoded = [m.decode(z[i]) for i, m in enumerate(self.active_models)]

        return torch.cat(decoded, dim=1), z, mu, logvar

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(z[i], mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum([l[1] for l in loss]) / float(len(loss))
        loss = sum([l[0] for l in loss]) / float(len(loss))

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_loss_kl", kl_loss)
        self.log("ptl/train_loss_recon", recon_loss)
        return loss + recon_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(z[i], mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum([l[1] for l in loss]) / float(len(loss))
        loss = sum([l[0] for l in loss]) / float(len(loss))

        self.log("ptl/val_loss", loss, prog_bar=True)
        self.log("ptl/val_loss_kl", kl_loss, prog_bar=True)
        self.log("ptl/val_loss_recon", recon_loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction, z, mu, logvar = self(x)
        recon_loss = self.loss_fn(prediction, y)
        loss = [self.cluster_model.loss_function(z[i], mu[i], logvar[i]) for i in range(len(z))]
        kl_loss = sum([l[1] for l in loss]) / float(len(loss))
        loss = sum([l[0] for l in loss]) / float(len(loss))

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
            "n_centroid" : self.n_centroid,
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
        model = VaDE(config=obj["config"], name=obj["name"],
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