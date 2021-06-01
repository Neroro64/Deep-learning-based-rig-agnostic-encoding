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

class RBF_Layer(nn.Module):
    """
       from JeremyLinux on GitHub {https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py}

       Transforms incoming data using a given radial basis function:
       u_{i} = rbf(||x - c_{i}|| / s_{i})
       Arguments:
           in_features: size of each input sample
           out_features: size of each output sample
       Shape:
           - Input: (N, in_features) where N is an arbitrary batch size
           - Output: (N, out_features) where N is an arbitrary batch size
       Attributes:
           centres: the learnable centres of shape (out_features, in_features).
               The values are initialised from a standard normal distribution.
               Normalising inputs to have mean 0 and standard deviation 1 is
               recommended.

           sigmas: the learnable scaling factors of shape (out_features).
               The values are initialised as ones.

           basis_func: the radial basis function used to transform the scaled
               distances.
       """

    def __init__(self, in_features: int=0, out_features: int=0, basis_func=None, device="cuda"):
        super(RBF_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.device = device
        self.identity = torch.diag(torch.ones(self.out_features, device=self.device, requires_grad=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, .01)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)  # ALT. / (2*sigma**2)
        return self.basis_func(distances)

    def freeze(self, flag=False):
        self.centres.requires_grad = flag
        self.sigmas.requires_grad = flag

    def loss(self, *args):
        return 0.1 * torch.mean(torch.abs(self.centres @ self.centres.T - self.identity))
        # return torch.nn.functional.mse_loss(self.centres @ self.centres.T, torch.diag(torch.ones(self.out_features, device=self.device)))
        # return torch.mean((1 - (self.centres @ self.centres.T).sum(dim=1)))



class RBF(pl.LightningModule):
    def __init__(self, config:dict=None, input_dims:list=None, pose_labels=None,
                 pos_dim=None, rot_dim=None, vel_dim=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model", save_period=5, workers=6):

        super(RBF, self).__init__()

        M = len(input_dims)

        self.name = name
        self.input_dims = input_dims
        self.input_slice = [0] + list(accumulate(add, input_dims))

        self.act = nn.ELU
        self.save_period = save_period
        self.workers = workers
        self.pose_labels = pose_labels if pose_labels is not None else [None for _ in range(M)]

        self.config = config
        self.basis_func = basis_func_dict()[config["basis_func"]]
        self.hidden_dim = config["hidden_dim"]
        self.k = config["k"]
        self.z_dim = config["z_dim"]

        self.learning_rate = config["lr"]
        self.batch_size = config["batch_size"]

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.AdamW
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None

        self.pos_dim = pos_dim if pos_dim is not None else [0 for _ in range(M)]
        self.rot_dim = rot_dim if rot_dim is not None else [0 for _ in range(M)]
        self.vel_dim = vel_dim if vel_dim is not None else [0 for _ in range(M)]

        self.best_val_loss = np.inf

        self.models = [MLP(config=self.config, dimensions=[self.input_dims[i]], pose_labels=self.pose_labels[i],
                           pos_dim=self.pos_dim[i], rot_dim=self.rot_dim[i], vel_dim=self.vel_dim[i],
                           name="M" + str(i), single_module=0) for i in range(M)]

        self.active_models = self.models

        self.cluster_model = RBF_Layer(in_features=self.k, out_features=self.z_dim, basis_func=self.basis_func, device=self.config["device"])

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.best_val_loss = np.inf

    def forward(self, x:torch.Tensor):
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]

        encoded = [m.encode(x_tensors[i]) for i, m in enumerate(self.active_models)]
        embeddings = [self.cluster_model(vec) for vec in encoded]
        decoded = [m.decode(embeddings[i]) for i, m in enumerate(self.active_models)]

        return decoded

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        y_tensors = [y[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]
        losses = [self.active_models[i].loss(prediction[i], y_tensors[i])[0] for i in range(len(prediction))]
        ortho_loss = self.cluster_model.loss()

        loss = sum(losses) / float(len(losses))
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_ortho_loss", ortho_loss)
        return loss+ortho_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        y_tensors = [y[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]
        losses = [self.active_models[i].loss(prediction[i], y_tensors[i])[0] for i in range(len(prediction))]

        loss = sum(losses) / float(len(losses))
        ortho_loss = self.cluster_model.loss()

        self.log('ptl/val_loss', loss, prog_bar=True)
        self.log("ptl/val_ortho_loss", ortho_loss, prog_bar=True)
        return {"val_loss":loss+ortho_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        y_tensors = [y[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]
        losses = [self.active_models[i].loss(prediction[i], y_tensors[i])[0] for i in range(len(prediction))]

        loss = sum(losses) / float(len(losses))
        ortho_loss = self.cluster_model.loss()

        self.log('ptl/test_loss', loss, prog_bar=True)
        self.log("ptl/test_ortho_loss", ortho_loss, prog_bar=True)
        return {"test_loss":loss+ortho_loss}

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
            "z_dim":self.z_dim,
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.opt,
            "scheduler": self.scheduler,
            "scheduler_param": self.scheduler_param,
            "basis_func":self.config["basis_func"],
        }

        model_paths = [m.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path) for m in self.models]
        model = {"config":config, "name":self.name, "model_paths":model_paths,
                 "pos_dim": self.pos_dim, "rot_dim": self.rot_dim, "vel_dim": self.vel_dim,
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
        model = RBF(config=obj["config"], name=obj["name"],
                    pos_dim=obj["pos_dim"], rot_dim=obj["rot_dim"], vel_dim=obj["vel_dim"],
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

    def freeze(self, flag=False) -> None:
        for m in self.active_models:
            m.freeze(flag)
        self.cluster_model.freeze(flag)

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