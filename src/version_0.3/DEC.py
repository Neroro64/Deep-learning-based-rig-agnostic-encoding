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



class DEC_Layer(nn.Module):
    def __init__(
            self, cluster_number:int,
            embedding_dimension: int,
            alpha:float = 1.0,
            cluster_centers=None,
            device="cuda",
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(DEC_Layer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.device = device
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float, device=self.device
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def loss(self, x):
        weight = (x ** 2) / torch.sum(x, 0)
        p = (weight.t() / torch.sum(weight, 1)).t().detach()
        loss = torch.mean(nn.functional.kl_div(x.log(), p, log_target=False),dim=0)
        return loss


class DEC(pl.LightningModule):
    def __init__(self, config:dict=None, input_dims:list=None, pose_labels=None,
                 pos_dim=None, rot_dim=None, vel_dim=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model", save_period=5, workers=6):

        super(DEC, self).__init__()

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

        self.cluster_model = DEC_Layer(embedding_dimension=self.k, cluster_number=self.z_dim, device=config["device"])

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
        dec_loss = [self.cluster_model.loss(y_tensors[i]) for i in range(len(y_tensors))]

        loss = sum(losses) / float(len(losses))
        dec_loss = sum(dec_loss) / float(len(dec_loss))

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_dec_loss", dec_loss)
        return loss+dec_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        y_tensors = [y[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]
        losses = [self.active_models[i].loss(prediction[i], y_tensors[i])[0] for i in range(len(prediction))]
        dec_loss = [self.cluster_model.loss(y_tensors[i]) for i in range(len(y_tensors))]

        loss = sum(losses) / float(len(losses))
        dec_loss = sum(dec_loss) / float(len(dec_loss))

        self.log('ptl/val_loss', loss, prog_bar=True)
        self.log("ptl/val_dec_loss", dec_loss, prog_bar=True)
        return {"val_loss":loss+dec_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        prediction = self(x)
        y_tensors = [y[:, d0:d1] for d0, d1 in zip(self.input_slice[:-1], self.input_slice[1:])]
        losses = [self.active_models[i].loss(prediction[i], y_tensors[i])[0] for i in range(len(prediction))]
        dec_loss = [self.cluster_model.loss(y_tensors[i]) for i in range(len(y_tensors))]

        loss = sum(losses) / float(len(losses))
        dec_loss = sum(dec_loss) / float(len(dec_loss))

        self.log('ptl/test_loss', loss, prog_bar=True)
        self.log("ptl/test_dec_loss", dec_loss, prog_bar=True)
        return {"test_loss":loss+dec_loss}

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
        model = DEC(config=obj["config"], name=obj["name"],
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