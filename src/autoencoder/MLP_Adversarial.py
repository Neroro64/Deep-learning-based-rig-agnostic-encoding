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
sys.path.append("../func")
from GlobalSettings import DATA_PATH, MODEL_PATH


class MLP_ADV(pl.LightningModule):
    """
    A simple autoencoder with two Multi-layer perceptron model and a built-in adversarial module

    Parameters:
        config (dict):                      a config for the model parameters
            example:
                {
                    "hidden_dim":           256,
                    "k":                    256,
                    "z_dim":                256,
                    "lr":                   1e-4,
                    "batch_size":           32,
                    "keep_prob":            0,
                    "loss_fn":              torch.nn.functional.mse_loss,
                    "optimizer":            torch.optim.AdamW,
                    "scheduler":            torch.optim.lr_scheduler.StepLR,
                    "scheduler_param":      {"step_size":80, "gamma":.9},
                    "basis_func":           "gaussian",
                    "n_centroid":           64,
                    "k_experts":            4,
                    "gate_size":            128,
                    "g_hidden_dim":         512,
                    "num_layers":           4,
                    "autoregress_prob":     0,
                    "autoregress_inc":      .3,
                    "autoregress_ep":       20,
                    "autoregress_max_prob": 1,
                    "cost_hidden_dim":      128,
                    "seq_len":              13,
                    "device":               "cuda"
                }
        dimensions (list):                  a list of layer sizes example: [input, hidden, hidden, output]
        pose_labels (torch.nn.Tensor):      a tensor of joint labels (left/right)
        h_dim (int):                        hight dimension for the convolutional layer in the adversarial model
        w_dim (int):                        width dimension for the convolutional layer in the adversarial model
        train_set (torch.nn.Tensor):        training set
        val_set (torch.nn.Tensor):          validation set
        test_set (torch.nn.Tensor):         test set
        pose_dim (int):                     dimension for the pose component
        rot_dim (int):                      dimension for the rotation component
        vel_dim (int):                      dimension for the velocity component
        name (str):                         model name
        single_module (int):                0 = {encoder, decoder}; -1 = {encoder}; 1 = {decoder}
        save_period (int):                  how often (in epochs) the checkpoint is saved during the training
        workers (int):                      the number of workers for the dataset loader

    Returns:
        (pytorch_lightning.LightninigModule) MLP_ADV model
    """
    def __init__(self, config:dict=None, dimensions:list=None, pose_labels:torch.nn.Tensor=None,
                 h_dim:int=0, w_dim:int=0,
                 train_set=None, val_set=None, test_set=None, pos_dim:int=0, rot_dim:int=0, vel_dim:int=0,
                 name:str="model", single_module:int=0, save_period:int=5,
                 workers:int=6):

        super(MLP_ADV, self).__init__()

        # Setting up the MLP network
        self.name = name
        self.dimensions = dimensions
        self.single_module = single_module
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.act = nn.ELU
        self.save_period = save_period
        self.workers = workers
        self.config=config

        self.pos_dim = pos_dim
        self.rot_dim = pos_dim + rot_dim
        self.vel_dim = self.rot_dim + vel_dim

        if config is not None:
            self.hidden_dim = config["hidden_dim"]
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.batch_size = config["batch_size"]

            self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
            self.opt = config["optimizer"] if "optimizer" in config else torch.optim.AdamW
            self.scheduler = config["scheduler"] if "scheduler" in config else None
            self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None

            if "device" not in config:
                config["device"] = "cuda"

            self.encoder.to(config["device"])
            self.decoder.to(config["device"])

        if dimensions is not None:
            self.dimensions = dimensions if len(dimensions) > 1 else \
                [dimensions[0], self.hidden_dim, self.hidden_dim, self.k]


        self.pose_labels = pose_labels  # should be Tensor(1,63) for example
        self.use_label = pose_labels is not None

        self.train_set, self.val_set, self.test_set = train_set, val_set, test_set
        self.best_val_loss = np.inf

        self.encoder, self.decoder = nn.Module(), nn.Module()
        self.build()

        self.encoder.apply(self.init_params)
        self.decoder.apply(self.init_params)

        # Setting up the adversarial network
        h_out = int((((((h_dim-2) - 3) / 3 + 1) - 2) - 3 ) / 3 + 1)
        w_out = int((((((w_dim-2) - 3) / 3 + 1) - 2) - 3 ) / 3 + 1)

        if h_dim > 0:
            self.convDiscriminator = nn.Sequential(
                nn.Conv2d(
                    in_channels = 1,out_channels = 1,
                    kernel_size = 3,stride = 1,padding = 0,
                ),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
                nn.BatchNorm2d(1), nn.ELU(),
                nn.Conv2d(
                    in_channels=1, out_channels=1,
                    kernel_size=3, stride=1, padding=0,
                ),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
                nn.BatchNorm2d(1), nn.ELU(),
                nn.Flatten(),
                nn.Linear(in_features=int(h_out*w_out), out_features=1)
            )

    def build(self) -> None:
        """
        Construct the layers
        """
        layer_sizes = list(sliding_window(2, self.dimensions))
        if self.single_module == -1 or self.single_module == 0:
            layers = []
            for i, size in enumerate(layer_sizes):
                layers.append(("fc"+str(i), nn.Linear(size[0], size[1])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), self.act()))
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
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x:torch.Tensor):
        return self.encoder(x)

    def decode(self, h:torch.Tensor):
        if self.use_label:  # Concatenate the pose labels to the embeddings
            h = torch.cat((h, self.pose_labels.expand(h.shape[0],-1)), dim=1)
        return self.decoder(h)

    def decode_label(self, h:torch.Tensor):
        h = torch.cat((h, self.pose_labels.expand(h.shape[0],-1)), dim=1)
        return self.decoder(h)

    def loss(self, x:torch.Tensor, y:torch.Tensor):
        px, py = x[:, :self.pos_dim].detach(), y[:, :self.pos_dim].detach()
        rx, ry = x[:, self.pos_dim:self.rot_dim].detach() % 2 * np.pi, y[:, self.pos_dim:self.rot_dim].detach() % 2 * np.pi

        pos_loss = nn.functional.mse_loss(px, py)

        rot_loss = nn.functional.mse_loss(rx, ry)
        recon_loss = self.loss_fn(x, y)
        return recon_loss, pos_loss, rot_loss

    def training_step(self, batch, _, optimizer_idx):
        x, y = batch
        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        if optimizer_idx == 0:
            d_real = self.convDiscriminator(y.unsqueeze(1))
            d_fake = self.convDiscriminator(prediction.unsqueeze(1))
            d_loss = 0.5 * (torch.mean(d_real - 1)**2 + torch.mean(d_fake**2))
            loss = d_loss

            self.log("ptl/train_d_loss", d_loss)
        else:

            d_fake = self.convDiscriminator(prediction.unsqueeze(1))
            g_loss = 0.5 * torch.mean((d_fake-1)**2) + recon_loss
            loss = g_loss

            self.log("ptl/train_g_loss", g_loss)

        self.log("ptl/train_loss", recon_loss)
        self.log("ptl/train_pos_loss", pos_loss)
        self.log("ptl/train_rot_loss", rot_loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch

        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        d_real = self.convDiscriminator(y.unsqueeze(1))
        d_fake = self.convDiscriminator(prediction.unsqueeze(1))
        d_loss = 0.5 * (torch.mean(d_real - 1)**2) + torch.mean(d_fake**2)

        d_fake = self.convDiscriminator(prediction.unsqueeze(1))
        g_loss = 0.5 * torch.mean((d_fake-1)**2)

        self.log("ptl/val_loss", recon_loss, prog_bar=True)
        self.log("ptl/val_d_loss", d_loss, prog_bar=True)
        self.log("ptl/val_g_loss", g_loss, prog_bar=True)
        self.log("ptl/val_pos_loss", pos_loss, prog_bar=True)
        self.log("ptl/val_rot_loss", rot_loss, prog_bar=True)
        return {"val_loss":recon_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        prediction = self(x)
        recon_loss, pos_loss, rot_loss = self.loss(prediction, y)

        d_real = self.convDiscriminator(y.unsqueeze(1))
        d_fake = self.convDiscriminator(prediction.unsqueeze(1))
        d_loss = 0.5 * (torch.mean(d_real - 1)**2) + torch.mean(d_fake**2)

        d_fake = self.convDiscriminator(prediction.unsqueeze(1))
        g_loss = 0.5 * torch.mean((d_fake-1)**2)

        self.log("ptl/test_loss", recon_loss, prog_bar=True)
        self.log("ptl/test_d_loss", d_loss, prog_bar=True)
        self.log("ptl/test_g_loss", g_loss, prog_bar=True)
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
            "optimizer":self.opt,
            "scheduler":self.scheduler,
            "scheduler_param":self.scheduler_param,
            "device":self.config["device"],
        }
        model = {"config":config, "name":self.name,"dimensions":self.dimensions,
                 "pose_labels": self.pose_labels,
                 "single_module":self.single_module,
                 "dims": [self.pos_dim, self.rot_dim, self.vel_dim, self.h_dim, self.w_dim],
                 "encoder":self.encoder.state_dict(),
                 "decoder":self.decoder.state_dict(),
                 "discriminator":self.convDiscriminator.state_dict()}

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
        model = MLP_ADV(config=obj["config"], single_module=obj["single_module"], pose_labels=obj["pose_labels"],
                        h_dim=obj["dims"][-2], w_dim=obj["dims"][-1],
                    name=obj["name"], dimensions=obj["dimensions"])

        model.encoder.load_state_dict(obj["encoder"])
        model.decoder.load_state_dict(obj["decoder"])
        model.convDiscriminator.load_state_dict(obj["discriminator"])
        model.pos_dim = obj["dims"][0]
        model.rot_dim = obj["dims"][1]
        model.vel_dim = obj["dims"][2]

        return model

    def freeze(self, flag=False):
        self.encoder.requires_grad_(flag)
        self.decoder.requires_grad_(flag)
        self.convDiscriminator.requires_grad_(flag)

    def configure_optimizers(self):
        optimizer_D = self.opt(self.convDiscriminator.parameters(), lr=self.learning_rate)
        optimizer_G = self.opt(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler_D = self.scheduler(optimizer_D, **self.scheduler_param)
            scheduler_G = self.scheduler(optimizer_G, **self.scheduler_param)
            return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]
        else:
            return [optimizer_D, optimizer_G]

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
