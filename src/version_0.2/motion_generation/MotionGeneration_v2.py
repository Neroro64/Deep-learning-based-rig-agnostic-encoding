import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import os, sys
import bz2
import _pickle as pickle
from cytoolz import accumulate, sliding_window
from operator import add

sys.path.append("../")
sys.path.append("../rig_agnostic_encoding/models/")
sys.path.append("../rig_agnostic_encoding/functions/")

from GlobalSettings import MODEL_PATH
from MLP import MLP


class MotionGenerationModel(pl.LightningModule):
    def __init__(self, config:dict=None, Model=None, pose_autoencoder=None, middle_layer=None, feature_dims=None,
                 input_slicers:list=None, output_slicers:list=None,
                 train_set=None, val_set=None, test_set=None, name="MotionGeneration", workers=8):
        super().__init__()

        self.feature_dims = feature_dims
        self.config=config
        self.workers = workers

        self.loss_fn = config["loss_fn"] if "loss_fn" in config else nn.functional.mse_loss
        self.opt = config["optimizer"] if "optimizer" in config else torch.optim.Adam
        self.scheduler = config["scheduler"] if "scheduler" in config else None
        self.scheduler_param = config["scheduler_param"] if "scheduler_param" in config else None
        self.batch_size = config["batch_size"]
        self.learning_rate = config["lr"]
        self.seq_len = config["seq_len"]

        self.autoregress_prob = config["autoregress_prob"]
        self.autoregress_inc = config["autoregress_inc"]
        self.autoregress_ep = config["autoregress_ep"]
        self.autoregress_max_prob = config["autoregress_max_prob"]

        self.best_val_loss = np.inf
        self.phase_smooth_factor = 0.9

        self.pose_autoencoder = pose_autoencoder if pose_autoencoder is not None else \
            MLP(config=config, dimensions=[feature_dims["pose_dim"]], name="PoseAE")
        self.middle_layer = middle_layer if middle_layer is not None else \
            nn.Linear(in_features=self.pose_autoencoder.dimensions[-1], out_features=self.pose_autoencoder.dimensions[-1])
        self.use_label = pose_autoencoder is not None and pose_autoencoder.use_label

        cost_hidden_dim = config["cost_hidden_dim"]
        self.cost_encoder = MLP(config=config,
                            dimensions=[feature_dims["cost_dim"], cost_hidden_dim, cost_hidden_dim, cost_hidden_dim],
                            name="CostEncoder", single_module=-1)

        self.generationModel =  Model(config=config,
                                      dimensions=[feature_dims["g_input_dim"], feature_dims["g_output_dim"]],
                                      phase_input_dim = feature_dims["phase_dim"])

        self.input_dims = input_slicers
        self.output_dims = output_slicers
        self.in_slices = [0] + list(accumulate(add, input_slicers))
        self.out_slices = [0] + list(accumulate(add, output_slicers))

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.name = name

        self.automatic_optimization = False

    def forward(self, x):
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.in_slices[:-1], self.in_slices[1:])]

        pose_h = self.middle_layer(self.pose_autoencoder.encode(x_tensors[1]))
        embedding = torch.cat([pose_h, self.cost_encoder(x_tensors[2])], dim=1)
        out = self.generationModel(embedding, x_tensors[0])
        out_tensors = [out[:, d0:d1] for d0, d1 in zip(self.out_slices[:-1], self.out_slices[1:])]
        phase = self.update_phase(x_tensors[0], out_tensors[0])
        new_pose = self.pose_autoencoder.decode(out_tensors[1])

        return [phase, new_pose, out_tensors[-1]]

    def computeCost(self, targets, trajectory):
        # targetPos = targets[:, :self.feature_dims["targetPosition"]]
        # targetRot = targets[:, self.feature_dims["targetPosition"]:]
        # posT = trajectory[:, :self.feature_dims["tPos"]]
        # rotT = trajectory[:, self.feature_dims["tPos"]:]
        #
        # targetPos = targetPos.reshape((-1, 12, 3))
        # posT = posT.reshape((-1, 12, 3))
        # # targetRot = targetRot.reshape((-1, 12, 3,3))
        # # rotT = rotT.reshape((-1, 12, 3, 3))
        #
        # posCost = torch.sum(((targetPos - posT)**2), axis=2).reshape((-1, self.feature_dims["posCost"]))
        # colLength = torch.sqrt(torch.clip(torch.sum(rotT**2, axis=2), 0))
        # rotT = rotT / colLength[:, :, :, None]

        # rotT = torch.transpose(rotT, dim0=2, dim1=3)
        # trace =torch.diagonal(targetRot @ rotT, offset=0, dim1=2, dim2=3).sum(dim=2)
        # rotCost = torch.abs(torch.arccos((torch.clamp( (trace - 1) / 2.0, -1, 1))))
        # torch.nan_to_num_(rotCost, 0)
        # rotCost = rotCost.reshape((-1, self.feature_dims["rotCost"]))

        return 0, 0

    def step(self, x, y):
        opt = self.optimizers()

        n = x.size()[1]
        tot_loss = 0

        x_c = x[:,0,:]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:,i-1,:]

            out= self(x_c)
            recon = torch.cat(out, dim=1)
            loss = self.loss_fn(recon, y_c)
            tot_loss += loss.detach()

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            if autoregress_bools[i]:
                x_c = torch.cat(out,dim=1).detach()
            else:
                x_c = x[:,i,:]

        tot_loss /= float(n)
        return tot_loss

    def step_eval(self, x, y):
        n = x.size()[1]
        tot_loss = 0

        x_c = x[:,0,:]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:,i-1,:]

            out= self(x_c)
            recon = torch.cat(out, dim=1)
            loss = self.loss_fn(recon, y_c)
            tot_loss += loss.detach()

            if autoregress_bools[i]:
                x_c = torch.cat(out,dim=1).detach()
            else:
                x_c = x[:,i,:]

        tot_loss /= float(n)
        return tot_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step(x,y)
        self.log("ptl/train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step_eval(x,y)
        self.log("ptl/val_loss", loss, prog_bar=True)

        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss = self.step_eval(x,y)
        self.log("ptl/test_loss", loss, prog_bar=True)

        return {"test_loss":loss}

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0 and self.current_epoch % self.autoregress_ep == 0:
            self.autoregress_prob = min(self.autoregress_max_prob, self.autoregress_prob+self.autoregress_inc)

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best_val_loss=self.best_val_loss.item())

    def save_checkpoint(self, best_val_loss=np.inf, checkpoint_dir=MODEL_PATH):
        path = os.path.join(checkpoint_dir, self.name)

        config = {}
        for key, val in self.config.items():
            config[key] = val

        pose_autoencoder_path = self.pose_autoencoder.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)
        cost_encoder_path = self.cost_encoder.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)
        generationModel_path = self.generationModel.save_checkpoint(best_val_loss=best_val_loss, checkpoint_dir=path)

        model = {"name":self.name,
                 "config":config,
                 "pose_autoencoder_path":pose_autoencoder_path,
                 "cost_encoder_path": cost_encoder_path,
                 "motionGenerationModelPath":generationModel_path,
                 "in_slices":self.in_slices,
                 "out_slices":self.out_slices,
                 "feature_dims":self.feature_dims,
                 "middle_layer_dict":self.middle_layer.state_dict(),
                 }

        if not os.path.exists(path):
            os.mkdir(path)
        with bz2.BZ2File(os.path.join(path,
                                      str(best_val_loss)+".pbz2"), "w") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_checkpoint(filename, Model, MiddleModel:nn.Module=None):
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        pose_autoencoder = MLP.load_checkpoint(obj["pose_autoencoder_path"])
        cost_encoder = MLP.load_checkpoint(obj["cost_encoder_path"])
        generationModel = Model.load_checkpoint(obj["motionGenerationModelPath"])

        model = MotionGenerationModel(config=obj["config"], feature_dims=obj["feature_dims"], Model=Model,
                                      input_slicers=obj["in_slices"], output_slicers=obj["out_slices"],
                                      name=obj["name"])
        if MiddleModel is None:
            MiddleModel = nn.Linear(in_features=pose_autoencoder.dimensions[-1], out_features=pose_autoencoder.dimensions[-11])

        MiddleModel.load_state_dict(obj["middle_layer_dict"])
        model.in_slices = obj["in_slices"]
        model.out_slices = obj["out_slices"]
        model.pose_autoencoder = pose_autoencoder
        model.cost_encoder = cost_encoder
        model.generationModel = generationModel

        return model

    def update_phase(self, p1, p2):
        return self.phase_smooth_factor * p2 + (1-self.phase_smooth_factor)*p1

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

    def swap_pose_encoder(self, pose_encoder=None,
                          input_dim=None, output_dim=None,
                          feature_dims=None, freeze=False):
        self.pose_autoencoder = pose_encoder
        self.input_dims = input_dim
        self.output_dims = output_dim
        self.feature_dims = feature_dims
        self.in_slices = [0] + list(accumulate(add, self.input_dims))
        self.out_slices = [0] + list(accumulate(add, self.output_dims))

        if freeze:
            self.generationModel.freeze()
            self.cost_encoder.freeze()
