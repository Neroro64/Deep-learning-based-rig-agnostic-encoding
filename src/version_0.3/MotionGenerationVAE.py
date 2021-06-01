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
#
# sys.path.append("../")
# sys.path.append("../rig_agnostic_encoding/models/")
# sys.path.append("../rig_agnostic_encoding/functions/")
#
from GlobalSettings import MODEL_PATH
from MLP import MLP


class MotionGenerationModel(pl.LightningModule):
    def __init__(self, config:dict=None, Model=None, pose_autoencoder=None, middle_layer=None, feature_dims=None,
                 input_slicers:list=None, output_slicers:list=None, use_advLoss=False,
                 train_set=None, val_set=None, test_set=None, name="MotionGeneration", workers=8):
        super().__init__()

        self.feature_dims = feature_dims
        self.config=config
        self.workers = workers
        self.use_advLoss = use_advLoss

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

        cost_hidden_dim = config["cost_hidden_dim"]
        self.input_dims = input_slicers
        self.output_dims = output_slicers
        self.in_slices = [0] + list(accumulate(add, input_slicers))
        self.out_slices = [0] + list(accumulate(add, output_slicers))

        self.phase_dim = feature_dims["phase_dim"]
        self.pose_dim = feature_dims["pose_dim"]
        self.cost_dim = feature_dims["cost_dim"]
        self.target_dim = feature_dims["target_dim"]

        self.posCost_dim = self.in_slices[-2] + feature_dims["posCost"]
        self.rotCost_dim = self.posCost_dim + feature_dims["rotCost"]
        self.pos_dim = feature_dims["pos_dim"]
        self.rot_dim = self.phase_dim + feature_dims["pos_dim"] + feature_dims["rot_dim"]
        self.vel_dim = self.rot_dim + feature_dims["vel_dim"]

        self.pose_autoencoder = pose_autoencoder if pose_autoencoder is not None else \
            MLP(config=config, dimensions=[feature_dims["pose_dim"]],
                pos_dim=self.pos_dim, rot_dim=feature_dims["rot_dim"], vel_dim=feature_dims["vel_dim"],
                name="PoseAE")
        self.use_label = pose_autoencoder is not None and pose_autoencoder.use_label

        self.middle_layer = middle_layer if middle_layer is not None else nn.Sequential()
        self.cost_encoder = MLP(config=config,
                                dimensions=[feature_dims["cost_dim"]+feature_dims["target_dim"], cost_hidden_dim, cost_hidden_dim, cost_hidden_dim],
                                name="CostEncoder", single_module=-1)

        self.generationModel = Model(config=config,
                                     dimensions=[feature_dims["g_input_dim"], feature_dims["g_output_dim"]],
                                     phase_input_dim=feature_dims["phase_dim"])

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.name = name

        self.automatic_optimization = False

    def forward(self, x):
        x_tensors = [x[:, d0:d1] for d0, d1 in zip(self.in_slices[:-1], self.in_slices[1:])]

        pose_h, mu, logvar = self.middle_layer(self.pose_autoencoder.encode(x_tensors[1]))
        embedding = torch.cat([pose_h, self.cost_encoder(torch.cat(x_tensors[2:],dim=1))], dim=1)
        out = self.generationModel(embedding, x_tensors[0])
        out_tensors = [out[:, d0:d1] for d0, d1 in zip(self.out_slices[:-1], self.out_slices[1:])]
        phase = self.update_phase(x_tensors[0], out_tensors[0])
        new_pose = self.pose_autoencoder.decode(out_tensors[1])

        return [phase, new_pose, out_tensors[-1], x_tensors[-1]], pose_h, mu, logvar

    def step(self, x, y):
        opt = self.optimizers()

        n = x.size()[1]
        tot_loss = 0
        tot_kl_loss = 0
        tot_recon_loss = 0

        tot_pos_loss = 0
        tot_rot_loss = 0

        min_pos_cost = 100
        min_target_pos_cost = 100
        min_rot_cost = 100
        min_target_rot_cost = 100

        sum_pos_cost = 0
        sum_target_pos_cost = 0
        sum_rot_cost = 0
        sum_target_rot_cost = 0

        x_c = x[:,0,:]
        predicted = [x_c[:, self.phase_dim:self.phase_dim + self.pose_dim].detach().unsqueeze(1)]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:,i-1,:]

            self.generationModel.reset_hidden(batch_size=y_c.shape[0])
            out, z, mu, logvar = self(x_c)
            recon = torch.cat(out, dim=1)
            predicted.append(out[1].unsqueeze(1))
            loss = self.loss_fn(recon, y_c)
            kl_loss = self.middle_layer.loss_function(z, mu, logvar)

            opt.zero_grad()
            self.manual_backward(loss + kl_loss)
            opt.step()

            recon_loss = loss.detach()
            kl_loss = kl_loss.detach()

            pos_cost_x = torch.mean(torch.abs(recon[:, self.in_slices[-2]:self.posCost_dim]))
            rot_cost_x = torch.mean(torch.abs(recon[:, self.posCost_dim:]))

            pos_cost_y = torch.mean(torch.abs(y_c[:, self.in_slices[-2]:self.posCost_dim]))
            rot_cost_y = torch.mean(torch.abs(y_c[:, self.posCost_dim:]))

            _, pos_loss, rot_loss = self.pose_autoencoder.loss(out[1], y_c[:, self.phase_dim:self.phase_dim + self.pose_dim])

            tot_pos_loss += pos_loss.detach()
            tot_rot_loss += rot_loss.detach()

            min_pos_cost = min(pos_cost_x, min_pos_cost)
            min_target_pos_cost = min(pos_cost_y, min_target_pos_cost)
            min_rot_cost = min(rot_cost_x, min_rot_cost)
            min_target_rot_cost = min(rot_cost_y, min_target_rot_cost)

            sum_pos_cost += pos_cost_x
            sum_target_pos_cost += pos_cost_y
            sum_rot_cost += rot_cost_x
            sum_target_rot_cost += rot_cost_y

            tot_recon_loss += recon_loss
            tot_loss += recon_loss + kl_loss
            tot_kl_loss += kl_loss

            if autoregress_bools[i]:
                x_c = torch.cat(out,dim=1).detach()
            else:
                x_c = x[:,i,:]

            del loss, kl_loss
        predicted = torch.cat(predicted, dim=1)

        tot_loss /= float(n)
        tot_recon_loss /= float(n)
        tot_kl_loss /= float(n)
        tot_pos_loss /= float(n)
        tot_rot_loss /= float(n)
        return tot_loss, tot_pos_loss, tot_rot_loss,tot_recon_loss, tot_kl_loss, predicted,\
               min_pos_cost, min_rot_cost, min_target_pos_cost, min_target_rot_cost,\
               sum_pos_cost, sum_rot_cost, sum_target_pos_cost, sum_target_rot_cost

    def step_eval(self, x, y):
        n = x.size()[1]
        tot_loss = 0
        tot_kl_loss = 0
        tot_recon_loss = 0

        tot_pos_loss = 0
        tot_rot_loss = 0

        min_pos_cost = 100
        min_target_pos_cost = 100
        min_rot_cost = 100
        min_target_rot_cost = 100

        sum_pos_cost = 0
        sum_target_pos_cost = 0
        sum_rot_cost = 0
        sum_target_rot_cost = 0

        x_c = x[:, 0, :]
        predicted = [x_c[:, self.phase_dim:self.phase_dim + self.pose_dim].detach().unsqueeze(1)]
        autoregress_bools = torch.randn(n) < self.autoregress_prob
        for i in range(1, n):
            y_c = y[:, i - 1, :]

            self.generationModel.reset_hidden(batch_size=y_c.shape[0])
            out, z, mu, logvar = self(x_c)
            recon = torch.cat(out, dim=1)
            predicted.append(out[1].unsqueeze(1))
            loss = self.loss_fn(recon, y_c)
            kl_loss = self.middle_layer.loss_function(z, mu, logvar)

            recon_loss = loss.detach()
            kl_loss = kl_loss.detach()

            pos_cost_x = torch.mean(torch.abs(recon[:, self.in_slices[-2]:self.posCost_dim]))
            rot_cost_x = torch.mean(torch.abs(recon[:, self.posCost_dim:]))

            pos_cost_y = torch.mean(torch.abs(y_c[:, self.in_slices[-2]:self.posCost_dim]))
            rot_cost_y = torch.mean(torch.abs(y_c[:, self.posCost_dim:]))

            _, pos_loss, rot_loss = self.pose_autoencoder.loss(out[1], y_c[:, self.phase_dim:self.phase_dim + self.pose_dim])

            tot_pos_loss += pos_loss.detach()
            tot_rot_loss += rot_loss.detach()

            min_pos_cost = min(pos_cost_x, min_pos_cost)
            min_target_pos_cost = min(pos_cost_y, min_target_pos_cost)
            min_rot_cost = min(rot_cost_x, min_rot_cost)
            min_target_rot_cost = min(rot_cost_y, min_target_rot_cost)

            sum_pos_cost += pos_cost_x
            sum_target_pos_cost += pos_cost_y
            sum_rot_cost += rot_cost_x
            sum_target_rot_cost += rot_cost_y

            tot_recon_loss += recon_loss
            tot_loss += recon_loss + kl_loss
            tot_kl_loss += kl_loss

            if autoregress_bools[i]:
                x_c = torch.cat(out, dim=1).detach()
            else:
                x_c = x[:, i, :]

        predicted = torch.cat(predicted, dim=1)

        tot_loss /= float(n)
        tot_recon_loss /= float(n)
        tot_kl_loss /= float(n)
        tot_pos_loss /= float(n)
        tot_rot_loss /= float(n)
        return tot_loss, tot_pos_loss, tot_rot_loss, tot_recon_loss, tot_kl_loss, predicted, \
               min_pos_cost, min_rot_cost, min_target_pos_cost, min_target_rot_cost, \
               sum_pos_cost, sum_rot_cost, sum_target_pos_cost, sum_target_rot_cost

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_shape = y.shape
        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss, tot_pos_loss, tot_rot_loss, recon_loss, kl_loss, predicted, \
               min_pos_cost, min_rot_cost, min_target_pos_cost, min_target_rot_cost,\
               sum_pos_cost, sum_rot_cost, sum_target_pos_cost, sum_target_rot_cost = self.step(x,y)

        if self.use_advLoss:
            d_fake = self.pose_autoencoder.convDiscriminator(predicted.view(y_shape[0], y_shape[1], -1).unsqueeze(1))
            g_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
            tot_adv_loss = g_loss.detach()
            self.log("ptl/train_adv_loss", tot_adv_loss)

        self.log("ptl/train_loss", loss, prog_bar=True)
        self.log("ptl/train_recon_loss", recon_loss)
        self.log("ptl/train_kl_loss", kl_loss)
        self.log("ptl/train_pos_loss", tot_pos_loss)
        self.log("ptl/train_rot_loss", tot_rot_loss)

        self.log("ptl/train_min_pos_cost", min_pos_cost)
        self.log("ptl/train_min_target_pos_cost", min_target_pos_cost)
        self.log("ptl/train_min_rot_cost", min_rot_cost)
        self.log("ptl/train_min_target_rot_cost", min_target_rot_cost)

        self.log("ptl/train_sum_pos_cost", sum_pos_cost)
        self.log("ptl/train_sum_target_pos_cost", sum_target_pos_cost)
        self.log("ptl/train_sum_rot_cost", sum_rot_cost)
        self.log("ptl/train_sum_target_rot_cost", sum_target_rot_cost)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_shape = y.shape
        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss, tot_pos_loss, tot_rot_loss, recon_loss, kl_loss, predicted, \
               min_pos_cost, min_rot_cost, min_target_pos_cost, min_target_rot_cost,\
               sum_pos_cost, sum_rot_cost, sum_target_pos_cost, sum_target_rot_cost = self.step_eval(x,y)

        if self.use_advLoss:
            d_fake = self.pose_autoencoder.convDiscriminator(predicted.view(y_shape[0], y_shape[1], -1).unsqueeze(1))
            g_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
            tot_adv_loss = g_loss.detach()
            self.log("ptl/val_adv_loss", tot_adv_loss, prog_bar=True)

        self.log("ptl/val_pos_loss", tot_pos_loss,prog_bar=True)
        self.log("ptl/val_rot_loss", tot_rot_loss,prog_bar=True)

        self.log("ptl/val_min_pos_cost", min_pos_cost)
        self.log("ptl/val_min_target_pos_cost", min_target_pos_cost)
        self.log("ptl/val_min_rot_cost", min_rot_cost)
        self.log("ptl/val_min_target_rot_cost", min_target_rot_cost)

        self.log("ptl/val_sum_pos_cost", sum_pos_cost)
        self.log("ptl/val_sum_target_pos_cost", sum_target_pos_cost)
        self.log("ptl/val_sum_rot_cost", sum_rot_cost)
        self.log("ptl/val_sum_target_rot_cost", sum_target_rot_cost)
        self.log("ptl/val_loss", loss, prog_bar=True)
        self.log("ptl/val_recon_loss", recon_loss, prog_bar=True)
        self.log("ptl/val_kl_loss", kl_loss, prog_bar=True)
        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_shape = y.shape
        x = x.view(-1, self.seq_len, x.shape[-1])
        y = y.view(-1, self.seq_len, y.shape[-1])
        loss, tot_pos_loss, tot_rot_loss, recon_loss, kl_loss, predicted, \
               min_pos_cost, min_rot_cost, min_target_pos_cost, min_target_rot_cost,\
               sum_pos_cost, sum_rot_cost, sum_target_pos_cost, sum_target_rot_cost = self.step_eval(x,y)

        if self.use_advLoss:
            d_fake = self.pose_autoencoder.convDiscriminator(predicted.view(y_shape[0], y_shape[1], -1).unsqueeze(1))
            g_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
            tot_adv_loss = g_loss.detach()
            self.log("ptl/test_adv_loss", tot_adv_loss, prog_bar=True)

        self.log("ptl/test_loss", loss, prog_bar=True)
        self.log("ptl/test_pos_loss", tot_pos_loss, prog_bar=True)
        self.log("ptl/test_rot_loss", tot_rot_loss, prog_bar=True)

        self.log("ptl/test_min_pos_cost", min_pos_cost, prog_bar=True)
        self.log("ptl/test_min_target_pos_cost", min_target_pos_cost, prog_bar=True)
        self.log("ptl/test_min_rot_cost", min_rot_cost, prog_bar=True)
        self.log("ptl/test_min_target_rot_cost", min_target_rot_cost, prog_bar=True)

        self.log("ptl/test_sum_pos_cost", sum_pos_cost, prog_bar=True)
        self.log("ptl/test_sum_target_pos_cost", sum_target_pos_cost, prog_bar=True)
        self.log("ptl/test_sum_rot_cost", sum_rot_cost, prog_bar=True)
        self.log("ptl/test_sum_target_rot_cost", sum_target_rot_cost, prog_bar=True)
        self.log("ptl/test_recon_loss", recon_loss, prog_bar=True)
        self.log("ptl/test_kl_loss", kl_loss, prog_bar=True)
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
                 "use_label":self.use_label,
                 "use_adv_loss":self.use_advLoss,
                 }

        if not os.path.exists(path):
            os.mkdir(path)
        filePath = os.path.join(path, str(best_val_loss) + ".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath

    @staticmethod
    def load_checkpoint(filename, Model, MiddleModel:nn.Module=None):
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        pose_autoencoder = MLP.load_checkpoint(obj["pose_autoencoder_path"])
        cost_encoder = MLP.load_checkpoint(obj["cost_encoder_path"])
        generationModel = Model.load_checkpoint(obj["motionGenerationModelPath"])

        model = MotionGenerationModel(config=obj["config"], feature_dims=obj["feature_dims"], Model=Model,
                                      use_advLoss=obj["use_adv_loss"],
                                      input_slicers=obj["in_slices"], output_slicers=obj["out_slices"],
                                      name=obj["name"])
        if MiddleModel is None:
            MiddleModel = nn.Sequential()
        else:
            MiddleModel.load_state_dict(obj["middle_layer_dict"])

        model.middle_layer = MiddleModel
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

    def freeze(self, flag=False) -> None:
        self.pose_autoencoder.freeze(flag)
        self.cost_encoder.freeze(flag)
        self.generationModel.freeze(flag)

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
