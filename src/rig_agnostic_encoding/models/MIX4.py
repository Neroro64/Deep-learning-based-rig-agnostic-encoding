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

class MIX4(pl.LightningModule):
    def __init__(self, model, config:dict=None, dim1:int=None, dim2:int=None, dim3:int=None, dim4:int=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model", load=False):

        super(MIX4, self).__init__()
        self.name = name
        if not load:
            if not config is None:
                self.learning_rate = config["lr"]
                self.loss_fn = config["loss_fn"]
                self.batch_size = config["batch_size"]

            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3
            self.dim4 = dim4

            self.model1 = model(config=config, dimensions=[dim1], name="M1")
            self.model2 = model(config=config, dimensions=[dim2], name="M2")
            self.model3 = model(config=config, dimensions=[dim3], name="M3")
            self.model4 = model(config=config, dimensions=[dim4], name="M4")

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.best_val_loss = np.inf

    def forward(self, x:torch.Tensor) -> (torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor):
        i1 = self.dim1
        i2 = i1+self.dim2
        i3 = i2+self.dim3
        input1, input2, input3, input4 = \
            x[:, :i1], x[:, i1:i2], x[:, i2:i3], x[:, i3:]

        h1 = self.model1.encode(input1)
        h2 = self.model2.encode(input2)
        h3 = self.model3.encode(input3)
        h4 = self.model4.encode(input4)

        out1, out2, out3, out4 = \
            self.model1.decode(h1), self.model2.decode(h2), self.model3.decode(h3), self.model4.decode(h4)
        return (out1, out2, out3, out4), (h1, h2, h3, h4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, h = self(x)
        loss = self.loss_fn(out, h, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out, h = self(x)
        loss = self.loss_fn(out, h, y)


        self.log('ptl/val_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        out, h = self(x)
        loss = self.loss_fn(out, h, y)


        self.log('ptl/test_loss', loss, prog_bar=True)
        return {"val_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint()

    def save_checkpoint(self, checkpoint_dir=MODEL_PATH):
        path = os.path.join(checkpoint_dir, self.name)
        loss = self.best_val_loss.cpu().numpy()

        model_path1 = self.model1.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)
        model_path2 = self.model2.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)
        model_path3 = self.model3.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)
        model_path4 = self.model4.save_checkpoint(best_val_loss=loss, checkpoint_dir=path)


        model = {"name":self.name,
                 "model1_name":self.model1.name,
                 "model1_single_module":self.model1.single_module, "model1_path" : model_path1,
                 "model2_name":self.model2.name,
                 "model2_single_module":self.model2.single_module, "model2_path" : model_path2,
                 "model3_name":self.model3.name,
                 "model3_single_module":self.model3.single_module, "model3_path" : model_path3,
                 "model4_name":self.model4.name,
                 "model4_single_module":self.model4.single_module, "model4_path" : model_path4,
                 }

        if not os.path.exists(path):
            os.mkdir(path)
        with bz2.BZ2File(os.path.join(path,
                                      str(loss)+"."+str(self.k)+".pbz2"), "w") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_checkpoint(model, filename):
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        model1 = model.load_checkpoint(obj["model1_path"])
        model2 = model.load_checkpoint(obj["model2_path"])
        model3 = model.load_checkpoint(obj["model3_path"])
        model4 = model.load_checkpoint(obj["model4_path"])

        model = MIX4(model=model, name=obj["name"], load=True)
        model.model1 = model1
        model.model2 = model2
        model.model3 = model3
        model.model4 = model4

        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True)