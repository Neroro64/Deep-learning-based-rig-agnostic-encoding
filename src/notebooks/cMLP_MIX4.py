


import os, sys, math, time
import numpy as np
import numpy.linalg as la
import plotly.graph_objects as go
import plotly.express as ex
from plotly.subplots import make_subplots
import pandas as pd

import json as js
import _pickle as pickle
import bz2
import ray

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from collections import OrderedDict

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import shutil
import tempfile
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from cytoolz import sliding_window
sys.path.append("../")
import func


ray.shutdown()
# Prepare train data
DATA_PATH = "/home/nuoc/Documents/MEX/data"
MODEL_PATH = "/home/nuoc/Documents/MEX/models"
RESULTS_PATH = "/home/nuoc/Documents/MEX/results"


# Test torch lightning + ray tune

class MLP(pl.LightningModule):
    def __init__(self, config:dict=None, dimensions:list=None,
                 train_set=None, val_set=None, test_set=None,
                 keep_prob:float=.2, name:str="model", load=False,
                 single_module:int=0):

        super(MLP, self).__init__()
        self.name = name
        self.dimensions = dimensions
        self.keep_prob = keep_prob
        self.single_module = single_module

        if load:
            self.build()
        else:
            self.k = config["k"]
            self.learning_rate = config["lr"]
            self.loss_fn = config["ae_loss_fn"]
            self.batch_size = config["batch_size"]

            dimensions.append(self.k)
            self.dimensions = dimensions
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
                layers.append(("fc"+str(i), nn.Linear(size[0], size[1])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), nn.ELU()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.encoder = nn.Sequential(OrderedDict(layers))
        else:
            self.encoder = nn.Sequential()

        if self.single_module == 0 or self.single_module == 1:
            layers = []
            for i, size in enumerate(layer_sizes[-1::-1]):
                layers.append(("fc"+str(i), nn.Linear(size[1], size[0])))
                if i < len(self.dimensions)-2:
                    layers.append(("act"+str(i), nn.ELU()))
                    layers.append(("drop"+str(i+1), nn.Dropout(self.keep_prob)))
            self.decoder = nn.Sequential(OrderedDict(layers))
        else:
            self.decoder = nn.Sequential()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

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
            self.save_checkpoint()

    def save_checkpoint(self, checkpoint_dir=MODEL_PATH):
        model = {"k":self.k, "dimensions":self.dimensions,"keep_prob":self.keep_prob, "name":self.name,
                 "encoder":self.encoder.state_dict(),
                 "decoder":self.decoder.state_dict()}


        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, "."+str(self.k)+".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath
    def load(self, state_dict1: 'OrderedDict[str, Tensor]'=None, state_dict2: 'OrderedDict[str, Tensor]'=None,
                        strict: bool = True):

        if self.single_module == -1 or self.single_module == 0:
            self.encoder.load_state_dict(state_dict1, strict)
        if self.single_module == 1 or self.single_module == 0:
            self.decoder.load_state_dict(state_dict2, strict)
    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)
        model = MLP(name=obj["name"], dimensions=obj["dimensions"], load=True)
        model.encoder.load_state_dict(obj["encoder"])
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

class MIX(pl.LightningModule):
    def __init__(self, config:dict=None, dim1:int=None, dim2:int=None, dim3:int=None, dim4:int=None,
                 train_set=None, val_set=None, test_set=None,
                 name:str="model"):

        super(MIX, self).__init__()
        self.name = name
        if not config is None:
            self.k = config["k"]
            self.hidden_dim = config["hidden_dim"]
            self.learning_rate = config["lr"]
            self.loss_fn = config["loss_fn"]
            self.batch_size = config["batch_size"]

        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.dimension1 = [dim1 , self.hidden_dim , self.k]
        self.dimension2 = [dim2 , self.hidden_dim , self.k]
        self.dimension3 = [dim3 , self.hidden_dim , self.k]
        self.dimension4 = [dim4 , self.hidden_dim , self.k]

        self.model1 = MLP(config=config, dimensions=self.dimension1, name="M1")
        self.model2 = MLP(config=config, dimensions=self.dimension2, name="M2")
        self.model3 = MLP(config=config, dimensions=self.dimension3, name="M3")
        self.model4 = MLP(config=config, dimensions=self.dimension4, name="M4")

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
        h1, h2, h3, h4 = \
            self.model1.encoder(input1), self.model2.encoder(input2), self.model3.encoder(input3), self.model4.encoder(input4)

        out1, out2, out3, out4 = \
            self.model1.decoder(h1), self.model2.decoder(h2), self.model3.decoder(h3), self.model4.decoder(h4)
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
        model_path1 = self.model1.save_checkpoint(checkpoint_dir=MODEL_PATH)
        model_path2 = self.model2.save_checkpoint(checkpoint_dir=MODEL_PATH)
        model_path3 = self.model3.save_checkpoint(checkpoint_dir=MODEL_PATH)
        model_path4 = self.model4.save_checkpoint(checkpoint_dir=MODEL_PATH)


        model = {"name":self.name,
                 "model1_name":self.model1.name,
                 "k": self.k,
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
                                      str(self.best_val_loss.cpu().numpy())+"."+str(self.k)+".pbz2"), "w") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_checkpoint(filename):
        # return torch.load(os.path.join(checkpoint_dir,filename))
        with bz2.BZ2File(filename, "rb") as f:
            obj = pickle.load(f)

        model1 = MLP.load_checkpoint(obj["model1_path"])
        model2 = MLP.load_checkpoint(obj["model2_path"])
        model3 = MLP.load_checkpoint(obj["model3_path"])
        model4 = MLP.load_checkpoint(obj["model4_path"])

        model = MIX(name=obj["name"])
        model.model1 = model1
        model.model2 = model2
        model.model3 = model3
        model.model4 = model4

        return model
        # self.encoder.load_state_dict(obj["encoder"])
        # self.decoder.load_state_dict(obj["decoder"])
        # self.best_val_loss = obj["val_loss"]

    def setup_data(self): pass
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=True)



def train_tune(config, dim1, dim2, dim3,dim4,
                 train_set=None, val_set=None, test_set=None,
                 num_epochs=300, num_cpus=24, num_gpus=1, model_name="model"):

    model = MIX(config=config, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4,
                train_set=train_set, val_set=val_set, test_set=test_set, name=model_name)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name, version="0.0"),
        progress_bar_refresh_rate=20,
        callbacks=[
            TuneReportCallback({"loss":"avg_val_loss",}, on="validation_end"),
            EarlyStopping(monitor="avg_val_loss")
        ],
        precision=16,
    )
    trainer.fit(model)



def prepare_data(datasets:list, featureList:list,
                 train_ratio:float=0.8, val_ratio:float=0.2, test_size:int=100, SEED:int=2021):
   # process data
    data = [func.processData(d, featureList, shutdown=False) for d in datasets]
    input_data = [np.vstack(d) for d in data]
    x_tensors = [func.normaliseT(torch.from_numpy(x).float()) for x in input_data]
    y_tensors = [torch.from_numpy(x).float() for x in input_data]

    # prepare datasets
    test_sets = [(x_tensor[-test_size:], y_tensor[-test_size:]) for x_tensor, y_tensor in zip(x_tensors, y_tensors)]
    x_training = torch.vstack([x_tensor[:-test_size] for x_tensor in x_tensors])
    y_training = torch.vstack([y_tensor[:-test_size] for y_tensor in y_tensors])
    dataset = TensorDataset(x_training, y_training)
    N = len(x_training)

    train_ratio = int(train_ratio*N)
    val_ratio = int(val_ratio*N)
    print("Train: ", train_ratio, ", Validation: ", val_ratio)
    train_set, val_set = random_split(dataset, [train_ratio, val_ratio], generator=torch.Generator().manual_seed(SEED))
    return train_set, val_set, test_sets

def train(train_set:Dataset, val_set:Dataset, dims:list,
          config:dict, EPOCHS:int=300,
          n_gpu=1, n_samples=20, model_name="model",
          ):

    dim1, dim2, dim3, dim4 = dims[0], dims[1], dims[2],dims[3]

    scheduler = ASHAScheduler(max_t = EPOCHS, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["k", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"],
        max_error_rows=5,
        max_progress_rows=5,
        max_report_frequency=10)
    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4,
            train_set = train_set, val_set = val_set,
            num_epochs = EPOCHS,
            num_gpus=n_gpu,
            model_name=model_name
        ),
        resources_per_trial= {"cpu":1, "gpu":n_gpu},
        metric="loss",
        mode="min",
        config=config,
        num_samples=n_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=model_name,
        verbose=False,
        checkpoint_freq=0,
        keep_checkpoints_num=1,
        checkpoint_score_attr="loss",
        checkpoint_at_end=True
    )

    print("-"*70)
    print("Done")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best achieved loss was: ", analysis.best_result)
    print("-"*70)

def clean_checkpoints(num_keep=3, path="../../models"):
    saved_checkpoints = []
    for dir, dname, files in os.walk(path):
        for fname in files:
            fname = fname.split(".")
            saved_checkpoints.append(fname)
        print("Num checkpoints in {}: {}".format(dir, len(saved_checkpoints)))

        saved_checkpoints.sort(key = lambda x: x[0] + x[1])
        for filename in saved_checkpoints[num_keep:]:
            os.remove(os.path.join(dir,".".join(filename)))
        break
    return ".".join(saved_checkpoints[0])

def test(model:torch.nn.Module, test_sets:list, loss_fn, set_names=list,
         save=True, path="../../results", model_name="model"):
    # Intra test performance
    with torch.no_grad():
        df = {}
        if set_names is None: set_names = np.arange(len(test_sets))
        for i,t1 in enumerate(test_sets):
            for j, t2 in enumerate(test_sets):
                x = t1[0]
                y = t2[1]
                out1, out2, h1, h2 = model(x)
                loss = loss_fn(out1, out2, h1, h2, y)
                df["{}-{}".format(set_names[i], set_names[j])] = [loss.cpu().numpy()]
                print("Test encoding {} to {}, MSE={:.2f}".format(set_names[i], set_names[j], loss))
        filepath = os.path.join(path, model_name)
        if not os.path.exists(filepath): os.mkdir(filepath)
        pd.DataFrame(df).to_csv(os.path.join(filepath, "tests.csv"))

def test2(model:torch.nn.Module, test_sets:list, loss_fn, set_names=list,
         save=True, path="../../results", model_name="model"):
    # Intra test performance
    with torch.no_grad():
        df = {}
        if set_names is None: set_names = np.arange(len(test_sets))
        for i,t1 in enumerate(test_sets):
            x = t1[0]
            y = t1[1]
            out, h = model(x)
            loss = loss_fn(out, h, y)
            df["{}".format(set_names[i])] = [loss.cpu().numpy()]
            print("Test encod>ing {} = {:.2f}".format(set_names[i], loss))
        filepath = os.path.join(path, model_name)
        if not os.path.exists(filepath): os.mkdir(filepath)
        pd.DataFrame(df).to_csv(os.path.join(filepath, "tests.csv"))


def train_multi_model(datapaths:list, featureList:list, config:dict=None,
                       n_samples:int=30, model_name:str="model", loss_fn=nn.functional.mse_loss,
                       dataset_names:list=None):
    # load data
    datasets1 = [func.load(os.path.join(DATA_PATH,path)) for path in datapaths[0]]
    datasets2 = [func.load(os.path.join(DATA_PATH,path)) for path in datapaths[1]]
    datasets3 = [func.load(os.path.join(DATA_PATH,path)) for path in datapaths[2]]
    datasets4 = [func.load(os.path.join(DATA_PATH,path)) for path in datapaths[3]]

    train_set1, val_set1, test_set1 = prepare_data(datasets1, featureList)
    train_set2, val_set2, test_set2 = prepare_data(datasets2, featureList)
    train_set3, val_set3, test_set3 = prepare_data(datasets3, featureList)
    train_set4, val_set4, test_set4 = prepare_data(datasets4, featureList)

    dims = [len(train_set1[0][0]),len(train_set2[0][0]),len(train_set3[0][0]),len(train_set4[0][0])]
    train_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(train_set1, train_set2, train_set3, train_set4)]
    val_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(val_set1, val_set2, val_set3, val_set4)]
    train(train_set=train_set, val_set=val_set, config=config, dims=dims,
          n_samples=n_samples, model_name=model_name)


    # test_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 # for x, y, z, w in zip(test_set1, test_set2, test_set3, test_set4)]
    # clean_checkpoints(path=MODEL_PATH)
    # best_model = MIX.load_checkpoint(best_model)
    return [test_set1, test_set2, test_set3, test_set4]
    # test2(best_model, test_set, loss_fn=loss_fn, set_names=dataset_names, path=RESULTS_PATH)


def mse_loss(out, h, y):
    return nn.functional.mse_loss(torch.cat((out[0], out[1], out[2], out[3]), dim=1), y)

def mse_mse_loss(out, h, y):
    i1 = out[0].size()[-1]
    i2 = i1 + out[1].size()[-1]
    i3 = i2 + out[2].size()[-1]
    y0, y1, y2, y3 = y[:, :i1], y[:, i1:i2],y[:, i2:i3], y[:, i3:]

    mse0 = nn.functional.mse_loss(out[0], y0)
    mse1 = nn.functional.mse_loss(out[1], y1)
    mse2 = nn.functional.mse_loss(out[2], y2)
    mse3 = nn.functional.mse_loss(out[3], y3)

    s_loss0 = nn.functional.mse_loss(h[0], h[1])
    s_loss1 = nn.functional.mse_loss(h[0], h[2])
    s_loss2 = nn.functional.mse_loss(h[0], h[3])
    s_loss3 = nn.functional.mse_loss(h[1], h[2])
    s_loss4 = nn.functional.mse_loss(h[1], h[3])
    s_loss5 = nn.functional.mse_loss(h[2], h[3])

    mse = (mse0 + mse1 + mse2 + mse3) / 4
    similarity_loss = (s_loss0 + s_loss1 + s_loss2 + s_loss3 + s_loss4 + s_loss5) / 6
    return (mse + similarity_loss) / 2.0

def mse_mae_loss(out, h, y):
    i1 = out[0].size()[-1]
    i2 = i1 + out[1].size()[-1]
    i3 = i2 + out[2].size()[-1]
    y0, y1, y2, y3 = y[:, :i1], y[:, i1:i2],y[:, i2:i3], y[:, i3:]

    mse0 = nn.functional.mse_loss(out[0], y0)
    mse1 = nn.functional.mse_loss(out[1], y1)
    mse2 = nn.functional.mse_loss(out[2], y2)
    mse3 = nn.functional.mse_loss(out[3], y3)

    s_loss0 = nn.functional.smooth_l1_loss(h[0], h[1])
    s_loss1 = nn.functional.smooth_l1_loss(h[0], h[2])
    s_loss2 = nn.functional.smooth_l1_loss(h[0], h[3])
    s_loss3 = nn.functional.smooth_l1_loss(h[1], h[2])
    s_loss4 = nn.functional.smooth_l1_loss(h[1], h[3])
    s_loss5 = nn.functional.smooth_l1_loss(h[2], h[3])

    mse = (mse0 + mse1 + mse2 + mse3) / 4
    similarity_loss = (s_loss0 + s_loss1 + s_loss2 + s_loss3 + s_loss4 + s_loss5) / 6
    return (mse + similarity_loss) / 2.0


def mae_mae_loss(out, h, y):
    i1 = out[0].size()[-1]
    i2 = i1 + out[1].size()[-1]
    i3 = i2 + out[2].size()[-1]
    y0, y1, y2, y3 = y[:, :i1], y[:, i1:i2],y[:, i2:i3], y[:, i3:]

    mse0 = nn.functional.smooth_l1_loss(out[0], y0)
    mse1 = nn.functional.smooth_l1_loss(out[1], y1)
    mse2 = nn.functional.smooth_l1_loss(out[2], y2)
    mse3 = nn.functional.smooth_l1_loss(out[3], y3)

    s_loss0 = nn.functional.smooth_l1_loss(h[0], h[1])
    s_loss1 = nn.functional.smooth_l1_loss(h[0], h[2])
    s_loss2 = nn.functional.smooth_l1_loss(h[0], h[3])
    s_loss3 = nn.functional.smooth_l1_loss(h[1], h[2])
    s_loss4 = nn.functional.smooth_l1_loss(h[1], h[3])
    s_loss5 = nn.functional.smooth_l1_loss(h[2], h[3])

    mse = (mse0 + mse1 + mse2 + mse3) / 4
    similarity_loss = (s_loss0 + s_loss1 + s_loss2 + s_loss3 + s_loss4 + s_loss5) / 6
    return (mse + similarity_loss) / 2.0

def mse_kl_loss(out, h, y):
    i1 = out[0].size()[-1]
    i2 = i1 + out[1].size()[-1]
    i3 = i2 + out[2].size()[-1]
    y0, y1, y2, y3 = y[:, :i1], y[:, i1:i2],y[:, i2:i3], y[:, i3:]

    mse0 = nn.functional.mse_loss(out[0], y0)
    mse1 = nn.functional.mse_loss(out[1], y1)
    mse2 = nn.functional.mse_loss(out[2], y2)
    mse3 = nn.functional.mse_loss(out[3], y3)

    s_loss0 = nn.functional.kl_div(h[0], h[1])
    s_loss1 = nn.functional.kl_div(h[0], h[2])
    s_loss2 = nn.functional.kl_div(h[0], h[3])
    s_loss3 = nn.functional.kl_div(h[1], h[2])
    s_loss4 = nn.functional.kl_div(h[1], h[3])
    s_loss5 = nn.functional.kl_div(h[2], h[3])

    mse = (mse0 + mse1 + mse2 + mse3) / 4
    similarity_loss = (s_loss0 + s_loss1 + s_loss2 + s_loss3 + s_loss4 + s_loss5) / 6
    return (mse + similarity_loss) / 2.0

def mse_nll_loss(out, h, y):
    i1 = out[0].size()[-1]
    i2 = i1 + out[1].size()[-1]
    i3 = i2 + out[2].size()[-1]
    y0, y1, y2, y3 = y[:, :i1], y[:, i1:i2],y[:, i2:i3], y[:, i3:]

    mse0 = nn.functional.mse_loss(out[0], y0)
    mse1 = nn.functional.mse_loss(out[1], y1)
    mse2 = nn.functional.mse_loss(out[2], y2)
    mse3 = nn.functional.mse_loss(out[3], y3)

    s_loss0 = nn.functional.nll_loss(h[0], h[1])
    s_loss1 = nn.functional.nll_loss(h[0], h[2])
    s_loss2 = nn.functional.nll_loss(h[0], h[3])
    s_loss3 = nn.functional.nll_loss(h[1], h[2])
    s_loss4 = nn.functional.nll_loss(h[1], h[3])
    s_loss5 = nn.functional.nll_loss(h[2], h[3])

    mse = (mse0 + mse1 + mse2 + mse3) / 4
    similarity_loss = (s_loss0 + s_loss1 + s_loss2 + s_loss3 + s_loss4 + s_loss5) / 6
    return (mse + similarity_loss) / 2.0



datapath1 = ["LOCO_R1-default-locomotion.pbz2",
             "LOCO_R1-default-locomotion-small.pbz2",
             "LOCO_R1-default-locomotion-large.pbz2"]
datapath2 = ["LOCO_R2-default-locomotion.pbz2",
             "LOCO_R2-default-locomotion-small.pbz2",
             "LOCO_R2-default-locomotion-large.pbz2"]
datapath3 = ["LOCO_R3-default-locomotion.pbz2",
             "LOCO_R3-default-locomotion-small.pbz2",
             "LOCO_R3-default-locomotion-large.pbz2"]
datapath4 = ["LOCO_R4-default-locomotion.pbz2",
             "LOCO_R4-default-locomotion-small.pbz2",
             "LOCO_R4-default-locomotion-large.pbz2"]
featureList = ["pos", "rotMat", "velocity"]




config = {
    "k": tune.randint(6, 256),
    "hidden_dim" : tune.choice([64, 128, 256, 512]),
    "lr": tune.loguniform(1e-2, 1e-7),
    "batch_size":tune.choice([5, 15, 30, 60]),
    "loss_fn":tune.choice([mse_mse_loss, mse_mae_loss, mae_mae_loss]),
    "ae_loss_fn":tune.choice([mse_loss])
}

train_multi_model([datapath1,datapath2,datapath3,datapath4], featureList, config,
                       n_samples=50, model_name="MIX4", loss_fn=mse_loss,
                       dataset_names=["R1","R2","R3","R4"])



# best_model = clean_checkpoints(path=os.path.join(MODEL_PATH, "MIX4"))





