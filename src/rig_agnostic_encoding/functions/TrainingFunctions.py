from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import sys, os
sys.path.append("..")
from settings.GlobalSettings import DATA_PATH, MODEL_PATH
import functions.DataProcessingFunctions as Data

def train(config=None, model=None , dim=None,
            train_set=None, val_set=None, test_set=None,
            num_epochs=300, num_cpus=24, num_gpus=1, model_name="model"):


    m = model[0](config=config, dim=dim,
                  train_set=train_set, val_set=val_set, test_set=test_set, name=model_name)

    fit(m, model_name, num_epochs, num_gpus)


def train4(config=None, model=None, dim1=None, dim2=None, dim3=None, dim4=None,
            train_set=None, val_set=None, test_set=None,
            num_epochs=300, num_cpus=24, num_gpus=1, model_name="model"):

    m = model[0](config=config, model=model[1], dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4,
                  train_set=train_set, val_set=val_set, test_set=test_set, name=model_name)

    fit(m, model_name, num_epochs, num_gpus)


def train4_withLabel(config=None, model=None, dim1=None, dim2=None, dim3=None,dim4=None, extra_feature_len:int=0, extra_feature_len2:int=0,
                 train_set=None, val_set=None, test_set=None,
                 num_epochs=300, num_cpus=24, num_gpus=1, model_name="model"):

    m = model[0](config=config, model=model[1], dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4,
                extra_feature_len=extra_feature_len, extra_feature_len2=extra_feature_len2,
                train_set=train_set, val_set=val_set, test_set=test_set, name=model_name)

    fit(m, model_name, num_epochs, num_gpus)


def fit(model=None, model_name="model", num_epochs=300, num_gpus=1):
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name, version="0.0"),
        progress_bar_refresh_rate=20,
        callbacks=[
            TuneReportCallback({"loss": "avg_val_loss", }, on="validation_end"),
            EarlyStopping(monitor="avg_val_loss")
        ],
        precision=16,
    )
    trainer.fit(model)


def _tune(model, train_set:Dataset, val_set:Dataset, dim:int,
          config:dict, EPOCHS:int=300,
          n_gpu=1, n_samples=20, model_name="model",
          ):


    scheduler = ASHAScheduler(max_t = EPOCHS, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["k", "lr", "batch_size", "loss_fn"],
        metric_columns=["loss", "training_iteration"],
        max_error_rows=5,
        max_progress_rows=5,
        max_report_frequency=10)
    analysis = tune.run(
        tune.with_parameters(
            train,
            model=model,
            dim=dim,
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
        verbose=False
    )

    print("-"*70)
    print("Done")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best achieved loss was: ", analysis.best_result)
    print("-"*70)


def tune4(model, train_set:Dataset, val_set:Dataset, dims:list,
          config:dict, EPOCHS:int=300,
          n_gpu=1, n_samples=20, model_name="model",
          ):

    dim1, dim2, dim3, dim4 = dims[0], dims[1], dims[2],dims[3]

    scheduler = ASHAScheduler(max_t = EPOCHS, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["k", "lr", "batch_size", "loss_fn"],
        metric_columns=["loss", "training_iteration"],
        max_error_rows=5,
        max_progress_rows=5,
        max_report_frequency=10)
    analysis = tune.run(
        tune.with_parameters(
            train4,
            model=model,
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
        verbose=False
    )

    print("-"*70)
    print("Done")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best achieved loss was: ", analysis.best_result)
    print("-"*70)


def tune4_withLabel(model, train_set:Dataset, val_set:Dataset, dims:list,
          config:dict, EPOCHS:int=300, extra_feature_len:int=0,extra_feature_len2:int=0,
          n_gpu=1, n_samples=20, model_name="model",
          ):

    dim1, dim2, dim3, dim4 = dims[0], dims[1], dims[2],dims[3]

    scheduler = ASHAScheduler(max_t = EPOCHS, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["k", "lr", "batch_size", "loss_fn"],
        metric_columns=["loss", "training_iteration"],
        max_error_rows=5,
        max_progress_rows=5,
        max_report_frequency=10)

    analysis = tune.run(
        tune.with_parameters(
            train4_withLabel,
            model=model,
            dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4,
            extra_feature_len=extra_feature_len,
            extra_feature_len2=extra_feature_len2,
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
        verbose=False
    )

    print("-"*70)
    print("Done")
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best achieved loss was: ", analysis.best_result)
    print("-"*70)


def train_multi_model_withLabel(model, datapaths:list, featureList:list, config:dict=None,
                      extra_feature_len:int=0, extra_feature_len2:int=0,
                      n_samples:int=30, model_name:str="model"):
    # load data
    datasets1 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[0]]
    datasets2 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[1]]
    datasets3 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[2]]
    datasets4 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[3]]

    train_set1, val_set1, test_set1 = Data.prepare_data_withLabel(datasets1, featureList, extra_feature_len=extra_feature_len)
    train_set2, val_set2, test_set2 = Data.prepare_data_withLabel(datasets2, featureList, extra_feature_len=extra_feature_len2)
    train_set3, val_set3, test_set3 = Data.prepare_data_withLabel(datasets3, featureList, extra_feature_len=extra_feature_len)
    train_set4, val_set4, test_set4 = Data.prepare_data_withLabel(datasets4, featureList, extra_feature_len=extra_feature_len2)

    dims = [len(train_set1[0][0]),len(train_set2[0][0]),len(train_set3[0][0]),len(train_set4[0][0])]
    train_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(train_set1, train_set2, train_set3, train_set4)]
    val_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(val_set1, val_set2, val_set3, val_set4)]

    tune4_withLabel(model=model, train_set=train_set, val_set=val_set, config=config, dims=dims,
          extra_feature_len=extra_feature_len, extra_feature_len2=extra_feature_len2,
          n_samples=n_samples, model_name=model_name)

    path=os.path.join(MODEL_PATH, model_name)
    Data.clean_checkpoints(num_keep=3, path=path)
    Data.save_testData([test_set1, test_set2, test_set3, test_set4], path=path)


def train_multi_model(model, datapaths:list, featureList:list, config:dict=None,
                      n_samples:int=30, model_name:str="model"):
    # load data
    datasets1 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[0]]
    datasets2 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[1]]
    datasets3 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[2]]
    datasets4 = [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[3]]

    train_set1, val_set1, test_set1 = Data.prepare_data(datasets1, featureList)
    train_set2, val_set2, test_set2 = Data.prepare_data(datasets2, featureList)
    train_set3, val_set3, test_set3 = Data.prepare_data(datasets3, featureList)
    train_set4, val_set4, test_set4 = Data.prepare_data(datasets4, featureList)

    dims = [len(train_set1[0][0]),len(train_set2[0][0]),len(train_set3[0][0]),len(train_set4[0][0])]
    train_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(train_set1, train_set2, train_set3, train_set4)]
    val_set = [(torch.cat([x[0],y[0],z[0],w[0]],dim=0),torch.cat([x[1],y[1],z[1],w[1]],dim=0))
                 for x, y, z, w in zip(val_set1, val_set2, val_set3, val_set4)]

    tune4(model=model, train_set=train_set, val_set=val_set, config=config, dims=dims,
          n_samples=n_samples, model_name=model_name)

    path=os.path.join(MODEL_PATH, model_name)
    Data.clean_checkpoints(num_keep=3, path=path)
    Data.save_testData([test_set1, test_set2, test_set3, test_set4], path=path)


def train_single_model(model, datapaths: list, featureList: list, config: dict = None,
                          n_samples: int = 30, model_name: str = "model"):
        # load data
        datasets= [Data.load(os.path.join(DATA_PATH, path)) for path in datapaths[0]]
        train_set, val_set, test_set = Data.prepare_data(datasets, featureList)

        dim = len(train_set[0][0])

        _tune(model=model, train_set=train_set, val_set=val_set, config=config, dim=dim,
                        n_samples=n_samples, model_name=model_name)

        path = os.path.join(MODEL_PATH, model_name)
        Data.clean_checkpoints(num_keep=3, path=path)
        Data.save_testData([test_set], path=path)