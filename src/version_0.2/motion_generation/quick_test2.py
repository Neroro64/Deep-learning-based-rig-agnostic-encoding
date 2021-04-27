from MoE import MoE
from GRU import GRU
from LSTM import LSTM
from MotionGeneration import MotionGenerationModel
from MotionGenerationRNN import MotionGenerationModelRNN
from MotionGenerationBatch import MotionGenerationModelBatch
import os, sys
sys.path.append("../rig_agnostic_encoding/models/")
from MLP import MLP
from VAE import VAE

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


config = {
    "hidden_dim": 256,
    "k": 32,
    "lr": 1e-4,
    "batch_size": 10,
    "keep_prob": .2,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.Adam,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":2000, "gamma":.1},
    "basis_func":"gaussian",
    "n_centroid":10,
    "k_experts": 4,
    "gate_size": 32,
    "g_hidden_dim": 128,
    "num_layers": 4,
    "autoregress_prob":.5,
    "autoregress_inc":.5,
    "autoregress_ep":3,
    "cost_hidden_dim":12,
    "device":"cpu"
    }


def test2():
    Ms = [ GRU, LSTM]
    name = ["MoE", "GRU", "LSTM"]

    X = torch.randn((100, 8, 30))
    X2 = torch.randn((100, 8, 40))
    Y = X
    Y2 = X2

    feature_dims = {
        "phase_dim":5,
        "pose_dim":15,
        "cost_dim":10,
        "g_input_dim": config["k"] + config["cost_hidden_dim"],
        "g_output_dim":5 + config["k"] + 10
    }

    feature_dims2 = {
        "phase_dim": 5,
        "pose_dim": 25,
        "cost_dim": 10,
        "g_input_dim": config["k"] + config["cost_hidden_dim"],
        "g_output_dim":5 + config["k"] + 10
    }

    in_slice = [5, 15, 10]
    in_slice2 = [5, 25, 10]
    out_slice = [5, config["k"], 10]
    out_slice2 = [5, config["k"], 10]


    dataset = TensorDataset(X, Y)
    dataset2 = TensorDataset(X2, Y2)
    train_set, val_set, test_set = random_split(dataset, [80, 10, 10])
    train_set2, val_set2, test_set2 = random_split(dataset2, [80, 10, 10])
    train_loader = DataLoader(train_set, batch_size=10)
    train_loader2 = DataLoader(train_set2, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    val_loader2 = DataLoader(val_set2, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)
    test_loader2 = DataLoader(test_set2, batch_size=10)

    for nam, M in zip(name, Ms):
        pose_encoder = MLP(config=config, dimensions=[15])
        pose_encoder2 = MLP(config=config, dimensions=[15], pose_labels=torch.arange(63).unsqueeze(0))
        pose_encoder3 = MLP(config=config, dimensions=[25, config["hidden_dim"], config["hidden_dim"], config["k"]], pose_labels=torch.arange(63).unsqueeze(0))

        m1 = MotionGenerationModelRNN(config=config, Model=M, pose_autoencoder=pose_encoder, feature_dims=feature_dims,
                                   input_slicers=in_slice, output_slicers=out_slice,
                                   train_set=train_set, val_set=val_set, test_set=test_set,
                                   )
        m2 = MotionGenerationModelRNN(config=config, Model=M, pose_autoencoder=pose_encoder2, feature_dims=feature_dims,
                                   input_slicers=in_slice, output_slicers=out_slice,
                                   train_set=train_set, val_set=val_set, test_set=test_set,
                                   )


        print("-"*50, nam, "-"*50)
        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(m1, test_loader)
        trainer.fit(m1, train_loader, val_loader,)
        print(nam, "-TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(m1, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

        print("-" * 50, nam, "-" * 50)
        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-" * 50)
        res1 = trainer.test(m2, test_loader)
        trainer.fit(m2, train_loader, val_loader, )
        print(nam, "-TEST RESULTS AFTER", "-" * 50)
        res2 = trainer.test(m2, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])

        m2.swap_pose_encoder(pose_encoder=pose_encoder3, input_dim=in_slice2, output_dim=out_slice2,feature_dims=feature_dims2, freeze=True)
        print("-" * 50, nam, "-" * 50)
        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-" * 50)
        res1 = trainer.test(m2, test_loader2)
        trainer.fit(m2, train_loader2, val_loader2)
        print(nam, "-TEST RESULTS AFTER", "-" * 50)
        res2 = trainer.test(m2, test_loader2)
        print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])


def test1():
    M = LSTM
    sample = torch.randn((32, 10))
    phase = torch.randn((32, 6))

    model = M(config=config, dimensions=[10, 10], phase_input_dim=6)
    x = torch.cat((sample,phase), dim=1).unsqueeze(1)
    model(x)

    f1 = model.save_checkpoint(best_val_loss=1)
    model = M.load_checkpoint(f1)
    model(x)


# test1()
test2()
# test3()
# test4()
