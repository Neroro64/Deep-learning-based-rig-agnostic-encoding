from MLP import MLP
from RBF import RBF
from VAE import VAE
from MLP_Adversarial import MLP_ADV
from MLP_MIX import MLP_MIX

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


config = {
    "hidden_dim": 256,
    "k": 32,
    "z_dim": 32,
    "lr": 1e-2,
    "batch_size": 10,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.AdamW,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":80, "gamma":.9},
    "basis_func":"gaussian",
    "n_centroid":64,
    "k_experts": 4,
    "gate_size": 128,
    "g_hidden_dim": 512,
    "num_layers": 4,
    "autoregress_prob":0.3,
    "autoregress_inc":20,
    "autoregress_ep":20,
    "autoregress_max_prob":1,
    "cost_hidden_dim":128,
    "seq_len":13,
    "device":"cpu"
    }

def test1():
    X = torch.randn((100, 100, 30))
    Y = X

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])

    model = MLP(config=config, dimensions=[30], name="MLP",
                train_set=train_set, val_set=val_set, test_set=test_set)

    print("-" * 50, "-" * 50)
    trainer = pl.Trainer(max_epochs=5)
    print("-TEST RESULTS BEFORE", "-" * 50)
    res1 = trainer.test(model)
    trainer.fit(model)
    print("-TEST RESULTS AFTER", "-" * 50)
    res2 = trainer.test(model)
    print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])

    path = model.save_checkpoint(best_val_loss=0.001)
    model = MLP.load_checkpoint(path)
def test2():
    X = torch.randn((100, 100, 30))
    Y = X

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])

    model = MLP_ADV(config=config, dimensions=[30], name="MLP",
                    h_dim=100, w_dim=30,
                train_set=train_set, val_set=val_set, test_set=test_set)

    print("-" * 50, "-" * 50)
    trainer = pl.Trainer(max_epochs=5)
    print("-TEST RESULTS BEFORE", "-" * 50)
    res1 = trainer.test(model)
    trainer.fit(model)
    print("-TEST RESULTS AFTER", "-" * 50)
    res2 = trainer.test(model)
    print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])

    path = model.save_checkpoint(best_val_loss=0.001)
    model = MLP.load_checkpoint(path)


def test3():
    # Ms = [MLP_MIX, RBF, VAE]
    Ms = [RBF]
    name = ["RBF"]
    # name = ["MLP_MIX", "RBF", "VAE"]

    X = torch.randn((100, 100, 30))
    Y = X

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])
    train_loader = DataLoader(train_set, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)

    for nam, M in zip(name, Ms):
        model = M(config=config, input_dims=[30], name=nam, pose_labels=torch.arange(63).unsqueeze(0))

        print("-"*50, nam, "-"*50)
        trainer = pl.Trainer(max_epochs=15)
        print(nam, "-TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(model, test_loader)
        trainer.fit(model, train_loader, val_loader,)
        print(nam, "-TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(model, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

        path = model.save_checkpoint(best_val_loss=0.001)
        model = M.load_checkpoint(path)




# test1()
# test2()
test3()
