from MLP import MLP
from RBF import RBF
from DEC import DEC
from VaDE import VaDE
from VAE import VAE
from MLP_MIX import MLP_MIX
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

config = {
    "hidden_dim": 256,
    "k": 256,
    "lr": 1e-4,
    "batch_size": 32,
    "keep_prob": .2,
    "basis_func":"gaussian",
    "n_centroid":10,
}
config2 = {
    "hidden_dim": 256,
    "k": 256,
    "lr": 1e-4,
    "batch_size": 32,
    "keep_prob": .2,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.Adam,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":2000, "gamma":.1},
    "basis_func":"gaussian",
    "n_centroid":10,
}

def test4():
    Ms = [MLP_MIX, RBF, DEC, VaDE, VAE]
    name = ["MLP_MIX", "RBF", "DEC", "VaDE", "VAE"]

    X = torch.randn((100, 30))
    X2 = torch.randn((100, 60))
    Y = X
    Y2 = X2

    dataset = TensorDataset(X, Y)
    dataset2 = TensorDataset(X2, Y2)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])
    train_set2, val_set2, test_set2 = random_split(dataset2, [70, 15, 15])
    train_loader = DataLoader(train_set, batch_size=10)
    train_loader2 = DataLoader(train_set2, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    val_loader2 = DataLoader(val_set2, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)
    test_loader2 = DataLoader(test_set2, batch_size=10)

    for nam, M in zip(name, Ms):
        model = M(config=config, input_dims=[10, 20], name=nam)
        model2 = M(config=config2, input_dims=[10, 20], name=nam, pose_labels=[torch.arange(63).unsqueeze(0),torch.arange(63).unsqueeze(0)])

        print("-"*50, nam, "-"*50)
        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(model, test_loader)
        trainer.fit(model, train_loader, val_loader,)
        print(nam, "-TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(model, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

        trainer = pl.Trainer(max_epochs=5)
        print("LABEL TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(model2, test_loader)
        trainer.fit(model2, train_loader, val_loader)
        print("LABEL TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(model2, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

        print("ADDING MODELS!!!!")
        model.add_models(input_dims=[30])
        model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)])

        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-" * 50)
        res1 = trainer.test(model, test_loader2)
        trainer.fit(model, train_loader2, val_loader2, )
        print(nam, "-TEST RESULTS AFTER", "-" * 50)
        res2 = trainer.test(model, test_loader2)
        print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])

        trainer = pl.Trainer(max_epochs=5)
        print("LABEL TEST RESULTS BEFORE", "-" * 50)
        res1 = trainer.test(model2, test_loader2)
        trainer.fit(model2, train_loader2, val_loader2)
        print("LABEL TEST RESULTS AFTER", "-" * 50)
        res2 = trainer.test(model2, test_loader2)
        print("IMPROVEMENT: ", res1[0]["test_loss"] - res2[0]["test_loss"])

        print("ADDING MODELS!!!!")
        model.add_models(input_dims=[30], freeze=True)
        model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)], freeze=True)

        trainer = pl.Trainer(max_epochs=5)
        print(nam, "-TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(model, test_loader)
        trainer.fit(model, train_loader, val_loader,)
        print(nam, "-TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(model, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

        trainer = pl.Trainer(max_epochs=5)
        print("LABEL TEST RESULTS BEFORE", "-"*50)
        res1 = trainer.test(model2, test_loader)
        trainer.fit(model2, train_loader, val_loader)
        print("LABEL TEST RESULTS AFTER", "-"*50)
        res2 = trainer.test(model2, test_loader)
        print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])


def test3():
    M = VAE
    name = "VAE"
    X = torch.randn((100, 30))
    Y = X

    model = M(config=config, input_dims=[10, 20], name=name)
    model2 = M(config=config2, input_dims=[10, 20], name=name, pose_labels=[torch.arange(63).unsqueeze(0),torch.arange(63).unsqueeze(0)])

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])
    train_loader = DataLoader(train_set, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)

    trainer = pl.Trainer(max_epochs=5)
    print("TEST RESULTS BEFORE", "-"*50)
    trainer.test(model, test_loader)
    trainer.fit(model, train_loader, val_loader)
    print("TEST RESULTS AFTER", "-"*50)
    trainer.test(model, test_loader)

    trainer = pl.Trainer(max_epochs=5)
    print("TEST RESULTS BEFORE", "-"*50)
    trainer.test(model2, test_loader)
    trainer.fit(model2, train_loader, val_loader)
    print("TEST RESULTS AFTER", "-"*50)
    trainer.test(model2, test_loader)


    #
    # o1 = model(sample)
    # o2 = model2(sample)
    #
    # f1 = model.save_checkpoint(best_val_loss=1)
    # f2 = model2.save_checkpoint(best_val_loss=2)
    #
    # model = M.load_checkpoint(f1)
    # model2 =M.load_checkpoint(f2)
    #
    # o11 = model(sample)
    # o12 = model2(sample)
    #
    # model.add_models(input_dims=[30])
    # sample = torch.randn(10, 60)
    # o1 = model(sample)
    #
    # model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)])
    # sample = torch.randn(10, 60)
    # o1 = model2(sample)
    #
    # model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)],freeze=True)
    # sample = torch.randn(10, 30)
    # o1 = model2(sample)


def test2():
    M = VAE
    name = "VAE"
    sample = torch.randn((10, 30))
    model = M(config=config, input_dims=[10, 20], name=name)
    model2 = M(config=config2, input_dims=[10, 20], name=name, pose_labels=[torch.arange(63).unsqueeze(0),torch.arange(63).unsqueeze(0)])

    o1 = model(sample)
    o2 = model2(sample)

    f1 = model.save_checkpoint(best_val_loss=1)
    f2 = model2.save_checkpoint(best_val_loss=2)

    model = M.load_checkpoint(f1)
    model2 =M.load_checkpoint(f2)

    o11 = model(sample)
    o12 = model2(sample)

    model.add_models(input_dims=[30])
    sample = torch.randn(10, 60)
    o1 = model(sample)

    model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)])
    sample = torch.randn(10, 60)
    o1 = model2(sample)

    model2.add_models(input_dims=[30], pose_labels=[torch.arange(63).unsqueeze(0)],freeze=True)
    sample = torch.randn(10, 30)
    o1 = model2(sample)


def test1():
    sample = torch.randn((10, 10))
    model = MLP(config=config, dimensions=[10], name="TEST", single_module=0)
    model2 = MLP(config=config, dimensions=[10], name="TEST", single_module=-1)
    model3 = MLP(config=config, dimensions=[10], name="TEST", single_module=1)
    model4 = MLP(config=config2, dimensions=[10], name="TEST", single_module=0)
    model4.configure_optimizers()
    model5 = MLP(config=config2, dimensions=[10], name="TEST", single_module=0, pose_labels=torch.arange(63).unsqueeze(0))

    o1 = model(sample)
    o2 = model2(sample)
    o4 = model4(sample)
    o5 = model5.decode_label(model5.encode(sample))

    f1 = model.save_checkpoint(best_val_loss=1)
    f2 = model2.save_checkpoint(best_val_loss=2)
    f3 = model3.save_checkpoint(best_val_loss=3)
    f4 = model4.save_checkpoint(best_val_loss=4)
    f5 = model5.save_checkpoint(best_val_loss=5)

    model = MLP.load_checkpoint(f1)
    model2 =MLP.load_checkpoint(f2)
    model3 =MLP.load_checkpoint(f3)
    model4 =MLP.load_checkpoint(f4)
    model5 =MLP.load_checkpoint(f5)

    o11 = model(sample)
    o12 = model2(sample)
    o14 = model4(sample)
    o15 = model5.decode_label(model5.encode(sample))

    assert (o1-o11).sum() < 1e-8, (o1-o11).sum()
    assert (o2-o12).sum() < 1e-8, (o2-o12).sum()
    assert (o4-o14).sum() < 1e-8, (o4-o14).sum()
    assert (o5-o15).sum() < 1e-8, (o5-o15).sum()

# test1()
# test2()
# test3()
test4()
