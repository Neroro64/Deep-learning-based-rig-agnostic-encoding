from MoE import MoE
from MoE_Z import MoE as MoE_Z
from GRU import GRU
from GRU_Z import GRU as GRU_Z
from LSTM import LSTM
from LSTM_Z import LSTM as LSTM_Z

from MotionGeneration import MotionGenerationModel as MoGen
from MotionGenerationEmbedd import MotionGenerationModel as MoGenZ
from MotionGenerationVAE import MotionGenerationModel as MoGenVAE
from MotionGenerationVAE_Embedd import MotionGenerationModel as MoGenVAE_Z

import os, sys
sys.path.append("../rig_agnostic_encoding/models/")
sys.path.append("../rig_agnostic_encoding/functions/")
from MLP import MLP
from MLP_MIX import MLP_MIX
from VAE import VAE
from RBF import RBF
from MLP_Adversarial import MLP_ADV

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


config = {
    "hidden_dim": 256,
    "k": 64,
    "z_dim": 32,
    "lr": 1e-3,
    "batch_size": 10,
    "keep_prob": 0,
    "loss_fn":torch.nn.functional.mse_loss,
    "optimizer":torch.optim.AdamW,
    "scheduler":torch.optim.lr_scheduler.StepLR,
    "scheduler_param": {"step_size":80, "gamma":.9},
    "basis_func":"gaussian",
    "n_centroid":64,
    "k_experts": 4,
    "gate_size": 12,
    "g_hidden_dim": 32,
    "num_layers": 4,
    "autoregress_prob":0.3,
    "autoregress_inc":20,
    "autoregress_ep":20,
    "autoregress_max_prob":1,
    "cost_hidden_dim":16,
    "seq_len":10,
    "device":"cuda"
    }

def test3():
    Enc = [MLP, MLP_ADV]
    # Enc = [MLP_ADV]
    MZ = [MoE_Z, GRU_Z, LSTM_Z]
    Middle = [MLP_MIX, RBF, VAE]
    # Middle = [VAE]

    X = torch.randn((100, 100, 60))
    Y = X

    feature_dims = {
        "phase_dim":10,
        "pose_dim":40,
        "cost_dim":5,
        "target_dim":5,
        "g_input_dim": config["k"] + config["cost_hidden_dim"],
        "g_output_dim":10 + config["k"] + 5,
        "targetPosition":2,
        "targetRotation":3,
        "pos_dim":10,
        "rot_dim":20,
        "vel_dim":10,
        "posCost":2,
        "rotCost":3,
    }

    in_slice = [10, 40, 5, 5]
    out_slice = [10, config["k"], 5]

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])
    train_loader = DataLoader(train_set, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)

    for i, enc in enumerate(Enc):
        for middle in Middle:
            for j, M in enumerate(MZ):
                if i == 0:
                    pose_encoder = enc(config=config, dimensions=[40],
                                       pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"], vel_dim=feature_dims["vel_dim"])
                    pose_encoder2 = enc(config=config, dimensions=[40], pose_labels=torch.arange(63).unsqueeze(0).to(config["device"]),
                                        pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"],
                                        vel_dim=feature_dims["vel_dim"])
                    use_adv = False
                else:
                    pose_encoder = enc(config=config, dimensions=[40], h_dim=100, w_dim=40,
                                        pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"],
                                        vel_dim=feature_dims["vel_dim"])
                    pose_encoder2 = enc(config=config, dimensions=[40], pose_labels=torch.arange(63).unsqueeze(0).to(config["device"]), h_dim=100, w_dim=40,
                                    pos_dim = feature_dims["pos_dim"], rot_dim = feature_dims["rot_dim"],
                                    vel_dim = feature_dims["vel_dim"])
                    use_adv = True

                middle_layer = middle(config=config, input_dims=[15])
                if middle != VAE:
                    m1 = MoGenZ(config=config, Model=M, pose_autoencoder=pose_encoder, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice, use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                    m2 = MoGenZ(config=config, Model=M, pose_autoencoder=pose_encoder2, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                else:
                    m1 = MoGenVAE_Z(config=config, Model=M, pose_autoencoder=pose_encoder, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                    m2 = MoGenVAE_Z(config=config, Model=M, pose_autoencoder=pose_encoder2, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )

                print("-"*50, enc, " + ", middle, " + ", M, "-"*50)
                trainer = pl.Trainer(max_epochs=5, gpus=1)
                print("-TEST RESULTS BEFORE", "-"*50)
                res1 = trainer.test(m1)
                trainer.fit(m1)
                print("-TEST RESULTS AFTER", "-"*50)
                res2 = trainer.test(m1)
                print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

                print("-"*50, enc, " + ", middle, " + ", M, "-"*50)
                trainer = pl.Trainer(max_epochs=5, gpus=1)
                print("-TEST RESULTS BEFORE", "-"*50)
                res1 = trainer.test(m2)
                trainer.fit(m2)
                print("-TEST RESULTS AFTER", "-"*50)
                res2 = trainer.test(m2)
                print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

                p1 = m1.save_checkpoint(best_val_loss=0.001)
                p2 = m2.save_checkpoint(best_val_loss=0.001)
                if middle != VAE:
                    m1 = MoGen.load_checkpoint(filename=p1, Model=M, MiddleModel=middle_layer.cluster_model)
                    m2 = MoGen.load_checkpoint(filename=p2, Model=M, MiddleModel=middle_layer.cluster_model)
                else:
                    m1 = MoGenVAE.load_checkpoint(p1, Model=M, MiddleModel=middle_layer.cluster_model)
                    m2 = MoGenVAE.load_checkpoint(p2, Model=M, MiddleModel=middle_layer.cluster_model)



def test2():
    Enc = [MLP, MLP_ADV]
    # Enc = [MLP_ADV]
    Ms = [MoE, GRU, LSTM]
    # MZ = [MoE_Z, GRU_Z, LSTM_Z]
    Middle = [MLP_MIX, RBF, VAE]
    # Middle = [VAE]

    X = torch.randn((100, 100, 60))
    Y = X

    feature_dims = {
        "phase_dim":10,
        "pose_dim":40,
        "cost_dim":5,
        "target_dim":5,
        "g_input_dim": config["z_dim"] + config["cost_hidden_dim"],
        "g_output_dim":10 + config["k"] + 5,
        "targetPosition":2,
        "targetRotation":3,
        "pos_dim":10,
        "rot_dim":20,
        "vel_dim":10,
        "posCost":2,
        "rotCost":3,
    }

    in_slice = [10, 40, 5, 5]
    out_slice = [10, config["k"], 5]

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(dataset, [70, 15, 15])
    train_loader = DataLoader(train_set, batch_size=10)
    val_loader = DataLoader(val_set, batch_size=10)
    test_loader = DataLoader(test_set, batch_size=10)

    for i, enc in enumerate(Enc):
        for middle in Middle:
            for j, M in enumerate(Ms):
                if i == 0:
                    pose_encoder = enc(config=config, dimensions=[40],
                                       pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"], vel_dim=feature_dims["vel_dim"])
                    pose_encoder2 = enc(config=config, dimensions=[40], pose_labels=torch.arange(63).unsqueeze(0).to(config["device"]),
                                        pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"],
                                        vel_dim=feature_dims["vel_dim"])
                    use_adv = False
                else:
                    pose_encoder = enc(config=config, dimensions=[40], h_dim=100, w_dim=40,
                                        pos_dim=feature_dims["pos_dim"], rot_dim=feature_dims["rot_dim"],
                                        vel_dim=feature_dims["vel_dim"])
                    pose_encoder2 = enc(config=config, dimensions=[40], pose_labels=torch.arange(63).unsqueeze(0).to(config["device"]), h_dim=100, w_dim=40,
                                    pos_dim = feature_dims["pos_dim"], rot_dim = feature_dims["rot_dim"],
                                    vel_dim = feature_dims["vel_dim"])
                    use_adv = True

                middle_layer = middle(config=config, input_dims=[15])
                if middle != VAE:
                    m1 = MoGen(config=config, Model=M, pose_autoencoder=pose_encoder, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice, use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                    m2 = MoGen(config=config, Model=M, pose_autoencoder=pose_encoder2, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                else:
                    m1 = MoGenVAE(config=config, Model=M, pose_autoencoder=pose_encoder, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )
                    m2 = MoGenVAE(config=config, Model=M, pose_autoencoder=pose_encoder2, feature_dims=feature_dims, middle_layer=middle_layer.cluster_model,
                                               input_slicers=in_slice, output_slicers=out_slice,use_advLoss=use_adv,
                                               train_set=train_set, val_set=val_set, test_set=test_set,
                                               )

                print("-"*50, enc, " + ", middle, " + ", M, "-"*50)
                trainer = pl.Trainer(max_epochs=5, gpus=1)
                print("-TEST RESULTS BEFORE", "-"*50)
                res1 = trainer.test(m1)
                trainer.fit(m1)
                print("-TEST RESULTS AFTER", "-"*50)
                res2 = trainer.test(m1)
                print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

                print("-"*50, enc, " + ", middle, " + ", M, "-"*50)
                trainer = pl.Trainer(max_epochs=5, gpus=1)
                print("-TEST RESULTS BEFORE", "-"*50)
                res1 = trainer.test(m2)
                trainer.fit(m2)
                print("-TEST RESULTS AFTER", "-"*50)
                res2 = trainer.test(m2)
                print("IMPROVEMENT: ", res1[0]["test_loss"]-res2[0]["test_loss"])

                p1 = m1.save_checkpoint(best_val_loss=0.001)
                p2 = m2.save_checkpoint(best_val_loss=0.001)
                if middle != VAE:
                    m1 = MoGen.load_checkpoint(filename=p1, Model=M, MiddleModel=middle_layer.cluster_model)
                    m2 = MoGen.load_checkpoint(filename=p2, Model=M, MiddleModel=middle_layer.cluster_model)
                else:
                    m1 = MoGenVAE.load_checkpoint(p1, Model=M, MiddleModel=middle_layer.cluster_model)
                    m2 = MoGenVAE.load_checkpoint(p2, Model=M, MiddleModel=middle_layer.cluster_model)


def test1():
    Models = [MoE, GRU, LSTM]
    ModelsZ = [MoE_Z, GRU_Z, LSTM_Z]
    sample = torch.randn((100,30))
    z = torch.randn((100, config["z_dim"]))
    phase = torch.randn((100, 6))

    for M in Models:
        model = M(config=config, dimensions=[30, 30], phase_input_dim=6, device="cpu")
        model.reset_hidden(batch_size=100)
        if M != MoE:
            out = model(sample, phase)
        else:
            out = model(sample, phase)

        f1 = model.save_checkpoint(best_val_loss=1)
        model = M.load_checkpoint(f1)
        model.reset_hidden(batch_size=100)
        loss = out - model(sample, phase)
        print("-"*35, M, "-"*35)
        print(torch.sum(loss))
        print("-"*70)

    for M in ModelsZ:
        model = M(config=config, dimensions=[30, 30], phase_input_dim=6, device="cpu")
        model.reset_hidden(batch_size=100)
        if M != MoE:
            out = model(sample, phase, z)
        else:
            out = model(sample, phase, z)

        f1 = model.save_checkpoint(best_val_loss=1)
        model = M.load_checkpoint(f1)
        model.reset_hidden(batch_size=100)
        loss = out - model(sample, phase, z)
        print("-"*35, M, "-"*35)
        print(loss.sum())
        print("-"*70)

test1()
test2()
test3()