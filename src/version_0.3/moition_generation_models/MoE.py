import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os, sys
import bz2
import _pickle as pickle

sys.path.append("../")
from GlobalSettings import MODEL_PATH

class MoE(nn.Module):
    def __init__(self, config=None, dimensions=None,
                 phase_input_dim:int=0, name="MoE", device="cuda"):
        super(MoE, self).__init__()

        self.phase_input_dim = phase_input_dim
        self.dimensions = dimensions
        self.act_fn = nn.ELU

        self.name = name
        self.config = config
        self.device = device

        self.k_experts = config["k_experts"]
        self.gate_size = config["gate_size"]
        self.keep_prob = config["keep_prob"]
        self.hidden_dim = config["g_hidden_dim"]

        self.dimensions = dimensions if len(dimensions) > 2 else \
            [dimensions[0], self.hidden_dim, self.hidden_dim, dimensions[-1]]

        self.layers = []

        self.build()
        self.gate = nn.Sequential(
            nn.Linear(phase_input_dim, self.gate_size),
            self.act_fn(),
            nn.Linear(self.gate_size, self.gate_size),
            self.act_fn(),
            nn.Linear(self.gate_size, self.k_experts)
        )
        self.init_params()

    def forward(self, x:torch.Tensor, phase) -> torch.Tensor:
        coefficients = F.softmax(self.gate(phase), dim=1)
        layer_out = x
        for (weight, bias, activation) in self.layers:
            if weight is None:
                layer_out = activation(layer_out, p=self.keep_prob)
            else:
                flat_weight = weight.flatten(start_dim=1, end_dim=2)
                mixed_weight = torch.matmul(coefficients, flat_weight).view(
                    coefficients.shape[0], *weight.shape[1:3]
                )
                input = layer_out.unsqueeze(1)
                mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
                out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
                layer_out = activation(out) if activation is not None else out
        return layer_out

    def build(self):
        layers = []
        for i, size in enumerate(zip(self.dimensions[0:], self.dimensions[1:])):
            if i < len(self.dimensions) - 2:
                layers.append(
                    (
                        nn.Parameter(torch.empty(self.k_experts, size[0], size[1])),
                        nn.Parameter(torch.empty(self.k_experts, size[1])),
                        self.act_fn()
                    )
                )
            else:
                layers.append(
                    (
                        nn.Parameter(torch.empty(self.k_experts, size[0], size[1])),
                        nn.Parameter(torch.empty(self.k_experts, size[1])),
                        None
                    )
                )
        self.layers = layers

    def reset_hidden(self, batch_size):
        pass

    def init_params(self):
        for i, (w, b, _) in enumerate(self.layers):
            if w is None:
                continue

            i = str(i)
            torch.nn.init.kaiming_uniform_(w)
            b.data.fill_(0.01)
            self.register_parameter("w" + i, w)
            self.register_parameter("b" + i, b)

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):
        config = dict(
                k_experts=self.k_experts,
                gate_size=self.gate_size,
                keep_prob=self.keep_prob,
                g_hidden_dim=self.hidden_dim
                      )

        model = {
            "config":config,
            "dimensions":self.dimensions,
            "name":self.name,
            "phase_input_dim":self.phase_input_dim,
            "generationNetwork":self.state_dict(),
            "gate":self.gate.state_dict(),
            "device":self.device
             }

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        filePath = os.path.join(path, str(best_val_loss)+".pbz2")
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(model, f)
        return filePath


    @staticmethod
    def load_checkpoint(filePath):
        with bz2.BZ2File(filePath, "rb") as f:
            obj = pickle.load(f)

        model = MoE(config=obj["config"], name=obj["name"], dimensions=obj["dimensions"], device=obj["device"],
                    phase_input_dim=obj["phase_input_dim"])

        model.load_state_dict(obj["generationNetwork"])
        model.gate.load_state_dict(obj["gate"])
        return model

    def freeze(self, flag=False):
        self.gate.requires_grad_(flag)
        for (weight, bias, _) in self.layers:
            if weight == None: continue
            weight.requires_grad_(flag)
            bias.requires_grad_(flag)


