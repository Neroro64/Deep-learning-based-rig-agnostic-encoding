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


class GRU(nn.Module):
    def __init__(self, config=None, dimensions=None, phase_input_dim:int=0, name="model", device="cuda"):
        super().__init__()

        self.dimensions = dimensions
        self.act_fn = nn.ELU
        self.name = name
        self.config = config
        self.device = device
        self.phase_input_dim = phase_input_dim

        self.keep_prob = config["keep_prob"]
        self.hidden_dim = config["g_hidden_dim"]
        self.num_layers = config["num_layers"]
        self.batch_size = config["batch_size"]

        self.dimensions = dimensions if len(dimensions) > 2 else \
            [dimensions[0]+phase_input_dim, self.hidden_dim, self.hidden_dim, dimensions[-1]]

        self.rnn = nn.GRU(input_size=self.dimensions[0], hidden_size=self.hidden_dim,
                           num_layers=self.num_layers, dropout=self.keep_prob, batch_first=True)
        self.decoder = nn.Linear(in_features=self.hidden_dim, out_features=dimensions[-1])

        self.reset_hidden(batch_size=self.batch_size)

    def forward(self, x:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        x = torch.cat((x,c), dim=1).unsqueeze(dim=1)
        h_t, h_n = self.rnn(x, self.hidden)
        self.hidden = h_n

        return self.decoder(h_t).squeeze(dim=1)

    def reset_hidden(self, batch_size=0):
        hidden_state = torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device))

        self.hidden = hidden_state

    def save_checkpoint(self, best_val_loss:float=np.inf, checkpoint_dir=MODEL_PATH):
        config = dict(
            num_layers=self.num_layers,
            keep_prob=self.keep_prob,
            g_hidden_dim=self.hidden_dim,
            batch_size=self.batch_size
        )

        model = {
            "config": config,
            "dimensions": self.dimensions,
            "name": self.name,
            "phase_input_dim": self.phase_input_dim,
            "rnn": self.rnn.state_dict(),
            "decoder": self.decoder.state_dict(),
            "device":self.device,
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

        model = GRU(config=obj["config"], name=obj["name"], dimensions=obj["dimensions"],device=obj["device"],
                    phase_input_dim=obj["phase_input_dim"])
        model.rnn.load_state_dict(obj["rnn"])
        model.decoder.load_state_dict(obj["decoder"])

        return model

    def freeze(self, flag=False):
        self.rnn.requires_grad_(flag)
        self.decoder.requires_grad_(flag)


