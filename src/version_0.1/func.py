"""

miscellaneous functions

"""
import _pickle as pickle
import bz2
import os
import numpy as np
import ray
import torch

def save(file:object, filename:str, path:str):
    """
    Writes and compresses an object to the disk
    :param file:
    :param filename:
    :param path:
    :return:
    """
    with bz2.BZ2File(os.path.join(path, filename+".pbz2"), "w") as f:
        pickle.dump(file, f)


def load(file_path:str):
    """
    Loads a bzip2-compressed binary file into memory
    :param file_path:
    :return:
    """
    with bz2.BZ2File(file_path, "rb") as f:
        obj = pickle.load(f)
        return obj


def calc_tta(input_dim:int, tta:int, basis:float=1e5):
    """
    Calculates time-to-arriaval-embeddings, according to the paper [Robust Motion In-betweening]
    :param input_dim:
    :param tta:
    :param basis
    :return:
    """
    if input_dim % 2 != 0:
        d1 = input_dim / 2 + 1
        d2 = input_dim / 2
    else:
        d1 = d2 = input_dim / 2

    sin = np.sin(tta / (np.power(basis, (2.0 * np.arange(d1) / d1))))
    cos = np.cos(tta / (np.power(basis, (2.0 * np.arange(d2) / d2))))
    z = np.zeros(input_dim)
    z[0::2] += sin
    z[1::2] += cos
    return z


@ray.remote
def loadFeatures(data, feature_list):
    data = pickle.loads(data)
    features = []
    for f in data["frames"]:
        p = []
        for feature in feature_list:
            if feature == "rotMat":
                p.append(np.concatenate([jo["rotMat"].ravel() for jo in f]))
            elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                p.append(np.concatenate([[jo[feature]] for jo in f]))
            else:
                p.append(np.concatenate([jo[feature] for jo in f]))

        p = np.concatenate(p)
        features.append(p)
    return np.vstack(features)


def processData(compressed_data, feature_list, num_cpus=24, shutdown=True):
    ray.init(num_cpus=num_cpus,ignore_reinit_error=True)
    data = [loadFeatures.remote(d, feature_list) for d in compressed_data]
    data = [ray.get(d) for d in data]
    if shutdown:
        ray.shutdown()
    return data


def normaliseT(x:torch.Tensor):
    std = torch.std(x)
    std[std==0] = 1
    return (x - torch.mean(x)) / std


def normaliseN(x:np.ndarray):
    std = np.std(x)
    std[std==0] = 1
    return (x-np.mean(x)) / std
