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
def load_features(data, feature_list, sampling_step=5, frame_window=15, use_window=False):
    data = pickle.loads(data)
    frames = data["frames"]
    clip = []
    keyJ = [i for i in range(len(frames)) if frames[0][i]["key"]]
    feature_dims = {}

    for f in frames:
        if use_window:
            f_idx = np.arange(f-frame_window, f+frame_window, sampling_step, dtype=int)
        else:
            f_idx = [f]

        row = []
        for feature in feature_list:
            # KEY JOINTS ONLY FEATURES
            if feature == "phase_vec":
                row.append(np.concatenate([frames[idx][jj]["phase"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "contact":
                row.append(np.asarray([frames[idx][jj]["contact"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "targetRotation" or feature == "targetPosition":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "tPos" :
                row.append(np.concatenate(
                    [frames[idx][jj]["pos"] for jj in keyJ for idx in f_idx]))
            elif feature == "tRot":
                row.append(np.concatenate(
                    [frames[idx][jj]["rotMat"].ravel() for jj in keyJ for idx in f_idx]))
            # ALL JOINTS FEATURES
            elif feature == "rotMat":
                row.append(np.concatenate([jo["rotMat"].ravel() for jo in f]))
            elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                row.append(np.concatenate([[jo[feature]] for jo in f]))
            else:
                row.append(np.concatenate([jo[feature] for jo in f]))


            if feature not in feature_dims:
                feature_dims[feature] = row[-1].shape[0]

            assert np.isnan(row[-1]).sum() == 0, "{} contains NaN".format(feature)

        row = np.concatenate(row)
        clip.append(row)
    return np.vstack(clip), feature_dims


def processData(compressed_data, feature_list, num_cpus=24, shutdown=True):
    ray.init(num_cpus=num_cpus,ignore_reinit_error=True)
    data = [load_features.remote(d, feature_list) for d in compressed_data]
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
