import _pickle as pickle
import bz2
import os
import numpy as np
import ray
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split

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
    if not ray.is_initialized():
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


def prepare_data(datasets:list, featureList:list,
                 train_ratio:float=0.8, val_ratio:float=0.2, test_size:int=100, SEED:int=2021):

   # process data
    data = [processData(d, featureList) for d in datasets]
    input_data = [np.vstack(d) for d in data]
    x_tensors = [normaliseT(torch.from_numpy(x).float()) for x in input_data]
    y_tensors = [torch.from_numpy(x).float() for x in input_data]

    # prepare datasets
    test_sets = [(x_tensor[-test_size:], y_tensor[-test_size:]) for x_tensor, y_tensor in zip(x_tensors, y_tensors)]
    x_training = torch.vstack([x_tensor[:-test_size] for x_tensor in x_tensors])
    y_training = torch.vstack([y_tensor[:-test_size] for y_tensor in y_tensors])
    dataset = TensorDataset(x_training, y_training)
    N = len(x_training)

    train_ratio = int(train_ratio*N)
    val_ratio = int(val_ratio*N)
    train_set, val_set = random_split(dataset, [train_ratio, val_ratio], generator=torch.Generator().manual_seed(SEED))
    return train_set, val_set, test_sets


def prepare_data_withLabel(datasets:list, featureList:list, extra_feature_len:int=0,
                 train_ratio:float=0.8, val_ratio:float=0.2, test_size:int=100, SEED:int=2021):
   # process data
    data = [processData(d, featureList, shutdown=False) for d in datasets]
    input_data = [np.vstack(d) for d in data]
    x_tensors = [normaliseT(torch.from_numpy(x).float()) for x in input_data]
    y_tensors = [torch.from_numpy(x[:, :-extra_feature_len]).float() for x in input_data]

    # prepare datasets
    test_sets = [(x_tensor[-test_size:], y_tensor[-test_size:]) for x_tensor, y_tensor in zip(x_tensors, y_tensors)]
    x_training = torch.vstack([x_tensor[:-test_size] for x_tensor in x_tensors])
    y_training = torch.vstack([y_tensor[:-test_size] for y_tensor in y_tensors])
    dataset = TensorDataset(x_training, y_training)
    N = len(x_training)

    train_ratio = int(train_ratio*N)
    val_ratio = int(val_ratio*N)
    train_set, val_set = random_split(dataset, [train_ratio, val_ratio], generator=torch.Generator().manual_seed(SEED))
    return train_set, val_set, test_sets


def clean_checkpoints(num_keep=3, path="../../models"):
    for dir, dname, files in os.walk(path):
        saved_checkpoints = []
        for fname in files:
            fname = fname.split(".")
            saved_checkpoints.append(fname)

        saved_checkpoints.sort(key=lambda x: x[0] + x[1])
        for filename in saved_checkpoints[num_keep:]:
            os.remove(os.path.join(dir, ".".join(filename)))


def save_testData(test_sets, path="../../models"):
    obj = {"data": test_sets}
    with bz2.BZ2File(path + "test_data.pbz2", "w") as f:
        pickle.dump(obj, f)
