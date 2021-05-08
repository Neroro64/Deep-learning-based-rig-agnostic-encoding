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


def load_features(data, feature_list, sampling_step=5, frame_window=15, use_window=False,
                        level=0):
    frames = data["frames"]
    n_frames = len(frames)
    clip = []
    keyJ = [i for i in range(len(frames[0])) if frames[0][i]["key"]]
    joIDs = [i for i in range(len(frames[0])) if frames[0][i]["level"] >= level]

    feature_dims = {}
    for f_id, f in enumerate(frames):
        if use_window:
            f_idx = np.arange(f_id-frame_window, f_id+frame_window, sampling_step, dtype=int)
            f_idx[f_idx < 0] = 0
            f_idx[f_idx >= n_frames] = n_frames -1
        else:
            f_idx = [f_id]

        row = []
        for feature in feature_list:
            # KEY JOINTS ONLY FEATURES
            if "phase_vec" in feature:
                row.append(np.concatenate([frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "contact":
                row.append(np.concatenate([frames[idx][jj]["contact"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "targetRotation" or feature == "targetPosition":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "tPos" :
                row.append(np.concatenate(
                    [frames[idx][jj]["pos"] for jj in keyJ for idx in f_idx]))
            elif feature == "tRot":
                row.append(np.concatenate(
                    [frames[idx][jj]["rotMat"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "posCostDistance" or \
                    feature == "rotCostAngle" or \
                    feature == "posCost" or \
                    feature == "rotCost":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))

            # ALL JOINTS FEATURES
            elif feature == "rotMat" or feature == "rotMat2":
                row.append(np.concatenate([f[jj][feature].ravel() for jj in joIDs]))
            elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                row.append(np.concatenate([[f[jj][feature]] for jj in joIDs]))
            else:
                row.append(np.concatenate([f[jj][feature].ravel() for jj in joIDs]))

            if feature not in feature_dims:
                feature_dims[feature] = row[-1].shape[0]

            assert np.isnan(row[-1]).sum() == 0, "{} contains NaN".format(feature)

        row = np.concatenate(row)
        clip.append(row)
    return np.vstack(clip), feature_dims

@ray.remote
def load_features_task(data, feature_list, sampling_step=5, frame_window=15, use_window=False,
                        level=0):
    data = pickle.loads(load(data))

    frames = data["frames"]
    n_frames = len(frames)
    clip = []
    keyJ = [i for i in range(len(frames[0])) if frames[0][i]["key"]]
    joIDs = [i for i in range(len(frames[0])) if frames[0][i]["level"] >= level]

    feature_dims = {}
    for f_id, f in enumerate(frames):
        if use_window:
            f_idx = np.arange(f_id-frame_window, f_id+frame_window, sampling_step, dtype=int)
            f_idx[f_idx < 0] = 0
            f_idx[f_idx >= n_frames] = n_frames -1
        else:
            f_idx = [f_id]

        row = []
        for feature in feature_list:
            # KEY JOINTS ONLY FEATURES
            if "phase_vec" in feature:
                row.append(np.concatenate([frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "tta":
                row.append(np.concatenate([frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "contact":
                row.append(np.concatenate([frames[idx][jj]["contact"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "targetRotation" or feature == "targetPosition":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "tPos" :
                row.append(np.concatenate(
                    [frames[idx][jj]["pos"] for jj in keyJ for idx in f_idx]))
            elif feature == "tRot":
                row.append(np.concatenate(
                    [frames[idx][jj]["rotMat"].ravel() for jj in keyJ for idx in f_idx]))
            elif feature == "posCostDistance" or \
                    feature == "rotCostAngle" or \
                    feature == "posCost" or \
                    feature == "rotCost":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))

            # ALL JOINTS FEATURES
            elif feature == "rotMat" or feature == "rotMat2":
                row.append(np.concatenate([f[jj][feature].ravel() for jj in joIDs]))
            elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                row.append(np.concatenate([[f[jj][feature]] for jj in joIDs]))
            else:
                row.append(np.concatenate([f[jj][feature].ravel() for jj in joIDs]))

            if f_id == 0 and feature not in feature_dims:
                feature_dims[feature] = row[-1].shape[0]

            if feature == "rotCost":
                row[-1] = np.nan_to_num(row[-1], nan=1)
            assert np.isnan(row[-1]).sum() == 0, "{} contains NaN".format(feature)

        row = np.concatenate(row)
        clip.append(row)

    output = np.vstack(clip)
    del data, clip, feature_list, keyJ, joIDs, f_idx, row
    return output, feature_dims


def process_data(file_paths, feature_list):
    clips = []
    feature_dims = {}
    for file in file_paths:
        data = pickle.loads(load(file))
        clip, feature_dims = load_features(data=data, feature_list=feature_list)
        clips.append(clip)
    return clips, feature_dims


def process_data_multithread(file_paths, feature_list,  level=0, sampling_step=5, frame_window=15, use_window=False, num_cpus=12, shutdown=True,):
    ray.init(num_cpus=num_cpus,ignore_reinit_error=True)
    data = [load_features_task.remote(path, feature_list, sampling_step, frame_window, use_window, level) for path in file_paths]
    data = ray.get(data)
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
