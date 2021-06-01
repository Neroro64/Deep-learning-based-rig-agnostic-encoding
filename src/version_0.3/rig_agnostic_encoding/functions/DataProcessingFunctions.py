import _pickle as pickle
import bz2
import os, sys
import numpy as np
import ray
from ray.exceptions import TaskCancelledError
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
import scipy.signal as signal
import time
sys.path.append("../../")
sys.path.append("../")
from GlobalSettings import MODEL_PATH

def save3(file:object, filename:str, path:str):
    """
    Writes and compresses an object to the disk
    :param file:
    :param filename:
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    with bz2.BZ2File(os.path.join(path, filename+".pbz2"), "w") as f:
        pickle.dump(file, f)


def save(file:object, filepath:str):
    """
    Writes and compresses an object to the disk
    :param file:
    :param filepath:
    :return:
    """
    with bz2.BZ2File(filepath, "w") as f:
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
                row.append(np.concatenate([jo[feature].ravel() for jo in f]).ravel())
            else:
                row.append(np.concatenate([jo[feature] for jo in f]))

            if feature not in feature_dims:
                feature_dims[feature] = row[-1].shape[0]

            assert np.isnan(row[-1]).sum() == 0, "{} contains NaN".format(feature)
            row = np.concatenate(row)
        clip.append(row)
    output = np.copy(np.vstack(clip))
    del data, frames, keyJ, clip, row
    return output, feature_dims

def load_features_local(data, feature_list, sampling_step=5, frame_window=15, use_window=False):
    data = pickle.loads(data)
    frames = data["frames"]
    clip = []
    keyJ = [i for i in range(len(frames[0])) if frames[0][i]["key"]]
    feature_dims = {}

    for f_id, f in enumerate(frames):
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
            elif feature == "posCostDistance" or \
                    feature == "rotCostAngle" or \
                    feature == "posCost" or \
                    feature == "rotCost":
                row.append(np.concatenate(
                    [frames[idx][jj][feature].ravel() for jj in keyJ for idx in f_idx]))

            # ALL JOINTS FEATURES
            elif feature == "rotMat":
                row.append(np.concatenate([jo["rotMat"].ravel() for jo in f]))
            elif feature == "isLeft" or feature == "chainPos" or feature == "geoDistanceNormalised":
                row.append(np.concatenate([jo[feature] for jo in f]))
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
    data = ray.get(data)
    if shutdown:
        ray.shutdown()
    return data

def normaliseT(x:torch.Tensor, dim:int):
    std = torch.std(x, dim=dim)
    std[std==0] = 1
    return (x - torch.mean(x, dim=dim)) / std

def normaliseN(x:np.ndarray, dim:int):
    std = np.std(x, dim=dim)
    std[std==0] = 1
    return (x-np.mean(x, dim=dim)) / std

def prepare_data(datasets:list, featureList:list,
                 train_ratio:float=0.8, val_ratio:float=0.2, test_size:int=100, SEED:int=2021):

    # process data
    data = [processData(d, featureList, shutdown=False) for d in datasets]
    # ray.shutdown()
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
    # ray.shutdown()
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


def clean_checkpoints(path=MODEL_PATH, num_keep=3):
    for dir, dname, files in os.walk(path):
        saved_checkpoints = []
        for fname in files:
            fname = fname.split(".")
            saved_checkpoints.append(fname)

        saved_checkpoints.sort(key=lambda x: x[0] + x[1])
        for filename in saved_checkpoints[num_keep:]:
            os.remove(os.path.join(dir, ".".join(filename)))


def save_testData(test_sets, path=MODEL_PATH):
    obj = {"data": test_sets}
    with bz2.BZ2File(path + "test_data.pbz2", "w") as f:
        pickle.dump(obj, f)


def normalise_block_function(block_fn:np.ndarray, window_size_half:int=30) -> np.ndarray:
    """
    Takes an ndarray of shape(n_joints, n_frames), where 1 = contact and 0 otherwise.
    :param block_fn:
    :param n_windows:
    :return:
    """
    Frames = block_fn.shape[1]
    t = np.arange(Frames)
    normalised_block_fn = np.zeros_like(block_fn, dtype=np.float32)

    for ts in t:
        low = max(ts-window_size_half, 0)
        high = min(ts+window_size_half, Frames)
        window = np.arange(low, high)
        slice = block_fn[:, window]
        mean = np.mean(slice, axis=1)
        std = np.std(slice, axis=1)
        std[std == 0] = 1
        normalised_block_fn[:, ts] = (block_fn[:, ts]-mean) / std

    filter = signal.butter(3, .1, "low", analog=False, output="sos")
    normalised_block_fn = signal.sosfilt(filter, normalised_block_fn)
    return normalised_block_fn


def extract_contact_velocity_info(data):
    contact_info = []
    velocity_info = []
    for f in data["frames"]:
        contacts = np.asarray([jo["contact"] for jo in f]).astype(np.float32)
        velocity = np.concatenate([jo["velocity"] for jo in f])

        contact_info.append(contacts)
        velocity_info.append(velocity)
    contact_info = np.vstack(contact_info)
    velocity_info = np.vstack(velocity_info)

    return contact_info.T, velocity_info.T

def calc_tta(contacts):
    """
    Calculates time-to-arriaval-embeddings, according to the paper [Robust Motion In-betweening]
    :param contacts:
    :param basis
    :return:
    """

    tta = np.zeros_like(contacts)
    for j, jo in enumerate(contacts):
        idx = np.arange(len(jo))
        idx[jo!=1] = 0
        diff = np.diff(idx, prepend=0)
        diff[diff < 0] = 0
        idx = np.where(diff > 0)[0]
        idx = [0] + list(idx)
        for i,k in zip(idx[:-1], idx[1:]):
            k2 = k-i
            tta[j][i:k] = np.arange(k2,0,-1)

    return tta


def calc_tta_embedding(embedd_dim:int, tta, basis: float = 1e5, window=30):
    """
    Calculates time-to-arriaval-embeddings, according to the paper [Robust Motion In-betweening]
    :param contacts:
    :param basis
    :return:
    """

    if embedd_dim % 2 != 0:
        d1 = embedd_dim / 2 + 1
        d2 = embedd_dim / 2
    else:
        d1 = d2 = embedd_dim / 2


    tta[tta > window] = 0
    d1 = np.arange(0, embedd_dim, 2)
    d2 = np.arange(1, embedd_dim, 2)

    z_sin = np.sin(tta / (np.power(basis, 2.0*d1 / embedd_dim)))
    z_cos = np.cos(tta / (np.power(basis, 2.0*d2 / embedd_dim)))

    z = np.zeros(embedd_dim)
    z[0::2] += z_sin
    z[1::2] += z_cos
    return z


@ray.remote(memory=300 * 1024 * 1024)
def remote_calc_motion_phases(datafile, id):
    new_data = []
    datafile = load(datafile)
    for d in datafile:
        d = pickle.loads(d)
        contacts, velocities = extract_contact_velocity_info(d)
        normalised_block_fn = normalise_block_function(contacts)
        velocities = np.reshape(velocities, (-1, 3, contacts.shape[1]))
        velocities = np.sqrt(np.sum(velocities ** 2, axis=1))
        sin_block_fn = np.sin(normalised_block_fn)
        cos_block_fn = np.cos(normalised_block_fn)
        sin_vec = sin_block_fn * velocities
        cos_vec = cos_block_fn * velocities

        ttas = calc_tta(contacts)

        for i, f in enumerate(d["frames"]):
            for j, jo in enumerate(f):
                jo["sin_normalised_contact"] = sin_block_fn[j,i]
                jo["cos_normalised_contact"] = cos_block_fn[j,i]
                jo["phase_vec"] = np.asarray([sin_vec[j,i], cos_vec[j,i]])
                jo["tta"] = ttas[j,i]

        new_data.append(pickle.dumps(d))
    return new_data, id

def calc_motion_phases(datafile):
    new_data = []
    for d in datafile:
        d = pickle.loads(d)
        contacts, velocities = extract_contact_velocity_info(d)
        normalised_block_fn = normalise_block_function(contacts)
        velocities = np.reshape(velocities, (-1, 3, contacts.shape[1]))
        velocities = np.sqrt(np.sum(velocities ** 2, axis=1))
        sin_block_fn = np.sin(normalised_block_fn)
        cos_block_fn = np.cos(normalised_block_fn)
        sin_vec = sin_block_fn * velocities
        cos_vec = cos_block_fn * velocities

        ttas = calc_tta(contacts)

        for i, f in enumerate(d["frames"]):
            for j, jo in enumerate(f):
                jo["sin_normalised_contact"] = sin_block_fn[j, i]
                jo["cos_normalised_contact"] = cos_block_fn[j, i]
                jo["phase_vec"] = np.asarray([sin_vec[j, i], cos_vec[j, i]])
                jo["tta"] = ttas[j, i]

        new_data.append(pickle.dumps(d))
    return new_data

def compute_local_motion_phase(dir_path=""):
    for dir, dname, files in os.walk(dir_path):
        for fname in files:
            file_path = os.path.join(dir, fname)
            data = calc_motion_phases(load(file_path))
            save(data, file_path)


def remote_compute_local_motion_phase(dir_path="", cpu=6):
    file_paths = []
    tasks = []
    start = time.time()
    ray.init(ignore_reinit_error=True, num_cpus=cpu)
    try:
        for dir, dname, files in os.walk(dir_path):
            for fname in files:
                file_path = os.path.join(dir, fname)
                file_paths.append(file_path)
                tasks.append(remote_calc_motion_phases.remote(file_path, len(file_paths)-1))
        while len(tasks):
            done_id, tasks = ray.wait(tasks)
            f, i = ray.get(done_id[0])
            save(f, file_paths[i])
            del f, done_id, i
    except TaskCancelledError:
        print("Failed")

    ray.shutdown()
    print("Done: {}".format(time.time() - start))
