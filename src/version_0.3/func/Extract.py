import os
import numpy as np
import json as js
import _pickle as pickle
import bz2
import ray
import time
import sys
import func
import scipy.signal as signal
import rig_agnostic_encoding.functions.DataProcessingFunctions as Data
"""
Global variables
"""
PATH = "C:/Users/Neroro/AppData/LocalLow/DefaultCompany/Procedural Animation"
THREADING = True
CPUs = 8


def parseFloat3(struct:dict) -> np.ndarray:
    """
    Reconstructs a Unity Vector3 as a numpy array of shape (3,)
    :param struct:
    :return: np.ndarray (3,)
    """
    return np.asarray([struct["x"], struct["y"], struct["z"]], dtype=np.float32)

def parseFloat4(struct:dict) -> np.ndarray:
    """
    Reconstructs a Unity Vector4 as a numpy array of shape (4,)
    :param struct:
    :return: np.ndarray (4,)
    """
    return np.asarray([struct["value"]["x"], struct["value"]["y"], struct["value"]["z"], struct["value"]["w"]], dtype=np.float32)

def parseFloat3x3(struct:dict) -> np.ndarray:
    """
    Reconstructs a Unity Matrix3x3 as a numpy array of shape (3,3)
    :param struct:
    :return: np.ndarray (3,3)
    """
    return np.column_stack(
        (parseFloat3(struct["c0"]), parseFloat3(struct["c1"]), parseFloat3(struct["c2"])))

def parseFloat2x3(struct:dict) -> np.ndarray:
    """
    Reconstructs a Unity Matrix3x3 as a numpy array of shape (3,3)
    :param struct:
    :return: np.ndarray (3,3)
    """
    return np.column_stack(
        (parseFloat3(struct["c0"]), parseFloat3(struct["c1"])))


def ParseRawSequence(clip:dict) -> dict:
    """
    Parses and converts an AnimationSequence object to Python dictionary
    :param clip: AnimationSequence
    :return df: dict(targetPos:list, targetRot:list, frames:list(joints: list(features))
    """

    # targetPos = clip["frames"][0]["targetsPositions"]
    # targetRot = clip["frames"][0]["targetsRotations"]

    df = {}
    # df["targetPos"] = [parseFloat3(t) for t in targetPos]
    # df["targetRot"] = [parseFloat4(t) for t in targetRot]

    frames = []
    for frame in clip["frames"]:
        joints = []
        for jojo in frame["joints"]:
            jo = {}

            jo["key"] = np.asarray([jojo["key"]], dtype=np.float32)
            jo["isLeft"] = np.asarray([jojo["isLeft"]], dtype=np.float32)
            jo["chainPos"] = np.asarray([jojo["chainPos"]], dtype=np.float32)
            jo["geoDistance"] = np.asarray([jojo["geoDistance"]], dtype=np.float32)
            jo["geoDistanceNormalised"] = np.asarray([jojo["geoDistanceNormalised"]], dtype=np.float32)
            jo["level"] = np.asarray([jojo["level"]], dtype=np.float32)

            jo["pos"] = parseFloat3(jojo["position"])
            jo["rotEuler"] = parseFloat3(jojo["rotEuler"])
            jo["rotQuaternion"] = parseFloat4(jojo["rotQuaternion"])
            jo["rotMat"] = parseFloat3x3(jojo["rotMat"])
            jo["rotMat2"] = parseFloat2x3(jojo["rotMat"])

            jo["inertialObj"] = parseFloat3x3(jojo["inertiaObj"])
            jo["inertial"] = parseFloat3x3(jojo["inertia"])

            jo["velocity"] = parseFloat3(jojo["velocity"])
            jo["velocityMagnitude"] = np.asarray([jojo["velocityMagnitude"]], dtype=np.float32)
            jo["angularVelocity"] = parseFloat3(jojo["angularVelocity"])
            jo["linearMomentum"] = parseFloat3(jojo["linearMomentum"])
            jo["angularMomentum"] = parseFloat3(jojo["angularMomentum"])
            jo["totalMass"] = jojo["totalMass"]

            jo["targetValue"] = parseFloat3({"x": jojo["x"]["TargetValue"], "y":jojo["y"]["TargetValue"], "z":jojo["z"]["TargetValue"]})
            jo["currentValue"] = parseFloat3({"x": jojo["x"]["CurrentValue"], "y":jojo["y"]["CurrentValue"], "z":jojo["z"]["CurrentValue"]})

            jo["tGoalPos"] = parseFloat3(jojo["tGoalPosition"])
            jo["tGoalDir"] = parseFloat3(jojo["tGoalDirection"])

            jo["posCost"] = parseFloat3(jojo["cost"]["ToTarget"])
            jo["rotCost"] = parseFloat3(jojo["cost"]["ToTargetRotation"])
            jo["targetPosition"] = parseFloat3(jojo["cost"]["TargetPosition"])
            jo["targetRotation"] = parseFloat3x3(jojo["cost"]["TargetRotation"])

            jo["posCostDistance"] = np.asarray([jojo["cost"]["ToTargetDistance"]],dtype=np.float32)
            jo["rotCostAngle"] = np.asarray([jojo["cost"]["ToTargetRotationAngle"]], dtype=np.float32)

            jo["contact"] = np.asarray([jojo["contact"]], dtype=np.float32)

            joints.append(jo)
        frames.append(joints)
    df["frames"] = frames
    return df

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

def calc_phase_vec_tta(clip, window_size_half=15):
    frames = clip["frames"]

    block_fn = []
    velocity = []
    for f in frames:
        block = np.concatenate([jo["contact"] for jo in f])
        v = np.concatenate([jo["velocity"] for jo in f])

        block_fn.append(block)
        velocity.append(v)

    block_fn = np.vstack(block_fn).T
    velocity = np.vstack(velocity).T

    velocity = np.sqrt(np.sum(
        (velocity.reshape((3,-1, velocity.shape[-1]))**2),axis=0))

    ttas = calc_tta(block_fn)

    Frames = block_fn.shape[1]
    t = np.arange(Frames)
    normalised_block_fn = np.zeros_like(block_fn, dtype=np.float32)

    for ts in t:
        low = max(ts-window_size_half, 0)
        high = min(ts+window_size_half, Frames)
        window = np.arange(low, high)
        if len(window) < window_size_half*2:
            window = np.pad(window, (window_size_half*2-len(window),), mode='edge')
        slice = block_fn[:, window]
        mean = np.mean(slice, axis=1)
        std = np.std(slice, axis=1)
        std[std == 0] = 1
        normalised_block_fn[:, ts] = (block_fn[:, ts]-mean) / std

    # normalised_block_fn = (block_fn2 - np.mean(block_fn2, axis=1, keepdims=True)) / np.std(block_fn2, axis=1, keepdims=True)
    filter = signal.butter(3, .1, "low", analog=False, output="sos")
    filtered = signal.sosfilt(filter, normalised_block_fn)
    sin_y = np.sin(filtered)
    cos_y = np.cos(filtered)
    sin_diff = np.diff(sin_y, prepend=0)
    cos_diff = np.diff(cos_y, prepend=0)
    cos_diff[0]=0

    phase_vec_l1 = np.stack([sin_y, cos_y],axis=0)
    phase_vec_l2 = np.stack([sin_y*sin_diff, cos_y*cos_diff],axis=0)
    phase_vec_l3 = np.stack([sin_y*sin_diff*velocity, cos_y*cos_diff*velocity],axis=0)
    for f_id, f in enumerate(frames):
        for jo_id, jo in enumerate(f):
            jo["filtered_contact"] = filtered[jo_id, f_id]
            jo["phase_vec_l1"] = phase_vec_l1[:, jo_id, f_id]
            jo["phase_vec_l2"] = phase_vec_l2[:, jo_id, f_id]
            jo["phase_vec_l3"] = phase_vec_l3[:, jo_id, f_id]
            jo["tta"] = ttas[jo_id, f_id]

@ray.remote
def parse(filepath:str, id:str) -> object:
    """
    A remote task function that
        parses the raw  animation sequence from json.
        computes some features
        return the sequence in compressed binary format
    :param filepath:
    :param id:
    :return:
    """
    raw = js.load(open(filepath, "r"))
    df = ParseRawSequence(raw)
    calc_phase_vec_tta(df)
    pickled = pickle.dumps(df)

    del raw, df
    return pickled, id

def main(raw_data_path:str, dirname:str, use_threads:int=THREADING, output_path:str="../../data"):
    """
    Given the path to the raw files, finds and parses all animation data. The data is then compressed as saved to <outputPath>.
    :param raw_data_path:
    :param dirname:
    :param use_threads:
    :param output_path:
    :return:
    """
    if use_threads:
        if (ray.is_initialized()):
            ray.shutdown()
        ray.init(num_cpus=CPUs)

    start = time.time()

    """
    Parse the raw data and compute features
    """
    tasks = []
    for dname, dirs, files in os.walk(raw_data_path):
        for fname in files:
            filePath = os.path.join(dname, fname)
            if use_threads:
                tasks.append(parse.remote(filePath, fname))
            else:
                raw = js.load(open(filePath, "r"))
                df = ParseRawSequence(raw)
                calc_phase_vec_tta(df)
                pickled = pickle.dumps(df)
                Data.save3(pickled, fname.replace(".json", ""), os.path.join(output_path, dirname))

    if use_threads:
        while len(tasks):
            done_id, tasks = ray.wait(tasks)
            pickled, f_id = ray.get(done_id[0])

            Data.save3(pickled, f_id.replace(".json", ""), os.path.join(output_path, dirname))
        ray.shutdown()

    print("Done: {}".format(time.time() - start))


if __name__ == "__main__":
    rawPath = sys.argv[1]
    dataPaths = []
    for dname, dirs, files in os.walk(rawPath):
        for dir in dirs:
            if not dir == ".idea" and not dir == "Unity":
                dataPaths.append(dir)
        break
    for path in dataPaths:
        main(raw_data_path=os.path.join(rawPath,path), dirname=path)


