import os
import numpy as np
import json as js
import _pickle as pickle
import bz2
import ray
import time
import sys
import func

"""
Global variables
"""
PATH = "C:/Users/Neroro/AppData/LocalLow/DefaultCompany/Procedural Animation"
THREADING = True
CPUs = 24


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


def ParseRawSequence(clip:dict) -> dict:
    """
    Parses and converts an AnimationSequence object to Python dictionary
    :param clip: AnimationSequence
    :return df: dict(targetPos:list, targetRot:list, frames:list(joints: list(features))
    """

    targetPos = clip["frames"][0]["targetsPositions"]
    targetRot = clip["frames"][0]["targetsRotations"]

    df = {}
    df["targetPos"] = [parseFloat3(t) for t in targetPos]
    df["targetRot"] = [parseFloat3(t) for t in targetRot]

    frames = []
    for frame in clip["frames"]:
        joints = []
        for jojo in frame["joints"]:
            jo = {}

            jo["key"] = jojo["key"]
            jo["isLeft"] = jojo["isLeft"]
            jo["chainPos"] = jojo["chainPos"]
            jo["geoDistance"] = jojo["geoDistance"]
            jo["geoDistanceNormalised"] = jojo["geoDistanceNormalised"]

            jo["pos"] = parseFloat3(jojo["position"])
            jo["rotEuler"] = parseFloat3(jojo["rotEuler"])
            jo["rotQuaternion"] = parseFloat4(jojo["rotQuaternion"])
            jo["rotMat"] = parseFloat3x3(jojo["rotMat"])

            jo["inertialObj"] = parseFloat3x3(jojo["inertiaObj"])
            jo["inertial"] = parseFloat3x3(jojo["inertia"])

            jo["velocity"] = parseFloat3(jojo["velocity"])
            jo["angularVelocity"] = parseFloat3(jojo["angularVelocity"])
            jo["linearMomentum"] = parseFloat3(jojo["linearMomentum"])
            jo["angularMomentum"] = parseFloat3(jojo["angularMomentum"])
            jo["totalMass"] = jojo["totalMass"]

            jo["upperLimit"] = parseFloat3({"x": jojo["x"]["UpperLimit"], "y":jojo["y"]["UpperLimit"], "z":jojo["z"]["UpperLimit"]})
            jo["lowerLimit"] = parseFloat3({"x": jojo["x"]["LowerLimit"], "y":jojo["y"]["LowerLimit"], "z":jojo["z"]["LowerLimit"]})
            jo["targetValue"] = parseFloat3({"x": jojo["x"]["TargetValue"], "y":jojo["y"]["TargetValue"], "z":jojo["z"]["TargetValue"]})
            jo["currentValue"] = parseFloat3({"x": jojo["x"]["CurrentValue"], "y":jojo["y"]["CurrentValue"], "z":jojo["z"]["CurrentValue"]})
            jo["currentVelocity"] = parseFloat3({"x": jojo["x"]["CurrentVelocity"], "y":jojo["y"]["CurrentVelocity"], "z":jojo["z"]["CurrentVelocity"]})
            jo["currentAcceleration"] = parseFloat3({"x": jojo["x"]["CurrentAcceleration"], "y":jojo["y"]["CurrentAcceleration"], "z":jojo["z"]["CurrentAcceleration"]})

            jo["tGoalPos"] = parseFloat3(jojo["tGoalPosition"])
            jo["tGoalDir"] = parseFloat3(jojo["tGoalDirection"])

            jo["posCost"] = parseFloat3(jojo["cost"]["ToTarget"])
            jo["rotCost"] = parseFloat3(jojo["cost"]["ToTargetRotation"])
            jo["targetPosition"] = parseFloat3(jojo["cost"]["TargetPosition"])
            jo["targetRotation"] = parseFloat3x3(jojo["cost"]["TargetRotation"])

            jo["contact"] = jojo["contact"]

            joints.append(jo)
        frames.append(joints)
    df["frames"] = frames
    return df


def computeFeatures(clip:dict, window:int=12, clipSize:int=120) -> None:
    """
    Computes some extra features given the parsed animation sequence
    :param clip:
    :param window:
    :param clipSize:
    :return:
    """

    contactFrame = 0
    contactJoID = 0
    for i in range(len(clip["frames"])-1):
        f = clip["frames"][i]
        f2 = clip["frames"][i+1]
        for j in range(len(f)):
            jo = f[j]
            jo2 = f2[j]

            jo["posTrajectoryVec"] = jo2["pos"] - jo["pos"]    # position change
            jo["dirTrajectoryVec"] = jo2["rotEuler"] - jo["rotEuler"]   # rotation change

            if jo["key"] and jo["contact"] and contactFrame == 0:
                contactFrame = i
                contactJoID = j

            jo["phase"] = [0, 0]
            jo["tta"] = [0]

    if (contactFrame > 0):
        cycle = contactFrame / clipSize
        for i in range(window, 0, -1):
            index = contactFrame - i
            if (index < 0):
                continue
            clip["frames"][index][contactJoID]["tta"] = i
            clip["frames"][index][contactJoID]["phase"] = [1 - i/window, 0]
        for i in range(clipSize-1):
            clip["frames"][i][contactJoID]["phase"][1] = (i%cycle)/cycle * 2 * np.pi


@ray.remote
def parse(raw:dict) -> object:
    """
    A remote task function that
        parses the raw  animation sequence from json.
        computes some features
        return the sequence in compressed binary format
    :param raw:
    :return:
    """
    df = ParseRawSequence(raw)
    computeFeatures(df)
    return pickle.dumps(df)


def main(rawDataPath:str, outputName:str="dataset", path:str=PATH, use_threads:int=THREADING, outputPath:str="../data"):
    """
    Given the path to the raw files, finds and parses all animation data. The data is then compressed as saved to <outputPath>.
    :param rawDataPath:
    :param outputName:
    :param path:
    :param use_threads:
    :param outputPath:
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
    dfs = []
    for dname, dirs, files in os.walk(os.path.join(path, rawDataPath)):
        for fname in files:
            filePath = os.path.join(dname, fname)
            raw = js.load(open(filePath, "r"))
            if use_threads:
                raw_obj = ray.put(raw)
                df = parse.remote(raw_obj)
                dfs.append(df)
            else:
                df = ParseRawSequence(raw)
                computeFeatures(df)
                save(df, fname.replace(".json", ""), rawDataPath)

    if use_threads:
        data = []
        while len(dfs):
            done_id, dfs = ray.wait(dfs)
            f = ray.get(done_id[0])
            data.append(f)

        save(data, os.path.join(outputPath, outputName))
        ray.shutdown()

    print("Done: {}".format(time.time() - start))


if __name__ == "__main__":
    rawPath = sys.argv[1]
    if (rawPath == "-a"):
        dataPaths = []
        for dname, dirs, files in os.walk(os.path.join(PATH)):
            for dir in dirs:
                if not dir == ".idea" and not dir == "Unity":
                    dataPaths.append(dir)
            break
        for path in dataPaths:
            main(rawDataPath=path, outputName=path)

    else:
        output_name = sys.argv[2]
        main(rawDataPath=rawPath, outputName=output_name)

