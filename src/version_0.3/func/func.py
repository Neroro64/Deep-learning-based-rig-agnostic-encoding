"""

miscellaneous functions

"""
import _pickle as pickle
import bz2
import os
import numpy as np
import ray
import torch
import json as js

@ray.remote
def remote_save(file:object, filename:str, path:str):
    """
    Writes and compresses an object to the disk
    :param file:
    :param filename:
    :param path:
    :return:
    """
    with bz2.BZ2File(os.path.join(path, filename+".pbz2"), "w") as f:
        pickle.dump(file, f)


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
                row.append(np.concatenate([f[jj][feature] for jj in joIDs]))
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
    if not ray.is_initialized:
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

def setVec3(struct, vec):
    struct["x"] = vec[0].item()
    struct["y"] = vec[1].item()
    struct["z"] = vec[2].item()

def setVec6(struct, vec):
    for r, cell in enumerate(["x", "y", "z"]):
        for col, column in enumerate(["c0", "c1"]):
            struct[column][cell] = vec[r, col].item()

def insert_pos(template,
               positions=None, rotations=None, velocity=None,
               tPos=None, tRot=None, name="Replay"):
    shape = positions.shape
    for c in range(shape[0]):
        for f in range(shape[1]):
            t = 0
            for j in range(shape[2]):
                jo = template["frames"][f]["joints"][j]
                if positions is not None:
                    setVec3(jo["position"], positions[c,f,j])
                if rotations is not None:
                    setVec3(jo["velocity"], velocity[c,f,j])
                if velocity is not None:
                    setVec6(jo["rotMat"], rotations[c,f,j])


                if jo["key"]:
                    if tPos is not None:
                        setVec3(jo["cost"]["TargetPosition"], tPos[c,f,t])
                    if tRot is not None:
                        setVec6(jo["cost"]["TargetRotation"], tRot[c,f,t])
                    t+=1
        with open("{}_{}.json".format(name, c), "w") as f:
            js.dump(template, f)


def local_generate_animation(model, test_set, feature_dims, template, target_dim=0, output_path=None, use_vae=False, n=5):
    idx = np.arange(n)
    with torch.no_grad():
        x = torch.stack([test_set[i][0] for i in idx])
        y = torch.stack([test_set[i][1] for i in idx])
        x_shape = x.shape
        y_shape = y.shape
        x = x.view(-1, x_shape[-1])

        model.generationModel.reset_hidden(batch_size=x.shape[0])
        if use_vae:
            out, z, mu, logvar = model(x)
        else:
            out, _ = model(x)
        x_c = torch.cat(out,dim=1).detach()
        generated = x_c
        generated = generated.view(y_shape)

    phase= feature_dims["phase_vec_l2"]
    toPosDim = phase+feature_dims["pos"]
    toRotDim = toPosDim + feature_dims["rotMat2"]
    toVelDim = toRotDim + feature_dims["velocity"]

    gPos = generated[:, :, phase:toPosDim]
    gRot = generated[:, :, toPosDim:toRotDim]
    gVel = generated[:, :, toRotDim:toVelDim]

    oPos = y[:, :, phase:toPosDim]
    oRot = y[:, :, toPosDim:toRotDim]
    oVel = y[:, :, toRotDim:toVelDim]

    tPos = y[:, :, -target_dim:-target_dim+3*4]
    tRot = y[:, :, -target_dim+3*4:]

    clip_length = gPos.shape[1]
    gPos_r = gPos.reshape((n, clip_length, -1, 3))
    gRot_r = gRot.reshape((n, clip_length, -1, 3, 2))
    gVel_r = gVel.reshape((n, clip_length, -1, 3))

    oPos_r = oPos.reshape((n, clip_length, -1, 3))
    oRot_r = oRot.reshape((n, clip_length, -1, 3, 2))
    oVel_r = oVel.reshape((n, clip_length, -1, 3))

    tPos_r = tPos.reshape((n, clip_length, -1, 3))
    tRot_r = tRot.reshape((n, clip_length, -1, 3, 3))

    insert_pos(template, oPos_r, oRot_r, oVel_r, tPos_r, tRot_r,output_path+"_O")
    insert_pos(template, gPos_r, gRot_r, gVel_r, tPos_r, tRot_r,output_path+"_G")



@ray.remote
def generate_animation(model, test_set, feature_dims, template, target_dim=0, output_path=None, use_vae=False, n=5):
    idx = np.arange(n)
    with torch.no_grad():
        model.eval()

        x = torch.stack([test_set[i][0] for i in idx])
        y = torch.stack([test_set[i][1] for i in idx])
        shape = x.shape
        x = x.view(-1, shape[-1])

        if use_vae:
            out, z, mu, logvar = model(x.cuda())
        else:
            out, _ = model(x.cuda())
        x_c = torch.cat(out,dim=1).detach()
        generated = x_c
        generated = generated.view(shape).to("cpu")

    phase= feature_dims["phase_vec_l2"]
    toPosDim = phase+feature_dims["pos"]
    toRotDim = toPosDim + feature_dims["rotMat2"]
    toVelDim = toRotDim + feature_dims["velocity"]

    gPos = generated[:, :, phase:toPosDim]
    gRot = generated[:, :, toPosDim:toRotDim]
    gVel = generated[:, :, toRotDim:toVelDim]

    oPos = y[:, :, phase:toPosDim]
    oRot = y[:, :, toPosDim:toRotDim]
    oVel = y[:, :, toRotDim:toVelDim]

    tPos = y[:, :, -target_dim:-target_dim+3*4]
    tRot = y[:, :, -target_dim+3*4:]

    clip_length = gPos.shape[1]
    gPos_r = gPos.reshape((n, clip_length, -1, 3))
    gRot_r = gRot.reshape((n, clip_length, -1, 3, 2))
    gVel_r = gVel.reshape((n, clip_length, -1, 3))

    oPos_r = oPos.reshape((n, clip_length, -1, 3))
    oRot_r = oRot.reshape((n, clip_length, -1, 3, 2))
    oVel_r = oVel.reshape((n, clip_length, -1, 3))

    tPos_r = tPos.reshape((n, clip_length, -1, 3))
    tRot_r = tRot.reshape((n, clip_length, -1, 3, 3))

    insert_pos(template, oPos_r, oRot_r, oVel_r, tPos_r, tRot_r,output_path+"_O")
    insert_pos(template, gPos_r, gRot_r, gVel_r, tPos_r, tRot_r,output_path+"_G")


@ray.remote
def generate_animation_ae(model, test_set, feature_dims, template, output_path, n=5):
    idx = np.arange(n)
    with torch.no_grad():
        model.eval()

        x = torch.stack([test_set[i][0] for i in idx])
        y = torch.stack([test_set[i][1] for i in idx])
        shape = x.shape
        x = x.view(-1, shape[-1])

        out = model(x.cuda())

        generated = out
        generated = generated.view(shape)

    toPosDim = feature_dims["pos"]
    toRotDim = toPosDim + feature_dims["rotMat2"]
    toVelDim = toRotDim + feature_dims["velocity"]

    gPos = generated[:, :, :toPosDim]
    gRot = generated[:, :, toPosDim:toRotDim]
    gVel = generated[:, :, toRotDim:toVelDim]

    oPos = y[:, :, :toPosDim]
    oRot = y[:, :, toPosDim:toRotDim]
    oVel = y[:, :, toRotDim:toVelDim]

    clip_length = gPos.shape[1]
    gPos_r = gPos.reshape((n, clip_length, -1, 3))
    gRot_r = gRot.reshape((n, clip_length, -1, 3, 2))
    gVel_r = gVel.reshape((n, clip_length, -1, 3))

    oPos_r = oPos.reshape((n, clip_length, -1, 3))
    oRot_r = oRot.reshape((n, clip_length, -1, 3, 2))
    oVel_r = oVel.reshape((n, clip_length, -1, 3))

    insert_pos(template, oPos_r, oRot_r, oVel_r, None, None,output_path+"_O")
    insert_pos(template, gPos_r, gRot_r, gVel_r, None, None, output_path+"_G")

