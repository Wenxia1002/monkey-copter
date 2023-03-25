#-- coding:UTF-8 --
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader
import sentencepiece as spm
import os
import torchsummary
from . import config
from .Transformer import Transformer as Transformer
from .dataSet import MKDataSet as MKDataSet
from . import dataSet as DS
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = MKDataSet(max_len=10)
loader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=config.batch_size)

def predictTrajectory(states, wps, model):
    '''
    :param states:
    :param profiles:
    :return: predicted trajectory list[(x,y,z)..]
    1. load profiles to states
    2. calculate initial points
    3. transformer trajectory list to redusidual list by minus initial points (and *1000)
    4. for loop: for every 10 states with waypoints, using loaded model to predict
    '''
    handled_states_with_wps = []
    for i in range(wps.shape[0]):
        for j in range(states.shape[1]):
            states[i][j][3] = wps[i][0]
            states[i][j][4] = wps[i][1]
            states[i][j][5] = wps[i][2]
            handled_states_with_wps.append(states[i][j][0:6])

    initial_state = [handled_states_with_wps[0][0], handled_states_with_wps[0][1], handled_states_with_wps[0][2]]
    scaled_residual_list = DS.transferStates2ScaledResidual(handled_states_with_wps)

    src_list = []

    src_len = 10
    for k in range(len(scaled_residual_list) - src_len):
        src_list.append(scaled_residual_list[k: k + src_len])


    predicted_scaled_residual_list = []
    for enc_input in src_list:
        # enc_input = torch.Tensor(s)
        # print(enc_input.shape)
        out = model.translate(torch.Tensor([enc_input]))
        predicted_scaled_residual_list.append(out.tolist())

    Result = DS.addScaledResidual2InitialState(initial_state, predicted_scaled_residual_list)
    return Result
    # return

def draw3DPicture(x, y, z, xlabel='x', ylabel='y', zlabel='z', title=''):
    plt.subplot(projection='3d')
    plt.plot(x, y, z)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.zlabel(zlabel)
    plt.title(title)
    plt.show()

def drwa3DTrajectoryAndProfile(xt, yt, zt, xp, yp, zp, xlabel='x', ylabel='y', zlabel='z', title=''):
    plt.subplot(projection='3d')
    plt.plot(xt, yt, zt)
    plt.plot(xp, yp, zp)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.zlabel(zlabel)
    plt.title(title)
    plt.show()

def check_outlier_AR_based(errors, std=6):
    if len(errors) <= 4:
        return True
    ## dismiss the first prediction
    means = np.mean(errors[1:-1],axis=0) # means = [mean of x_error, mean of y_error, mean of z_error]
    stds = np.std(errors[1:-1],axis=0) # stds = [std of x_error, std of y_error, std of z_error]
    std_n = std
    for i in range(len(means)):
        if errors[-1][i] > means[i] + std_n * stds[i] or errors[-1][i] < means[i] - std_n * stds[i]:
            return False
    return True

def TransformerBasedLabel(state, profile, std = 6):
    model = Transformer().to(config.device)
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))
        print("加载模型成功,参数总量为 :", sum(p.numel() for p in model.parameters() if p.requires_grad))

    state = np.array(state)

    predict_results = predictTrajectory(torch.Tensor([state]), torch.Tensor([profile]), model)

    for i in range(len(predict_results)):
        err = state[i + 10][:3] - predict_results[i][:3]
        if not check_outlier_AR_based(err,std):
            return False
    return True






def test2(states, wps):
    states = np.array(states)
    model = Transformer().to(config.device)
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))
    predictedTrajectory = predictTrajectory(states, wps, model)
    # print(torch.Tensor(predictedTrajectory).shape)
    predictedTrajectory = np.array(predictedTrajectory)
    states1 = numpy.reshape(states, (states.shape[0] * states.shape[1], states.shape[2]))
    print(states1.shape)
    drwa3DTrajectoryAndProfile(predictedTrajectory[:, 0], predictedTrajectory[:, 1], predictedTrajectory[:, 2], states1[:,0], states1[:,1], states1[:,2] )



def test1(states, wps, label):
    print(len(wps))
    for mission_id in range(0, len(wps)):  # $mission_id$ is in fact way point id, not the mission id.
        if mission_id % 3 == 0:  # NOTE: all the simulations are assumed to be run in the Guided mode!
            continue
        state_temp = states[mission_id]  # $state_temp$ is in fact a trajectory segment, corresponding to the $mission_id$th way point.
        # by default, there should be 20 elements in the trajectory segment.
        if len(state_temp) > 60:
            state_temp = state_temp[:80:4]
        elif len(state_temp) > 20:
            state_temp = state_temp[:40:2]
        profile_temp = wps[mission_id][:3]  # here we only focus on lat, lon, alt, profile_temp is in fact [lat, lon, alt] of the
        # $mission_id$th way point.
        if not TransformerBasedLabel(state_temp, profile_temp):
            label = False
            break

if __name__ == '__main__':
    states = np.load('data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/states_np_5000000_0.npy')
    wps = np.load('data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/profiles_np_5000000_0.npy')
    label = True
    test2(states, wps)