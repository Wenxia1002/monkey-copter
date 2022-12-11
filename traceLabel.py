import imp
from operator import ne
from matplotlib.pyplot import flag
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import cond
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
import BPNN
from BPNN import MLP


n = 3
q = 3
p = 10
AR_dimension = 3
waypoint_num = 20
delta=0.1
count=0

class MyDataset(Dataset):
    def __init__(self,X):
        self.x_data = X
        self.x_data = self.x_data.astype(np.float32)
        self.len = self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len

def simulationPhysicalTrajectoriesExtraction(cfg,index_from,index_to):
    dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    physical_trajectories = []
    reference_trajectories = []
    for simulate_id in range(index_from,index_to+1):
        physical_trajectory_full = np.load(dir + 'states_np_%d_0.npy'%simulate_id)
        physical_trajectory = []
        for state in physical_trajectory_full:
            physical_trajectory.append([[x[0],x[1],x[2]] for x in state])
        reference_trajectories.append(np.load(dir + 'profiles_np_%d_0.npy'%simulate_id))
        physical_trajectories.append(physical_trajectory)
    return physical_trajectories,reference_trajectories


def simulationPhysicalTrajectoriesExtractionHA(cfg,index_from,index_to):
    dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    physical_trajectories = []
    for simulate_id in range(index_from,index_to+1):
        physical_trajectory_full = np.load(dir + 'states_np_%d_0.npy'%simulate_id)
        physical_trajectory = []
        for state in physical_trajectory_full:
            physical_trajectory.append([[x[3],x[4],x[5],x[6],x[7],x[8]] for x in state])
        physical_trajectories.append(physical_trajectory)
    return physical_trajectories

def kalmanExtraction(cfg,index_from,index_to):
    dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    physical_trajectories = []
    reference_trajectories = []
    for simulate_id in range(index_from,index_to+1):
        physical_trajectory_full = np.load(dir + 'states_np_%d_0.npy'%simulate_id)
        physical_trajectory = []
        for state in physical_trajectory_full:
            physical_trajectory.append([[x[0],x[1],x[2],x[6],x[7],x[8]] for x in state])
        reference_trajectories.append(np.load(dir + 'profiles_np_%d_0.npy'%simulate_id))
        physical_trajectories.append(physical_trajectory)
    return physical_trajectories

# def labelTraces_KalmanFilter(states=None,profiles=None,std=6):
#     true_labels = 0
#     labels = []
#     false_id = []
#     for simulate_id,state in enumerate(states):
#         profile = profiles[simulate_id]
#         label = True
#         for mission_id in range(0,len(profile)):
#             if mission_id % 3 == 0:
#                 continue
#             state_temp = state[mission_id] # all the states for one mission
#             if len(state_temp) > 60:
#                 state_temp = state_temp[:80:4]
#             elif len(state_temp) > 20:
#                 state_temp = state_temp[:40:2]
#             profile_temp = profile[mission_id][:AR_dimension] # here we only focus on lat, lon, alt
#             if not KalmanFilterBasedLabel3(state_temp,profile_temp,std=std,id=mission_id):
#                 label = False
#                 break
#         if label:
#             true_labels += 1
#             labels.append(0)
#         else:
#             labels.append(1)
#             false_id.append(simulate_id)
#     print('total traces:%d'%len(states))
#     print('positive labels:%d'%(len(states)-true_labels))
#     return labels,false_id

# def KalmanFilterBasedLabel3(state,profile,std=6,id=0):
#     #state=[lan,lon,alt]
#     state_temp=state

#     #transition_matrices
#     A=np.array([[1, 0,  0],
#                  [0, 1, 0],
#                  [0, 0, 1]])
#     #transition_covariance
#     #Q = np.eye(3)
#     R=0.1*np.eye(3)
#     #observation_matrics
#     H=A

#     #A=np.array([[1, 0,  0,  delta,  0,      0],
#     #            [0, 1,  0,  0,      delta,  0],
#     #            [0, 0,  1,  0,      0,      delta],
#     #            [0, 0,  0,  1,      0,      0],
#      #           [0, 0,  0,  0,      1,      0],
#      #           [0, 0,  0,  0,      0,      1]])
    
#     #H=np.array([ [1,0,0,0,0,0],
#     #             [0,1,0,0,0,0],
#     #             [0,0,1,0,0,0]])

#     #means0=np.array([0,0,0,0,0,0])
#     #means0=np.array([state_temp[0][0],state_temp[0][1],state_temp[0][2],0,0,0])

#     means0=state_temp[0]
#     covariances0=15*np.eye(3)

#     kf = KalmanFilter(  transition_matrices=A,
#                         #observation_matrices=H,
#                         observation_covariance=R)
#                         #transition_covariance= Q)

#     filter_state=[]
#     errors = []
#     filter_state.append([means0[0],means0[1],means0[2]])
#     for i in range(1,len(state_temp)):
#         new_mean, new_covariance = kf.filter_update(means0, covariances0, state_temp[i])
#         filter_state.append([new_mean[0],new_mean[1],new_mean[2]])
#         means0=new_mean
#         covariances0=new_covariance
#         #if i >= 10:
#         #    errors.append([new_mean[0]-state_temp[i][0],new_mean[1]-state_temp[i][1],new_mean[2]-state_temp[i][2]])
#         #    if not check_outlier_AR_based(errors,std):
#         #        return False
#     #smooth_mean,smooth_cov=kf.smooth(state_temp)
#     # plot_filter(state_temp,filter_state,id)
#     #count=count+1
#     return True


# def KalmanFilterBasedLabel6(state,profile,std=6):
#     #state=[lan,lon,alt,v1,v2,v3]
#     state_temp=state
#     U=profile

#     #transition_matrices
#     A=np.array([[1, 0,  0,  delta,  0,      0],
#                 [0, 1,  0,  0,      delta,  0],
#                 [0, 0,  1,  0,      0,      delta],
#                 [0, 0,  0,  1,      0,      0],
#                 [0, 0,  0,  0,      1,      0],
#                 [0, 0,  0,  0,      0,      1]])

#     #transition_covariance
#     Q = 0.03*np.eye(A.shape[0])

#     #observation_matrics
#     H=np.array([ [1,0,0,0,0,0],
#                  [0,1,0,0,0,0],
#                  [0,0,1,0,0,0]])

#     kf = KalmanFilter(  transition_matrices=A,
#                         #observation_matrices=H,
#                         transition_covariance= Q)

#     means, covariances = kf.filter([state_temp[0]])

#     filtered_state_sequence = [means[0,0]]

#     for i in range(1,len(state_temp)):
#         new_mean, new_covariance = kf.filter_update(
#                                                     means[-1], 
#                                                     covariances[-1], 
#                                                     [state_temp[i]]
#                                                     )
#         means = np.vstack([means,new_mean])
#         covariances = np.vstack([covariances,new_covariance.reshape((1,6,6))])
#         filtered_state_sequence.append(means[i,0])

#     label=False
#     return True
def labelTraces_NN(states=None,profiles=None,std=6,filepath=None):
    labels = []
    false_id = []
    states_all=[]
    
    for states_id,state in enumerate(states):
        tmp=[]
        last_profile=[0,0,0]
        for task_id,task in enumerate(state):
                # add waypoint infomation
            if task_id>=12:
                break
            tmp.append(profiles[states_id][task_id][0])
            tmp.append(profiles[states_id][task_id][1])
            tmp.append(profiles[states_id][task_id][2])
            if task_id==0:
                last_profile=task[0]
            delta_target=(profiles[states_id][task_id]-last_profile)/len(task)

            for count,point in enumerate(task):
                tmp.append(point[0])
                tmp.append(point[1])
                tmp.append(point[2])
                tmp.append(point[0]-(delta_target[0]*count+last_profile[0]))
                tmp.append(point[1]-(delta_target[1]*count+last_profile[1]))
                tmp.append(point[2]-(delta_target[2]*count+last_profile[2]))
            
            last_profile=profiles[states_id][task_id]
            
                
        states_all.append(tmp)

    model1=torch.load(filepath)
    
    for simulate_id in range(len(states_all)):
        data=states_all[simulate_id]
        data=torch.tensor(data)
        output = model1(data)
        values,predicted = torch.max(output.data,0)
        if predicted==0:
            # true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
        
    
    return labels,false_id

def labelTraces_PA(states=None,profiles=None,std=6):
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states): # the len of state is 12, which means a trajectory 
        profile = profiles[simulate_id]# the profile is all the waypoints in a trajectory (12)
        label = True
        for mission_id in range(0,len(profile)):
            if mission_id % 3 == 0:
                continue
            state_temp = state[mission_id] # all the states for one mission in 2s
            if len(state_temp) > 60:
                state_temp = state_temp[:80:4]
            elif len(state_temp) > 20:
                state_temp = state_temp[:40:2]
            profile_temp = profile[mission_id][:AR_dimension] # here we only focus on lat, lon, alt
            if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
                label = False
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
    #print('total traces:%d'%len(states))
    #print('positive labels:%d'%(len(states)-true_labels))
    return labels,false_id

def labelTraces_HR(states=None,profiles=None,std=6):
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        label = True
        for mission_id in range(0,len(state)):
            state_temp = state[mission_id]
            pre_state = []
            for s in state_temp:
                if np.abs(s[3]) > 31.5 or np.abs(s[4]) > 28.16 or np.abs(s[5]) > 30:
                    label = False
                    break
                if np.sqrt(np.power(s[3],2) + np.power(s[4],2) + np.power(s[5],2)) > 20:
                    label = False
                    break
                if len(pre_state) == 0:
                    pre_state = s
                elif np.abs(s[0] - pre_state[0]) / 0.1 > 3.5 or np.abs(s[2] - pre_state[2]) / 0.1 > 3.6:
                    label = False
                    break
            if label == False:
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
    #print('total traces:%d'%len(states))
    #print('positive labels:%d'%(len(states)-true_labels))
    return labels,false_id

def labelTraces_PA1(states=None,profiles=None,std=6):
    true_labels = 0
    labels = []
    false_id = []
    for simulate_id,state in enumerate(states):
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = state[mission_id]
            if len(state_temp) > 60:
                state_temp = state_temp[:80:4]
            elif len(state_temp) > 20:
                state_temp = state_temp[:40:2]
            profile_temp = profile[mission_id][:AR_dimension]
            if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
                label = False
                break
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
    #print('total traces:%d'%len(states))
    #print('positive labels:%d'%(len(states)-true_labels))
    return labels,false_id

def test_labelTraces(states=None,profiles=None,dir=None,start=0,end=1):
    true_labels = 0
    labels = []
    false_id = []
    false_labels_curve = 0
    false_labels_stable = 0
    for simulate_id,state in enumerate(states):
        curve_label = True
        stable_label = True
        profile = profiles[simulate_id]
        label = True
        for mission_id in range(0,len(profile)):
            state_temp = state[mission_id]
            profile_temp = profile[mission_id][:AR_dimension]
            if not LinearRegressionBasedLabel(state_temp,profile_temp,mission_id):
                label = False
                if mission_id > 0 and mission_id % 3 == 0:
                    curve_label = False
                else:
                    stable_label = False
        if label:
            true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
        if not curve_label and stable_label:
            false_labels_curve += 1
        if not stable_label and curve_label:
            false_labels_stable += 1
    false_nums = len(states) - true_labels
    print('total traces:%d'%len(states))
    print('positive labels:%d'%false_nums)
    print('false_curve_labels:%d, proportion : %f'%(false_labels_curve,float(false_labels_curve)/false_nums))
    print('false_stable_labels:%d, proportion : %f'%(false_labels_stable,float(false_labels_stable)/false_nums))
    print('simultaneously false labels:%d, proportion : %f'%(false_nums - false_labels_curve-false_labels_stable,float(false_nums-false_labels_curve-false_labels_stable)/false_nums))

    return labels,false_id

def check_outlier_AR_based(errors,std=6):
    if len(errors) <= 4:
        return True
    ## dismiss the first prediction
    means = np.mean(errors[1:-1],axis=0)
    stds = np.std(errors[1:-1],axis=0)
    std_n = std
    for i in range(len(means)):
        if errors[-1][i] > means[i] + std_n * stds[i] or errors[-1][i] < means[i] - std_n * stds[i]:
            return False
    return True

# def AR_based(state,profile=None):
#     dimension = len(state[0])
#     train,test = state[0:10],state[10:]
#     models = [AR([x[i] for x in train]) for i in range(dimension)]
#     model_fits = [models[i].fit() for i in range(dimension)]
#     window = model_fits[0].k_ar
#     coefs = [model_fits[i].params for i in range(dimension)]
#     history = train[len(train)-window:]
#     history = [history[i] for i in range(0,len(history))]
#     predictions = []
#     errors = []
#     for t in range(len(test)):
#         length = len(history)
#         lag = [history[i] for i in range(length-window,length)]
#         prediction = []
#         error = []
#         for dimension_id in range(dimension):
#             yhat = coefs[dimension_id][0]
#             for d in range(window):
#                 yhat += coefs[dimension_id][d+1] * lag[window-d-1][dimension_id]
#             obs = test[t][dimension_id]
#             prediction.append(yhat)
#             error.append(yhat - obs)
#         predictions.append(prediction)
#         errors.append(error)
#         history.append(test[t])
        
#         if not check_outlier_AR_based(errors):
#             return False
#     return True

def computeYk(k,state,U):
    Yk = state[k-1]
    for index in range(k-2,k-p-1,-1):
        Yk = np.append(Yk,state[index])
    Yk = np.append(Yk,U)
    Yk = np.transpose(Yk)
    return Yk.reshape(n*p+q,1)

def LinearRegressionBasedLabel(state,profile,std=6):
    state = np.array(state)
    labels = []
    lambd = 0.95
    U = profile
    # X_i = state[p].reshape(n,1)
    Yk = computeYk(p,state,U)
    YkT = np.transpose(Yk)
    # Y_i = Yk

    psi_i = np.ones((n,n*p+q))
    phi_i = np.identity(n*p+q)
    phi_i_inv = inv(phi_i)
    A_i = np.mat(psi_i)*np.mat(phi_i_inv)
    errors = []
    i = p
    while i < len(state):
        X_est = np.mat(A_i)*np.mat(Yk)
        errors.append(X_est.A1 - state[i])
        if not check_outlier_AR_based(errors,std):
            return False
        ## start to predict state[i+1], so we need training data up to state[i]
        Yk = computeYk(i+1,state,U) # up to state[i]
        YkT = np.transpose(Yk)
        # X_i = np.concatenate((state[i].reshape(n,1),math.sqrt(lambd)*X_i),axis=1)
        # Y_i = np.concatenate((Yk,math.sqrt(lambd)*Y_i),axis=1)
        psi_i = lambd*psi_i + np.mat(state[i].reshape(n,1))*np.mat(YkT)
        phi_i = lambd*phi_i + np.mat(Yk)*np.mat(YkT)
        phi_i_inv = phi_i_inv / lambd - math.pow(lambd,-2)*np.mat(phi_i_inv)*np.mat(Yk)*inv(np.mat(np.identity(1)+np.mat(YkT)*np.mat(phi_i_inv)*np.mat(Yk)/lambd))*np.mat(YkT)*np.mat(phi_i_inv)
        A_i = np.mat(psi_i)*np.mat(phi_i_inv)
        i += 1
    return True


if __name__ == '__main__':
    start = 0
    end = start
    exit(0)


