from re import S
from tkinter import FALSE
from tokenize import group

from cv2 import trace
from traceLabel import *
from injector_copter_3_6_12_V1 import *
from configparser import ConfigParser
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import numpy as np
import sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
traj_paths=['trajectories/realife_bugfree_3000_cloud2/','trajectories/realife_bugfree_3000_cloud3/','trajectories/realife_bugfree_3000_cloud4/',
            'trajectories/realife_single_2_3000/','trajectories/realife_single_3_3000/','trajectories/realife_single_4_3000/',
            'trajectories_artificial_single/artificial_single_0_3000/','trajectories_artificial_single/artificial_single_1_3000/','trajectories_artificial_single/artificial_single_3_3000/',
            'trajectories_artificial_single/artificial_single_11_3000/','trajectories_artificial_single/artificial_single_14_3000/',
            'trajectories_bugfree_20220606/bugfree_3000_Cloud3/','trajectories_bugfree_20220606/bugfree_3000_Cloud4/']

            
traj_path415_buggy=['traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_0/','traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_1/',
                    'traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_2/','traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_3/',
                    'traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_4/','traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_5/',
                    'traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_6/','traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_7/',
                    'traj415/Ardupilot_4.1.5_buggy_trajectories/bug_version_8/']
#  'traj415/Ardupilot_4.1.5_nonbuggy_trajectories/batch_5/'
traj_path415_nonbuggy=['traj415/Ardupilot_4.1.5_nonbuggy_trajectories/batch_1/','traj415/Ardupilot_4.1.5_nonbuggy_trajectories/batch_2/','traj415/Ardupilot_4.1.5_nonbuggy_trajectories/batch_3/',
                        'traj415/Ardupilot_4.1.5_nonbuggy_trajectories/batch_4/',
                        ]

traj_path36_buggy=['/traj/ArduCopter3_6_12 bug0 10000/','/traj/ArduCopter3_6_12 bug1 10000/',
                   '/traj/ArduCopter3_6_12 bug2 10000/','/traj/ArduCopter3_6_12 bug3 10000/',
                   '/traj/ArduCopter3_6_12 bug4 10000/','/traj/ArduCopter3_6_12 bug5 10000/',
                   '/traj/ArduCopter3_6_12 bug6 10000/','/traj/ArduCopter3_6_12 bug7 10000/',
                   '/traj/ArduCopter3_6_12 bug8 10000/','/traj/ArduCopter3_6_12 bug9 10000/',
                   '/traj/ArduCopter3_6_12 bug10 10000/','/traj/ArduCopter3_6_12 bug11 10000/',
                   '/traj/ArduCopter3_6_12 bug12 10000/','/traj/ArduCopter3_6_12 bug13 10000/',
                   '/traj/ArduCopter3_6_12 bug14 10000/']
traj_path36_nonbuggy=[]


def statics(bug_id_list,group,cfg):
    all_lines = set()
    for bug_id in bug_id_list:
        with open(cfg.get('param','root_dir')+group[bug_id]['file'],'r') as f:
            for line_no,line in enumerate(f,1):
                if 'EXECUTE_MARK()' in line:
                    all_lines.add(group[bug_id]['file']+'-'+str(line_no))
    return list(all_lines)

def simulationResultClean(cfg, index_from, index_to,path):
    # TODO: Change the director to the experiment_history
    dir = path+'output/PA/0/'
    # dir = '/home/csszhang/ArduPilotTestbedForOracleResearch_V2/arduPilot/experiment_history/experiment_Artifitual_single_10_11_12_2000_2022_03_27/output/PA/0/'
    simplified_trajectories = [] # was called "states" in the old source code, 
                                 # standardized by Qixin Wang on 27/4/2022.
    way_point_sequences = [] # was called "profiles" in the old source code, 
                             # standardized by Qixin Wang on 27/4/2022.
    for simulation_id in range(index_from, index_to+1): 
    # simulation_id was called "simulate_id" in the old source code, 
    # standardized by Qixin Wang on 27/4/2022.
        trajectory = np.load(dir + 'states_np_%d_0.npy' % simulation_id) 
        # trajectory was called "state" in the old source code, 
        # standardized by Qixin Wang on 27/4/2022.
        simplified_trajectory = []
        for trajectory_segment in trajectory:
            simplified_trajectory.append([[state[0], state[1], state[2]] for state in trajectory_segment])
        way_point_sequences.append(np.load(dir + 'profiles_np_%d_0.npy' % simulation_id))
        simplified_trajectories.append(simplified_trajectory)
    return simplified_trajectories, way_point_sequences

def process_all_server(index_from,index_to,record_path,bug_id_list,is_nonbuggy=0):
    record_files = [f for f in os.listdir(record_path) if f.startswith('server') ]
        
    record_files.sort()
    simplified_trajectories = []
    way_point_sequences = []
    physical_trajectories = []
    count=0

    group=bug_group
    traces = []
    for record_file in record_files:
        path=record_path+record_file
        # dir=record_path+'/output/PA/0/'
        dir=path+'/experiment/output/PA/0/'
        if is_nonbuggy:
            index_from+=(count*10000)
            index_to+=(count*10000)
            if 'batch_2/server18_1081000_1081999' in path:
                index_to=1081830
            
        else:
            index_from+=(count*1000)
            index_to+=(count*1000)
        count=1
        print("processing "+record_file+"...")
        
        for simulation_id in range(index_from, index_to+1): 
        # simulation_id was called "simulate_id" in the old source code, 
        # standardized by Qixin Wang on 27/4/2022.
            # if record_file.startswith('server4'):
            #     id=simulation_id-10000
            # else:
            id=simulation_id
            trajectory = np.load(dir + 'states_np_%d_0.npy' % id,allow_pickle=True) 
            # trajectory was called "state" in the old source code, 
            # standardized by Qixin Wang on 27/4/2022.
            simplified_trajectory = []
            physical_trajectory = []
            for trajectory_segment in trajectory:
                simplified_trajectory.append([state for state in trajectory_segment])
                physical_trajectory.append([[x[3],x[4],x[5],x[6],x[7],x[8]] for x in trajectory_segment])
            way_point_sequences.append(np.load(dir + 'profiles_np_%d_0.npy' % id))
            simplified_trajectories.append(simplified_trajectory)
            physical_trajectories.append(physical_trajectory)

            current_trace = []
            with open(dir+'raw_%s_0.txt'%id) as f:
                for line in f: ### each line no appears only once
                    for bug_id in bug_id_list:
                        if group[bug_id]['file'] in line:
                            temp_str = line.split('-')[2].strip()
                            if ':' in temp_str: # the record of sum of execution times
                                # todo : take advantage of the sum times
                                continue
                            new_line = group[bug_id]['file']+'-'+str(temp_str)
                            current_trace.append(new_line)
                            break
            current_trace.sort()
            if len(current_trace) == 0:
                print('log execution traces fail for simulate_id %d'%simulation_id)
            traces.append(current_trace)

            
        
    return simplified_trajectories,way_point_sequences,physical_trajectories,traces

        
def extractionTraces_allserver(bug_id_list,group,index_from,index_to,record_path,is_nonbuggy):
    record_files = [f for f in os.listdir(record_path) if f.startswith('server') ]
    num=0
    traces = []
    for record_file in record_files:
        path=record_path+record_file
        dir=path+'/experiment/output/PA/0/'
        if is_nonbuggy:
            index_from+=(num*10000)
            index_to+=(num*10000)
        else:
            index_from+=(num*1000)
            index_to+=(num*1000)
        print("extraction "+record_file+"...")
        num=1
        for simulate_id in range(index_from,index_to+1):
            current_trace = []
            with open(dir+'raw_%s_0.txt'%simulate_id) as f:
                for line in f: ### each line no appears only once
                    for bug_id in bug_id_list:
                        if group[bug_id]['file'] in line:
                            temp_str = line.split('-')[2].strip()
                            if ':' in temp_str: # the record of sum of execution times
                                # todo : take advantage of the sum times
                                continue
                            new_line = group[bug_id]['file']+'-'+str(temp_str)
                            current_trace.append(new_line)
                            break
            current_trace.sort()
            if len(current_trace) == 0:
                print('log execution traces fail for simulate_id %d'%simulate_id)
            traces.append(current_trace)


    return traces



def executionTracesClean(bug_id_list,group,start,end,cfg,traj_path):
    output_dir = traj_path+'output/PA/0/'
    traces = []
    for simulate_id in range(start,end):
        current_trace = []
        with open(output_dir+'raw_%s_0.txt'%simulate_id) as f:
            for line in f: ### each line no appears only once
                for bug_id in bug_id_list:
                    if group[bug_id]['file'] in line:
                        temp_str = line.split('-')[2].strip()
                        if ':' in temp_str: # the record of sum of execution times
                            # todo : take advantage of the sum times
                            continue
                        new_line = group[bug_id]['file']+'-'+str(temp_str)
                        current_trace.append(new_line)
                        break
        current_trace.sort()
        if len(current_trace) == 0:
            print('log execution traces fail for simulate_id %d'%simulate_id)
        traces.append(current_trace)
    return traces

def simulationPhysicalTrajectoriesExtractionHA(cfg,index_from,index_to,traj_path):
    output_dir = traj_path+'output/PA/0/'
    dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
    physical_trajectories = []
    for simulate_id in range(index_from,index_to+1):

        physical_trajectory_full = np.load(output_dir + 'states_np_%d_0.npy'%simulate_id)
        physical_trajectory = []
        for state in physical_trajectory_full:
            physical_trajectory.append([[x[3],x[4],x[5],x[6],x[7],x[8]] for x in state])
        physical_trajectories.append(physical_trajectory)
    return physical_trajectories
   
def parserConfig():
    cfg = ConfigParser()
    # cfg.read('config.ini')
    cfg.read(os.path.join(BASE_DIR ,'config.ini'))
    config = {}
    config['root_dir'] = cfg.get('param','root_dir')
    config['real_life'] = cfg.get('param','real_life')
    config['mutiple_bugs'] = cfg.get('param','mutiple_bugs')
    config['start'] = int(cfg.get('param','start'))
    config['end'] = int(cfg.get('param','end'))
    config['rounds'] = int(cfg.get('param','rounds'))
    return config


def process(config,std):
    states_all=[]# states_all 1476 dimensions
    labels_ARSI = []
    labels_HO = []
    labels_Truth=[]
    start = 2000000
    end = 2000999
    num=0
    bug_list=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14]]
    index=-1
    group=bug_group
    for traj_path1 in traj_path36_buggy:

        print("in "+ traj_path1)
        record_path = '/home/ubuntu/workspace/3.6.12/ardupilot' + traj_path1
        start+=(num*10000)
        end+=(num*10000)
        Truth_0=0
        Truth_1=0
        index=index+1
        bug_id_list=bug_list[index]
        
        states,profiles,states2,traces=process_all_server(start,end,record_path,bug_id_list,0)

        bug_id=bug_id_list[0]
        print("bug id: %d"%bug_id)
        for i in range(0,len(traces)):
            flag=FALSE
            
            
            for line_no in group[bug_id]['lineno']:
                line = group[bug_id]['file'] + '-' + str(line_no)
                if line in traces[i] :
                    flag=True
                    break
            
            if flag:
                labels_Truth.append(1)
                Truth_1=Truth_1+1
            else:
                labels_Truth.append(0)
                Truth_0=Truth_0+1
        
        print("Truth:\n")
        print("1: %d\n"%Truth_1)
        print("0: %d\n"%Truth_0)
        
        num=1
        # for states_id,state in enumerate(states):
        #     tmp=[]
        #     last_profile=[0,0,0]
        #     for task_id,task in enumerate(state):
        #             # add waypoint infomation
        #         tmp.append(profiles[states_id][task_id][0])
        #         tmp.append(profiles[states_id][task_id][1])
        #         tmp.append(profiles[states_id][task_id][2])
        #         if task_id==0:
        #             last_profile=task[0]
        #         delta_target=(profiles[states_id][task_id]-last_profile)/len(task)

        #         for count,point in enumerate(task):
        #             tmp.append(point[0])
        #             tmp.append(point[1])
        #             tmp.append(point[2])
        #             tmp.append(point[0]-(delta_target[0]*count+last_profile[0]))
        #             tmp.append(point[1]-(delta_target[1]*count+last_profile[1]))
        #             tmp.append(point[2]-(delta_target[2]*count+last_profile[2]))
                
        #         last_profile=profiles[states_id][task_id]
                
                    
        #     states_all.append(tmp)
        
        # ARSI_0=0
        # ARSI_1=0
        # for simulate_id,state in enumerate(states): # the len of state is 12, which means a trajectory 
        #     profile = profiles[simulate_id]# the profile is all the waypoints in a trajectory (12)
        #     label = True
        #     for mission_id in range(0,len(profile)):
        #         if mission_id % 3 == 0:
        #             continue
        #         state_temp = state[mission_id] # all the states for one mission in 2s
        #         if len(state_temp) > 60:
        #             state_temp = state_temp[:80:4]
        #         elif len(state_temp) > 20:
        #             state_temp = state_temp[:40:2]
        #         profile_temp = profile[mission_id][:AR_dimension] # here we only focus on lat, lon, alt
        #         if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
        #             label = False
        #             break
        #     if label:
        #         labels_ARSI.append(0)
        #         ARSI_0=ARSI_0+1
        #     else:
        #         labels_ARSI.append(1)
        #         ARSI_1=ARSI_1+1
        
        # print("ARSI:\n")
        # print("1: %d\n"%ARSI_1)
        # print("0: %d\n"%ARSI_0)

        # HO_0=0
        # HO_1=0
        # for simulate_id,state in enumerate(states2):
        #     label = True
        #     for mission_id in range(0,len(state)):
        #         state_temp = state[mission_id]
        #         pre_state = []
        #         for s in state_temp:
        #             if np.abs(s[3]) > 31.5 or np.abs(s[4]) > 28.16 or np.abs(s[5]) > 30:
        #                 label = False
        #                 break
        #             if np.sqrt(np.power(s[3],2) + np.power(s[4],2) + np.power(s[5],2)) > 20:
        #                 label = False
        #                 break
        #             if len(pre_state) == 0:
        #                 pre_state = s
        #             elif np.abs(s[0] - pre_state[0]) / 0.1 > 3.5 or np.abs(s[2] - pre_state[2]) / 0.1 > 3.6:
        #                 label = False
        #                 break
        #         if label == False:
        #             break
        #     if label:
        #         labels_HO.append(0)
        #         HO_0=HO_0+1
        #     else:
        #         labels_HO.append(1)
        #         HO_1=HO_1+1
        # print("HO:\n")
        # print("1: %d\n"%HO_1)
        # print("0: %d\n"%HO_0)



    start=1000000
    end=1000999
    num=0
    for traj_path1 in traj_path36_nonbuggy:
        print("in "+ traj_path1)
        record_path = config['root_dir'] + traj_path1
        start+=(num*1000)
        end+=(num*1000)
        states,profiles,states2=process_all_server(start,end,record_path,1)
        
        num=1
        # for states_id,state in enumerate(states):
        #     tmp=[]
        #     last_profile=[0,0,0]
        #     for task_id,task in enumerate(state):
        #             # add waypoint infomation
        #         tmp.append(profiles[states_id][task_id][0])
        #         tmp.append(profiles[states_id][task_id][1])
        #         tmp.append(profiles[states_id][task_id][2])
        #         if task_id==0:
        #             last_profile=task[0]
        #         delta_target=(profiles[states_id][task_id]-last_profile)/len(task)

        #         for count,point in enumerate(task):
        #             tmp.append(point[0])
        #             tmp.append(point[1])
        #             tmp.append(point[2])
        #             tmp.append(point[0]-(delta_target[0]*count+last_profile[0]))
        #             tmp.append(point[1]-(delta_target[1]*count+last_profile[1]))
        #             tmp.append(point[2]-(delta_target[2]*count+last_profile[2]))
                
        #         last_profile=profiles[states_id][task_id]
                
        #     states_all.append(tmp)
        
        # for simulate_id,state in enumerate(states): # the len of state is 12, which means a trajectory 
        #     profile = profiles[simulate_id]# the profile is all the waypoints in a trajectory (12)
        #     label = True
        #     for mission_id in range(0,len(profile)):
        #         if mission_id % 3 == 0:
        #             continue
        #         state_temp = state[mission_id] # all the states for one mission in 2s
        #         if len(state_temp) > 60:
        #             state_temp = state_temp[:80:4]
        #         elif len(state_temp) > 20:
        #             state_temp = state_temp[:40:2]
        #         profile_temp = profile[mission_id][:AR_dimension] # here we only focus on lat, lon, alt
        #         if not LinearRegressionBasedLabel(state_temp,profile_temp,std=std):
        #             label = False
        #             break
        #     if label:
        #         labels_ARSI.append(0)
        #     else:
        #         labels_ARSI.append(1)
            



        # for simulate_id,state in enumerate(states2):
        #     label = True
        #     for mission_id in range(0,len(state)):
        #         state_temp = state[mission_id]
        #         pre_state = []
        #         for s in state_temp:
        #             if np.abs(s[3]) > 31.5 or np.abs(s[4]) > 28.16 or np.abs(s[5]) > 30:
        #                 label = False
        #                 break
        #             if np.sqrt(np.power(s[3],2) + np.power(s[4],2) + np.power(s[5],2)) > 20:
        #                 label = False
        #                 break
        #             if len(pre_state) == 0:
        #                 pre_state = s
        #             elif np.abs(s[0] - pre_state[0]) / 0.1 > 3.5 or np.abs(s[2] - pre_state[2]) / 0.1 > 3.6:
        #                 label = False
        #                 break
        #         if label == False:
        #             break
        #     if label:
        #         labels_HO.append(0)
        #     else:
        #         labels_HO.append(1)

    # np.save("train_36_6000.npy",states_all)
    # np.save("label_ARSI_36_6000.npy",labels_ARSI)
    # np.save("label_HO_36_6000.npy",labels_HO)
    # np.save("lanel_Truth_36_6000.npy",labels_Truth)
        
    

    #     for record_file in record_files:
    #         print(record_path+record_file)
    #         cfg = ConfigParser()
    #         cfg.read(record_path+record_file)
    #         # bug_id_list = [int(t.strip()) for t in temp.split(',')]
            
    #         if cfg.get('param','real_life') == 'True':
    #             group = real_life_bug_group
    #         else:
    #             group = bug_group
            
    #         start = 100000
    #         end = 109999
            
    #         # traces = executionTracesClean(bug_id_list,group,start,end,cfg,record_path)
    #         states,profiles=simulationResultClean(cfg,start,end-1,record_path)
    #         states2=simulationPhysicalTrajectoriesExtractionHA(cfg,start,end-1,record_path)

            

            

    #         # print('traces number: ' + str(len(traces)))
    #         # print('states number: ' + str(len(states)))
    #         # print('profiles number: ' + str(len(profiles)))

        


    #         # print("completed!")
    
    # # np.save("train_with_input_39000.npy",states_all)
    # np.save("labels_with_input_39000_ARSI.npy",labels_ARSI)

    # np.save("labels_with_input_39000_HumanOracle.npy",labels_HO)

def label_with_truth():
    tmp=[]
    for i in range(0,9000):
        tmp.append(0.0)
    for i in range(0,24000):
        tmp.append(1.0)
    for i in range(0,6000):
        tmp.append(0.0)
    np.save("labels_with_input_39000.npy",tmp)

def label_with_ARSI(states=None,profiles=None,std=6):
    labels = []
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
            labels.append(0)
        else:
            labels.append(1)
    
    np.save("labels_with_input_39000_ARSI.npy",labels)

def label_with_HumanOracle(states2=None,profiles=None,std=6):
    labels = []
    for simulate_id,state in enumerate(states2):
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
            labels.append(0)
        else:
            labels.append(1)
    np.save("labels_with_input_39000_HumanOracle.npy",labels)



if __name__ == '__main__':
    # label_with_truth()
    
    config = parserConfig()
    process(config,6.0)
    exit(0)

    
        
