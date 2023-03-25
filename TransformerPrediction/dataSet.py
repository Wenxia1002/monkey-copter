#-- coding:UTF-8 --
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np

class MKDataSet(Dataset):
    def __init__(self, start_index=0, max_len=None):
        # with open("./data/s04_8000_word2num.plk", "rb") as f:
        #     self.en_num, self.ch_num = pickle.load(f)
        # if max_len is not None:
        #     self.en_num = self.en_num[start_index:start_index + max_len]
        #     self.ch_num = self.ch_num[start_index:start_index + max_len]
        self.src_list = []
        self.tar_list = []
       
        for serial_no in range(5000000, 5001000):
            temp_src_list, temp_tar_list = handleOneTrajectorySerial('/home/ubuntu/workspace/ArduV2/ArduPilotTestbedForOracleResearch_V2/arduPilot/monkey-copter/TransformerPrediction/data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/', serial_no)
            self.src_list += temp_src_list
            self.tar_list += temp_tar_list
        # self.src_list, self.tar_list = handleOneTrajectorySerial('data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/', 5000000)

    def __len__(self):
        assert len(self.src_list) == len(self.tar_list)
        return len(self.src_list)

    def __getitem__(self, index):
        return self.src_list[index], self.tar_list[index]

    def collate_fn(self, batch):

        if batch == []:
            return (None, None, None)
        src_num, tgt_num = [], []
        # src_len, tgt_len = [], []

        for src, tgt in batch:
            src_num.append(src)
            tgt_num.append(tgt)
        # src_max_len = max(src_len)
        # tgt_max_len = max(tgt_len)
        # src_num = [i + [0] * (src_max_len - len(i)) for i in src_num]
        # tgt_num = [i + [0] * (tgt_max_len - len(i)) for i in tgt_num]
        # dec_input_batch = []
        # dec_output_batch = []
        # for tgt in tgt_num:
        #     # print(tgt)
        #     dec_input = np.r_[[[0.0,0.0,0.0,0.0,0.0,0.0]], tgt]
        #     # print(dec_input)
        #     dec_outputs = np.r_[tgt, [[0.0,0.0,0.0,0.0,0.0,0.0]]]
        #     dec_input_batch.append(dec_input)
        #     dec_output_batch.append(dec_outputs)

        return torch.Tensor(src_num), torch.Tensor(src_num), torch.Tensor(tgt_num)

def handleOneTrajectorySerial(path, serial_no, window_size = 10):
    # wps = np.load(path + 'profiles_np_' + str(serial_no) + '_0.npy')
    # states = np.load(path + 'states_np_' + str(serial_no) + '_0.npy')
    
    # states = np.load('data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/states_np_5000000_0.npy')
    # wps = np.load('data/ArduCopter3_6_12 bug_free 10000/server0_5000000_5000999/experiment/output/PA/0/profiles_np_5000000_0.npy')

    states = np.load(path + 'states_np_' + str(serial_no) + '_0.npy')
    wps = np.load(path + 'profiles_np_' + str(serial_no) + '_0.npy')

    handled_states_with_wps = []
    for i in range(wps.shape[0]):
        for j in range(states.shape[1]):
            states[i][j][3] = wps[i][0]
            states[i][j][4] = wps[i][1]
            states[i][j][5] = wps[i][2]
            handled_states_with_wps.append(states[i][j][0:6])

    scaled_residual_list = transferStates2ScaledResidual(handled_states_with_wps)

    src_list = []
    tar_list = []

    for k in range(len(scaled_residual_list) - window_size - 1):
        src_list.append(scaled_residual_list[k: k + window_size])
        tar_list.append(scaled_residual_list[k + 1 : k + window_size + 1])

    return src_list, tar_list

def transferStates2ScaledResidual(src_list):
    initial_state = src_list[0][:3]
    scaled_residual_list = []
    for states_with_wps in src_list:
        scaled_residual = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        scaled_residual[0] = (states_with_wps[0] - initial_state[0]) * 10000
        scaled_residual[1] = (states_with_wps[1] - initial_state[1]) * 10000
        scaled_residual[2] = (states_with_wps[2] - initial_state[2])
        scaled_residual[3] = (states_with_wps[3] - initial_state[0]) * 10000
        scaled_residual[4] = (states_with_wps[4] - initial_state[1]) * 10000
        scaled_residual[5] = (states_with_wps[5] - initial_state[2])
        scaled_residual_list.append(scaled_residual)
    return scaled_residual_list

def addScaledResidual2InitialState(initial_state, scaled_residual_list):
    states_with_wps_list = []
    for scaled_residual in scaled_residual_list:
        states_with_wps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        states_with_wps[0] = initial_state[0] + scaled_residual[0] / 10000
        states_with_wps[1] = initial_state[1] + scaled_residual[1] / 10000
        states_with_wps[2] = initial_state[2] + scaled_residual[2]
        states_with_wps[3] = initial_state[0] + scaled_residual[3] / 10000
        states_with_wps[4] = initial_state[1] + scaled_residual[4] / 10000
        states_with_wps[5] = initial_state[2] + scaled_residual[5]
        states_with_wps_list.append(states_with_wps)
    return states_with_wps_list

if __name__ == '__main__':
    data = MKDataSet()
    dataloder = DataLoader(data, collate_fn=data.collate_fn, batch_size=2)
    for enc_inputs, dec_inputs, dec_outputs in dataloder:
        print('enc_inputs:', enc_inputs)
        print('dec_inputs:', dec_inputs)
        print('dec_outputs:', dec_outputs)
        print("*" * 10)
        pass
