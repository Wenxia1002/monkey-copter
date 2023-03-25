from traceLabel import *
from injector_copter_3_6_12_V1 import *
from configparser import ConfigParser
from prepareforBPNN import *
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import numpy as np
import sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#traj_path36_buggy=['/traj/ArduCopter3_6_12 realife single bug6 10000/']
# traj_path36_buggy=['/traj/ArduCopter3_6_12 multi bug5_8_10_12_13 10000/']
traj_3bug=['/traj/ArduCopter3_6_12 bug0 10000/',
            '/traj/ArduCopter3_6_12 bug1 10000/',
            '/traj/ArduCopter3_6_12 bug2 10000/',
            '/traj/ArduCopter3_6_12 bug3 10000/',
            '/traj/ArduCopter3_6_12 bug4 10000/',
            '/traj/ArduCopter3_6_12 bug5 10000/',
            '/traj/ArduCopter3_6_12 bug6 10000/',
            '/traj/ArduCopter3_6_12 bug7 10000/',
            '/traj/ArduCopter3_6_12 bug8 10000/',
            '/traj/ArduCopter3_6_12 bug9 10000/',
            '/traj/ArduCopter3_6_12 bug10 10000/']

# bug_list=[[6]]
# multi_bug_list=[[5, 8, 10, 12, 13]]
bug3_list=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

def statics(bug_id_list,group,cfg):
    all_lines = set()
    for bug_id in bug_id_list:
        with open(cfg.get('param','root_dir')+group[bug_id]['file'],'r') as f:
            for line_no,line in enumerate(f,1):
                if 'EXECUTE_MARK()' in line:
                    all_lines.add(group[bug_id]['file']+'-'+str(line_no))
    return list(all_lines)

def executionTracesClean(bug_id_list,group,start,end,cfg):
    output_dir = cfg.get('param','root_dir')+'experiment/output/PA/0/'
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
    
# def compressSameValue(suspicious):
#     sus = []
#     for element in suspicious:
#         if len(sus) == 0 or element[0] != sus[-1][0]:
#             sus.append([element[0],[element[1]]])
#         else:
#             sus[-1][1].append(element[1])
#     return sus

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# learning_rate = 0.00000001

# class NeuralNet(nn.Module):
#     def __init__(self,input_size,hidden_size,num_classes):
#         super(NeuralNet,self).__init__()
#         self.fc1 = nn.Linear(input_size,hidden_size)
#         self.sigmoid = nn.Sigmoid()
#         self.fc2 = nn.Linear(hidden_size,num_classes)
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.sigmoid(out)
#         out = self.fc2(out)
#         return out

# class BPNN_Dataset(torch.utils.data.Dataset):
#     def __init__(self,all_lines,traces,labels,transform,train):
#         if train:
#             self.data = torch.zeros((len(traces),len(all_lines)))
#             for i,trace in enumerate(traces):
#                 for j,line in enumerate(all_lines):
#                     if line in trace:
#                         self.data[i][j] = 1.0
#             self.labels = torch.zeros((len(labels),1),dtype=torch.float)
#             for i,label in enumerate(labels):
#                 if labels[i] == 1:
#                     self.labels[i][0] = 1.0
#         else:
#             self.data = torch.eye(len(all_lines))
#             self.labels = torch.zeros((len(all_lines),1))
#         self.transform = transform
    
#     def __getitem__(self,index):
#         # print(self.data[index])
#         return self.data[index], self.labels[index]

#     def __len__(self):
#         return len(self.data)

# def BPNN(all_lines,traces,labels):
#     num_epochs = 20
#     batch_size = 200
#     train_data = BPNN_Dataset(all_lines,traces,labels,transform=transforms.ToTensor(),train=True)
#     train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
#     test_data = BPNN_Dataset(all_lines,traces,labels,transform=transforms.ToTensor(),train=False)
#     test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=1,shuffle=False)
#     model = NeuralNet(len(all_lines),3,1).to(device)
#     criterion = nn.MSELoss()
#     # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#     optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#     ### train the network
    
#     for epoch in range(num_epochs):
#         for i,(data,label) in enumerate(train_loader):
#             data = data.to(device)
#             label = label.to(device)

#             outputs = model(data)
#             # print(outputs)
#             # print(label)
#             loss = criterion(outputs,label)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 5 == 0:
#                 print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1,num_epochs,i+1,len(train_loader),loss.item()))

#     suspicious = []
#     with torch.no_grad():
#         for i,(data, label) in enumerate(test_loader):
#             output = model(data)
#             suspicious.append([output.item(),all_lines[i]])
#     suspicious.sort(reverse=True)
#     return suspicious

def get_success_cover_information(line,traces,labels):
    ncs = 0.0
    ncf = 0.0
    nus = 0.0
    nuf = 0.0
    for trace_id,trace in enumerate(traces):
        if line in trace:
            if labels[trace_id] == 0:
                ncs += 1
            else:
                ncf += 1
        else:
            if labels[trace_id] == 0:
                nus += 1
            else:
                nuf += 1
    return ncs,ncf,nus,nuf

def tarantula(all_lines,traces,labels):
    suspicious = []
    total_failed = sum(labels)
    if total_failed == 0:
        print('no positive labels')
        return 
    total_passed = len(labels) - total_failed
    if total_passed == 0:
        print('no negative labels')
        return
    for line in all_lines:
        ncs,ncf,nus,nuf = get_success_cover_information(line,traces,labels)
        nf = ncf + nuf
        ns = ncs + nus
        if ncs + ncf == 0:
            suspici = 0
        else:
            suspici = ncf / nf / (ncs/ns+ncf/nf)
        suspicious.append([suspici,line])
    suspicious.sort(reverse=True)
    return suspicious
    #return compressSameValue(suspicious)
    
def DStar(all_lines,traces,labels):
    suspicious = []
    for line in all_lines:
        ncs,ncf,nus,nuf = get_success_cover_information(line,traces,labels)
        if nuf + ncs == 0:
            if ncf != 0:
                suspicious.append([sys.float_info.max,line])
            else:
                suspicious.append([0,line])
        else:
            suspicious.append([ncf*ncf/(nuf+ncs),line])
    suspicious.sort(reverse=True)
    return suspicious

def Ochiai(all_lines,traces,labels):
    suspicious = []
    for line in all_lines:
        ncs,ncf,nus,nuf = get_success_cover_information(line,traces,labels)
        nf = ncf + nuf
        nc = ncf + ncs
        if nf == 0 or nc == 0:
            suspicious.append([0,line])
        else:
            suspicious.append([ncf/math.sqrt(nf*nc),line])
    suspicious.sort(reverse=True)
    return suspicious

def Ochiai2(all_lines,traces,labels):
    suspicious = []
    for line in all_lines:
        ncs,ncf,nus,nuf = get_success_cover_information(line,traces,labels)
        if ncf + ncs == 0 or nus + nuf == 0 or ncf + nuf == 0 or ncs + nus == 0:
            suspicious.append([0,line])
        else:
            suspicious.append([(ncf*nus/math.sqrt((ncf+ncs)*(nus+nuf)*(ncf+nuf)*(ncs+nus))),line])
    suspicious.sort(reverse=True)
    return suspicious

def crosstab(all_lines,traces,labels):
    suspicious = []
    total_num = len(labels)
    total_failed = sum(labels)
    total_passed = total_num - total_failed
    for line in all_lines:
        ncs,ncf,nus,nuf = get_success_cover_information(line,traces,labels)
        ns = ncs + nus
        nf = ncf + nuf
        nc = ncs + ncf
        nu = nus + nuf
        ecf = (nc * total_failed) / total_num
        ecs = (nc * total_passed) / total_num
        euf = (nu * total_failed) / total_num
        eus = (nu * total_passed) / total_num
        if ecf == 0:
            t1 = 0
        else:
            t1 = math.pow(ncf-ecf,2)/ecf
        if ecs == 0:
            t2 = 0
        else:
            t2 = math.pow(ncs-ecs,2)/ecs
        if euf == 0:
            t3 = 0
        else:
            t3 = math.pow(nuf-euf,2)/euf
        if eus == 0:
            t4 = 0
        else:
            t4 = math.pow(nus-eus,2)/eus
        chi_square = t1 + t2 + t3 + t4
        M = chi_square / total_num
        if nf != 0 and ncs != 0:
            phi = ncf / nf / ncs * ns
        else:
            phi = 0
        if phi > 1:
            suspicious.append([M,line])
        elif phi == 1:
            suspicious.append([0,line])
        else:
            suspicious.append([-M,line])
    suspicious.sort(reverse=True)
    return suspicious
    #return compressSameValue(suspicious)
    
def print_line_info(all_lines,traces,lineno):
    result = []
    positive_traces_id = []
    for line_no in all_lines:
        temp = 0
        l = []
        for trace_id,trace in enumerate(traces):
            if line_no in trace:
                temp += 1
                if line_no in lineno:
                    positive_traces_id.append(trace_id)
                l.append(trace_id)
        # result.append([line_no,temp])
        result.append([temp,line_no])
        if line_no in lineno:
            print([line_no,temp])
    print(set(positive_traces_id))
    result.sort(reverse=True)
    print(result)

def sus_analysis(lines,sus_list,output_f):
    for sus in sus_list:
        result = []
        if sus == None:
            output_f.write(str(result)+'\n')
            continue
        for line_nos in lines:
            min_rank = 10000
            for line in line_nos:
                for i in range(0,len(sus)):
                    if line == sus[i][1]:
                        #print('line no %s rank #%d sus : %f'%(line,i,sus[i][0]))
                        if i < min_rank:
                            min_rank = i
            if min_rank != 10000:
                result.append(min_rank)
        output_f.write(str(result)+'\n')
        #print(str(result)+'\n')
        #print('~~~~~~~~~~~')

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

def analysis(cfg,bug_id_list,output_f1,output_f2,std,start,end,ind):
    if cfg.get('param','real_life') == 'True':
        group = real_life_bug_group
    else:
        group = bug_group
    #print(bug_id_list)
    
    if ind>=10:
        start = 2000000 + ind%10 * 10000
        end = 2000999 + ind%10 * 10000
    else:
        start = 2000000 + ind * 10000
        end = 2000999 + ind * 10000

    # start=2000000
    # end=2000999

    print("in "+ traj_3bug[ind] )
    record_path = cfg.get('param','root_dir') + traj_3bug[ind]

    
    # bug_id_list=multi_bug_list[ind]
    bug_id_list=bug3_list[ind]
    print(bug_id_list)
    
    all_lines = statics(bug_id_list,group,cfg)
    
    states,profiles,states2,traces=process_all_server(start,end,record_path,bug_id_list,0)

    print('traces number: ' + str(len(traces)))
    #for bug_id in bug_id_list:
        #print( group[bug_id]['file'])
        #print(str(bug_id) + ':' + group[bug_id]['file'])
    # type1='/home/ubuntu/workspace/3.6.12/ardupilot/monkey-copter/bpnn_ARSI_36_6000.pt'
    # type2='/home/ubuntu/workspace/3.6.12/ardupilot/monkey-copter/bpnn_Truth_36_6000.pt'
    # type3='/home/ubuntu/workspace/3.6.12/ardupilot/monkey-copter/bpnn_HO_36_6000.pt'
    # physical_trajectories,reference_trajectories = simulationPhysicalTrajectoriesExtraction(cfg,start,end-1)
    # physical_trajectories_HA = simulationPhysicalTrajectoriesExtractionHA(cfg,start,end - 1)
    # labels1,positive1 = labelTraces_PA(states,profiles,std)
    labels1,positive1 = labelTraces_Transformer(states,profiles,std)
    # labels2,positive2 = labelTraces_HR(states2,profiles,std)
    # labels3,positive3 = labelTraces_NN(physical_trajectories,reference_trajectories,std,type1)
    # labels4,positive4 = labelTraces_NN(physical_trajectories,reference_trajectories,std,type2)
    # labels5,positive5 = labelTraces_NN(physical_trajectories,reference_trajectories,std,type3)
    # labelTraces_BPNN(physical_trajectories,reference_trajectories,std)

    # states_kalman=kalmanExtraction(cfg,start,end - 1)
    # labels_kalman,postive_kalman=labelTraces_KalmanFilter(physical_trajectories,reference_trajectories,std)
    
    positive_id = set()
    for i in range(0,len(traces)):
        for bug_id in bug_id_list:
            for line_no in group[bug_id]['lineno']:
                line = group[bug_id]['file'] + '-' + str(line_no)
                if line in traces[i] :
                    positive_id.add(i)
                    break 
    positive_id = list(positive_id)
    #print('ground truth : ' + str(len(positive_id)))
    negative_id = []
    for i in range(0,len(traces)):
        if i not in positive_id:
            negative_id.append(i)
    false_positive1 = 0
    for id in positive1:
        if id in negative_id:
            false_positive1 += 1
    # false_positive2 = 0
    # for id in positive2:
    #     if id in negative_id:
    #         false_positive2 += 1
    # false_positive3 = 0
    # for id in positive3:
    #     if id in negative_id:
    #         false_positive3 += 1
    # false_positive4 = 0
    # for id in positive4:
    #     if id in negative_id:
    #         false_positive4 += 1
    # false_positive5 = 0
    # for id in positive5:
    #     if id in negative_id:
    #         false_positive5 += 1
    false_negative1 = 0
    for i in range(0,len(traces)):
        if i not in positive1 and i in positive_id:
            false_negative1 += 1
    # false_negative2 = 0
    # for i in range(0,len(traces)):
    #     if i not in positive2 and i in positive_id:
    #         false_negative2 += 1
    # false_negative3 = 0
    # for i in range(0,len(traces)):
    #     if i not in positive3 and i in positive_id:
    #         false_negative3 += 1
    # false_negative4 = 0
    # for i in range(0,len(traces)):
    #     if i not in positive4 and i in positive_id:
    #         false_negative4 += 1
    # false_negative5 = 0
    # for i in range(0,len(traces)):
    #     if i not in positive5 and i in positive_id:
    #         false_negative5 += 1
    
    
    output_f1.write('POSITIVE: %d\n'%positive_id.__len__())
    output_f1.write('NEGATIVE: %d\n'%negative_id.__len__())
    output_f1.write('FALSE POSITIVE: %d\n'%false_positive1)
    output_f1.write('FALSE NEGATIVE: %d\n'%false_negative1)

    # output_f2.write('FALSE POSITIVE: %d\n'%false_positive2)
    # output_f2.write('FALSE NEGATIVE: %d\n'%false_negative2)

    # output_f3.write('FALSE POSITIVE: %d\n'%false_positive3)
    # output_f3.write('FALSE NEGATIVE: %d\n'%false_negative3)

    # output_f4.write('FALSE POSITIVE: %d\n'%false_positive4)
    # output_f4.write('FALSE NEGATIVE: %d\n'%false_negative4)

    # output_f5.write('FALSE POSITIVE: %d\n'%false_positive5)
    # output_f5.write('FALSE NEGATIVE: %d\n'%false_negative5)

    if len(negative_id) != 0:
        print('false positive rate1 : %f'%(float(false_positive1)/len(negative_id)))
        output_f1.write('fpr1 : %f\n'%(float(false_positive1)/len(negative_id)))
        # print('false positive rate2 : %f'%(float(false_positive2)/len(negative_id)))
        # output_f2.write('fpr2 : %f\n'%(float(false_positive2)/len(negative_id)))
        # print('false positive rate3 : %f'%(float(false_positive3)/len(negative_id)))
        # output_f3.write('fpr3 : %f\n'%(float(false_positive3)/len(negative_id)))
        # print('false positive rate4 : %f'%(float(false_positive4)/len(negative_id)))
        # output_f4.write('fpr4 : %f\n'%(float(false_positive4)/len(negative_id)))
        # print('false positive rate5 : %f'%(float(false_positive5)/len(negative_id)))
        # output_f5.write('fpr5 : %f\n'%(float(false_positive5)/len(negative_id)))
    else:
        print('false positive rate1 : None')
        output_f1.write('fpr1 : None\n')
        # print('false positive rate2 : None')
        # output_f2.write('fpr2 : None\n')
        # print('false positive rate3 : None')
        # output_f3.write('fpr3 : None\n')
        # print('false positive rate4 : None')
        # output_f4.write('fpr4 : None\n')
        # print('false positive rate5 : None')
        # output_f5.write('fpr5 : None\n')
    if len(positive_id) != 0:
        print('false negative rate1 : %f'%(float(false_negative1)/len(positive_id)))
        output_f1.write('fnr1 : %f\n'%(float(false_negative1)/len(positive_id)))
        # print('false negative rate2 : %f'%(float(false_negative2)/len(positive_id)))
        # output_f2.write('fnr2 : %f\n'%(float(false_negative2)/len(positive_id)))
        # print('false negative rate3 : %f'%(float(false_negative3)/len(positive_id)))
        # output_f3.write('fnr3 : %f\n'%(float(false_negative3)/len(positive_id)))
        # print('false negative rate4 : %f'%(float(false_negative4)/len(positive_id)))
        # output_f4.write('fnr4 : %f\n'%(float(false_negative4)/len(positive_id)))
        # print('false negative rate5 : %f'%(float(false_negative5)/len(positive_id)))
        # output_f5.write('fnr5 : %f\n'%(float(false_negative5)/len(positive_id)))
    else:
        print('false negative rate1 : None')
        output_f1.write('fnr1 : None\n')
        # print('false negative rate2 : None')
        # output_f2.write('fnr2 : None\n')
        # print('false negative rate3 : None')
        # output_f3.write('fnr3 : None\n')
        # print('false negative rate4 : None')
        # output_f4.write('fnr4 : None\n')
        # print('false negative rate5 : None')
        # output_f5.write('fnr5 : None\n')

    sus_tar1 = tarantula(all_lines,traces,labels1)
    # sus_tar2 = tarantula(all_lines,traces,labels2)
    # sus_tar3 = tarantula(all_lines,traces,labels3)
    # sus_tar4 = tarantula(all_lines,traces,labels4)
    # sus_tar5= tarantula(all_lines,traces,labels5)
    sus_cro1 = crosstab(all_lines,traces,labels1)
    # sus_cro2 = crosstab(all_lines,traces,labels2)
    # sus_cro3 = crosstab(all_lines,traces,labels3)
    # sus_cro4 = crosstab(all_lines,traces,labels4)
    # sus_cro5 = crosstab(all_lines,traces,labels5)
    # sus_bp1 = BPNN(all_lines,traces,labels1)
    # sus_bp2 = BPNN(all_lines,traces,labels2)
    # sus_tar1 = DStar(all_lines,traces,labels1)
    # sus_tar2 = DStar(all_lines,traces,labels2)
    # sus_cro1 = Ochiai(all_lines,traces,labels1)
    # sus_cro2 = Ochiai(all_lines,traces,labels2)
    sus_bp1 = Ochiai2(all_lines,traces,labels1)
    # sus_bp2 = Ochiai2(all_lines,traces,labels2)
    # sus_bp3 = Ochiai2(all_lines,traces,labels3)
    # sus_bp4 = Ochiai2(all_lines,traces,labels4)
    # sus_bp5 = Ochiai2(all_lines,traces,labels5)

    lines = []
    for bug_id in bug_id_list:
        temps = []
        for line_no in group[bug_id]['lineno']:
            temps.append(group[bug_id]['file']+'-'+str(line_no))
        lines.append(temps)
    print(lines)
    sus_analysis(lines,[sus_tar1,sus_cro1,sus_bp1],output_f1)
    # sus_analysis(lines,[sus_tar2,sus_cro2,sus_bp2],output_f2)
    # sus_analysis(lines,[sus_tar3,sus_cro3,sus_bp3],output_f3)
    # sus_analysis(lines,[sus_tar4,sus_cro4,sus_bp4],output_f4)
    # sus_analysis(lines,[sus_tar5,sus_cro5,sus_bp5],output_f5)
    output_f1.write(str(len(all_lines))+'\n')
    # output_f2.write(str(len(all_lines))+'\n')
    # output_f3.write(str(len(all_lines))+'\n')
    # output_f4.write(str(len(all_lines))+'\n')
    # output_f5.write(str(len(all_lines))+'\n')

def mainRecord(config,std,id):
    record_path = config['root_dir'] + 'experiment/'
    record_files = [f for f in os.listdir(record_path) if f.startswith('foo') ]
    print(record_files)
    for record_file in record_files:
        print(record_file)
        cfg = ConfigParser()
        cfg.read(record_path+record_file)
        temp = cfg.get('param','bug')[1:-1]
        bug_id_list = [int(t.strip()) for t in temp.split(',')]
        start = int(cfg.get('param','start'))
        end = int(cfg.get('param','end'))
        delta = int((end - start) / 3)
        startArr = [start,start + delta, start + 2 * delta]
        endArr = [start + delta, start + 2 * delta,end]
        analysis(cfg,bug_id_list,output_f1,output_f2,std,start,end,id)
        output_f1.write('------\n')
        output_f2.write('------\n')
        # output_f3.write('------\n')
        # output_f4.write('------\n')
        # output_f5.write('------\n')
        print("completed!")
     #for i in range(len(startArr)):
            #analysis(cfg,bug_id_list,output_f1,output_f2,std,startArr[i],endArr[i])
            #output_f1.write('------\n')
            #output_f2.write('------\n')

if __name__ == '__main__':
    config = parserConfig()
    # mainRecord(config,7.0)
    # exit(0)
    # for id in range(0,3):
    #     print('id = '+ str(id))
    id=0
    for id in range(0,10,1):
        for std in np.arange(4.0,8.5,1):
            output_f1 = open('log/arti_Trans_BUGID_'+str(id)+'_STD_' + str(std)+'.log','w')
            output_f2 = open('log/arti_HO_BUGID_'+str(id)+'_STD_' + str(std)+'.log','w')
            mainRecord(config,std,id)