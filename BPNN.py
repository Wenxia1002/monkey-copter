import imp
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F   
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset,Dataset,DataLoader
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


has_input=1

path_trainX1='/home/ubuntu/workspace/3.6.12/ardupilot/train_36_6000.npy'
path_trainY1='/home/ubuntu/workspace/3.6.12/ardupilot/label_Truth_36_6000.npy'
path_trainX2='/home/ubuntu/workspace/3.7/ArduPilotTestbedForOracleResearch_V2/arduPilot/monkey-copter/train_no_input_39000.npy'
path_trainY2='/home/ubuntu/workspace/3.7/ArduPilotTestbedForOracleResearch_V2/arduPilot/monkey-copter/labels_no_input_39000.npy'


if has_input==1:
    input_dim=1476
    pathX=path_trainX1
    pathY=path_trainY1
else:
    input_dim=720
    pathX=path_trainX2
    pathY=path_trainY2
class MyDataset(Dataset):
    def __init__(self,X,Y):
        # data=np.load(os.path.join(BASE_DIR ,'train.npy'))
        # label=np.load(os.path.join(BASE_DIR ,'labelsfortrain.txt'))
        self.x_data = X
        self.x_data = self.x_data.astype(np.float32)
        self.y_data = Y
        self.y_data = self.y_data.astype(np.float32)
        self.len = self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class MLP(torch.nn.Module):   
    def __init__(self):
        super(MLP,self).__init__()     
        
        self.fc1 = torch.nn.Linear(input_dim,512)  
        self.fc2 = torch.nn.Linear(512,256)  
        self.fc3 = torch.nn.Linear(256,32) 
        self.fc4 = torch.nn.Linear(32,2)   
        
    def forward(self,din):
              
        dout = F.relu(self.fc1(din))   
        dout = F.relu(self.fc2(dout))
        dout = F.relu(self.fc3(dout))
        dout = self.fc4(dout)
        
        return dout



loss_set=[]
acc_set=[]
def train():
    
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001, momentum=0.9)
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data,target in train_loader:
            optimizer.zero_grad()   
            output = model(data)
            target=target.long()
            loss = lossfunc(output,target)  
            loss.backward()         
            optimizer.step()        
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(X_train)
        loss_set.append(train_loss)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

        test()


def test():
    correct = 0
    total = 0
    with torch.no_grad():  
        for data,target in test_loader:
            outputs = model(data)
            values,predicted = torch.max(outputs.data, 1)
            # print(values)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc=100 * correct / total
            print('Accuracy of the network on the test dataset: %d %%' % (acc))
    acc_set.append(acc)
    return acc

def predict(states_all):
    labels=[]
    false_id=[]
    model1=torch.load('bpnn.pt')
    simulate_id=0
    for simulate_id,data in states_all:
        output = model1(data)
        values,predicted = torch.max(output.data, 1)
        if predicted==0:
            # true_labels += 1
            labels.append(0)
        else:
            labels.append(1)
            false_id.append(simulate_id)
        simulate_id+=1
    
    return labels,false_id

model = MLP()
# model.apply(weights_init)
n_epochs = 100   


train_data=np.load(pathX)
train_target=np.load(pathY)

X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.2)
print("train dataset size is "+ str(X_train.__len__()))
print("test dataset size is "+ str(X_test.__len__()))

traindataset=MyDataset(X_train,y_train)
train_loader = torch.utils.data.DataLoader(dataset=traindataset,
                                            batch_size=128,
                                            shuffle=True)

testdataset=MyDataset(X_test,y_test)
test_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                            batch_size=128,
                                            shuffle=True)


if __name__ == '__main__':
    train()
    torch.save(model, 'bpnn_Truth_36_6000.pt')
    # model=torch.load('./monkey-copter/bpnn.pt')
    print(loss_set)
    print(acc_set)
