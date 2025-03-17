import torch
import torch.nn as nn
import numpy as np
import shutil
import os



class PDEnet(nn.Module):
    def copy_all_txt_files(self,time):
        str = '{:10.3e}'.format(0.0)


        # dir_list = os.listdir(path)

    def __init__(self,time):
        super(PDEnet,self).__init__()
        self.copy_all_txt_files(time)


        W00 = np.loadtxt('W00.txt')
        self.N = N
        W01 = np.loadtxt('W01.txt')
        W = torch.Tensor([W00, W01])
        W1 = np.loadtxt('W1.txt')
        fc1 = nn.Linear(2,self.N)
        fc1.weight = torch.nn.Parameter(W.T)
        fc1.bias = torch.nn.Parameter(torch.zeros_like(fc1.bias))

        x = torch.ones((2))
        x = x.reshape(1,2)
        y = fc1(x)
        act_fc1 = torch.sigmoid(y)
        fc2 = nn.Linear(self.N, 1)
        fc2.weight = torch.nn.Parameter(torch.from_numpy(W1).reshape(1, self.N).float())
        fc2.bias = torch.nn.Parameter(torch.zeros((1)))
        ac2_torch = fc2(act_fc1.reshape(1, self.N))
        self.fc1 = fc1
        # self.act1 = torch.nn.Sigmoid
        #y = fc2(ac2_torch)
        self.fc2 = fc2
        qq = 0

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y



pde = PDEnet(0.0)
qq = 0