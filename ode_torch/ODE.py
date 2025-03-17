import torch
import torch.nn as nn
import numpy as np

class ODEnet(nn.Module):
    def __init__(self,N):
        super(ODEnet,self).__init__()
        W00 = np.loadtxt('W00.txt')
        W00 = torch.from_numpy(W00).to(torch.float)
        self.N = N

        fc1 = nn.Linear(1,self.N)
        y_test = fc1(torch.zeros(1))
        W00 = W00.reshape(10,1)
        # W00 = torch.from_numpy(W00)
        # fc1.weight = torch.nn.Parameter(W00)

        y_test = fc1(torch.zeros(1))
        y_test = fc1(torch.zeros(1))
        # fc1.bias = torch.nn.Parameter(torch.zeros_like(fc1.bias))
        self.fc1 = fc1
        x = torch.zeros(1)
        # x = x.to(torch.double)
        y = self.fc1(x)
        act_fc1 = torch.sigmoid
        y = act_fc1(y)
        self.act_fc1 = act_fc1

        W01 = np.loadtxt('W01.txt')
        W01 = torch.from_numpy(W01).to(torch.float)

        fc2 = nn.Linear(self.N, 1)

        # fc2.weight = torch.nn.Parameter(W01.reshape(1, self.N).float())
        # fc2.bias = torch.nn.Parameter(torch.zeros((1)))
        act_fc2 = torch.sigmoid #2_torch = fc2(act_fc1.reshape(1, self.N))
        y= fc2(y)
        self.fc2 = fc2
        # self.fc1 = fc1
        # self.act1 = torch.nn.Sigmoid
        self.act_fc2 = act_fc2
        qq = 0

    def forward(self,x):
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.fc2(x)
        # x = self.act_fc2(x)
        return x
