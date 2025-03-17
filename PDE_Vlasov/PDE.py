import torch
import torch.nn as nn
import numpy as np
import shutil
import os



class PDEnet(nn.Module):
    def make_fn(self,base,time):
        ts = '{:10.3e}'.format(time)
        fn = base+'_'+ts+'.txt'
    def copy_all_txt_files(self,time,dir):
        ts = '{:10.3e}'.format(time)
        names = ['E','f','rho','x']
        fnames  = [self.make_fn(n) for n in names]

        for fn in fnames:
            shutil.copy(dir+'/'+fn,fn)
            qq = 0


        #dir_list = os.listdir(dir)
        qq = 0
    def read_physical_variables(self,E_fn,f_fn):
        self.E = np.loadtxt(E_fn)
        self.f = np.loadtxt(f_fn)
        N = self.E.shape[0]

        # for f - second Matlab index first - check !!!
        self.f = self.f.reshape(N,N)

    def __init__(self,time):
        super(PDEnet,self).__init__()
        self.copy_all_txt_files(time,'../Sonnendrucker')
        self.time = time
        E_fn = self.make_fn('E',self.time)
        f_fn = self.make_fn('f', self.time)
        self.read_physical_variables(E_fn,f_fn)


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