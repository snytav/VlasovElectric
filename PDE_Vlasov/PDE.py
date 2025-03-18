import torch
import torch.nn as nn
import numpy as np
import shutil
import os



class PDEnet(nn.Module):
    def make_fn(self,base,time):
        ts = '{:10.3e}'.format(time)
        fn = base+'_'+ts+'.txt'
        return fn
    def copy_all_txt_files(self,time,dir):
        ts = '{:10.3e}'.format(time)
        names = ['E','f','rho','x','v']
        fnames  = [self.make_fn(n,time) for n in names]

        for fn in fnames:
            shutil.copy(dir+'/'+fn,fn)
            qq = 0


        #dir_list = os.listdir(dir)
        qq = 0
    def read_physical_variables(self,E_fn,f_fn,x_fn,v_fn):
        self.E = np.loadtxt(E_fn)
        self.f = np.loadtxt(f_fn)
        N = self.E.shape[0]
        self.N = N
        self.v = np.loadtxt(v_fn)
        self.x = np.loadtxt(x_fn)
        dx = self.x[1] - self.x[0]
        dv = self.v[1] - self.v[0]
        self.dx = torch.tensor([dx,dv])

        # for f - second Matlab index first - check !!!
        self.f = self.f.reshape(N,N)
        qq = 0

    def __init__(self,time):
        super(PDEnet,self).__init__()
        self.copy_all_txt_files(time,'../Sonnendrucker')
        self.time = time
        E_fn = self.make_fn('E',self.time)
        f_fn = self.make_fn('f', self.time)
        x_fn = self.make_fn('x',self.time)
        v_fn = self.make_fn('v', self.time)
        self.read_physical_variables(E_fn,f_fn,x_fn,v_fn)
        fc1 = nn.Linear(2, self.N)
        fc2 = nn.Linear(self.N, 1)
        self.simplified = True




        x = torch.ones((2))
        x = x.reshape(1,2)
        y = fc1(x)
        act_fc1 = torch.sigmoid(y)
        #fc2 = nn.Linear(self.N, 1)
       # fc2.weight = torch.nn.Parameter(torch.from_numpy(W1).reshape(1, self.N).float())
       #  fc2.bias = torch.nn.Parameter(torch.zeros((1)))
        ac2_torch = fc2(act_fc1.reshape(1, self.N))
        self.fc1 = fc1
        # self.act1 = torch.nn.Sigmoid
        #y = fc2(ac2_torch)
        self.fc2 = fc2
        qq = 0

    def forward(self,x):

        if self.simplified:
            ic = torch.floor(torch.div(x,self.dx)).int()
            y = self.f[ic[0]][ic[1]]
            qq = 0
        else:
            x = x.reshape(1, 2)
            y = self.fc1(x)
            y = torch.sigmoid(y)
            y = self.fc2(y.reshape(1, self.N))
        return y



pde = PDEnet(0.0)
y = pde(0.1*torch.ones(2))
qq = 0