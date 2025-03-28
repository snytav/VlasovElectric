import torch
import torch.nn as nn
import numpy as np
import shutil
import os
import glob



class PDEnet(nn.Module):
    def make_fn(self,base,time):
        ts = '{:10.3e}'.format(time)
        fn = base+'_'+ts+'.txt'
        return fn
    def copy_all_txt_files(self,times,dir):
        for time in times:
            ts = '{:10.3e}'.format(time)
            names = ['E','f','rho','x','v']
            fnames  = [self.make_fn(n,time) for n in names]

            for fn in fnames:
                shutil.copy(dir+'/'+fn,fn)
            qq = 0


        #dir_list = os.listdir(dir)
        qq = 0

    def get_time_moments(self,dir):
        cur_dir  = os.getcwd()
        os.chdir(dir)
        e_files = glob.glob('E*.txt')
        times = [n.split('_')[1].split('.txt')[0] for n in e_files]
        f_times = [float(s) for s in times]
        f_times = torch.tensor(f_times)
        sorted, indices = torch.sort(f_times)
        os.chdir(cur_dir)
        return sorted


    def read_physical_variables(self,E_fn,f_fn,x_fn,v_fn):
        E = np.loadtxt(E_fn)
        f = np.loadtxt(f_fn)
        N = E.shape[0]
        self.N = N
        f = f.reshape(N,N)
        v = np.loadtxt(v_fn)
        x = np.loadtxt(x_fn)
        dt = self.times[1] - self.times[0]
        dx = x[1] - x[0]
        dv = v[1] - v[0]
        self.dx = torch.tensor([dt,dx,dv])
        x = torch.from_numpy(x)
        v = torch.from_numpy(v)
        f = torch.from_numpy(f)
        E = torch.from_numpy(E)


        # for f - second Matlab index first - check !!!
        #f = f.reshape(N,N)
        qq = 0
        return E,f,x,v

    def fill_all_moments_f_v_E(self,times):
        nt = times.shape[0]
        self.f_sonn = torch.zeros(nt,self.N,self.N)
        self.v_sonn = torch.zeros(nt, self.N)
        self.E_sonn = torch.zeros(nt, self.N)

        for i,t in enumerate(times):
            E_fn = self.make_fn('E', t)
            f_fn = self.make_fn('f', t)
            x_fn = self.make_fn('x',t)
            v_fn = self.make_fn('v', t)
            E,f,x,v = self.read_physical_variables(E_fn, f_fn, x_fn, v_fn)
            self.f_sonn[i,:,:] = f
            self.E_sonn[i,:]   = E
            self.v_sonn[i,:]   = v


    def __init__(self,time):
        super(PDEnet,self).__init__()
        self.times = self.get_time_moments('../Sonnendrucker')
        self.copy_all_txt_files(self.times,'../Sonnendrucker')
        # self.time = 0.0
        self.E = []
        # self.x = []
        # self.v = []
        self.f = []

        for time in self.times:
            E_fn = self.make_fn('E',time)
            f_fn = self.make_fn('f',time)
            x_fn = self.make_fn('x',time)
            v_fn = self.make_fn('v',time)

            E,f,x,v = self.read_physical_variables(E_fn,f_fn,x_fn,v_fn)
            self.E.append(E)
            self.f.append(f)
            self.x = x
            self.v = v

        # self.f = torch.tensor(self.f)
        # self.E = torch.tensor(self.E)


        self.Lx = torch.max(self.x)
        self.Lv = torch.max(self.v)
        self.N  = self.v.shape[0]
        fc1 = nn.Linear(2, self.N)
        fc2 = nn.Linear(self.N, 1)
        self.x0 = torch.tensor([torch.min(self.times),torch.min(self.x),torch.min(self.v)])
        self.simplified = True

        self.fill_all_moments_f_v_E(self.times)




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
        qq = 0

        if self.simplified:
            ic = torch.floor(torch.div(x-self.x0,self.dx)).int()
            y = self.f_sonn[ic[0]][ic[1]][ic[2]]
            qq = 0
        else:
            x = x.reshape(1, 2)
            y = self.fc1(x)
            y = torch.sigmoid(y)
            y = self.fc2(y.reshape(1, self.N))
        return y

    # def A(x):
    #     return x[1] * np.sin(np.pi * x[0])

    def A(self,x):
        ic = torch.floor(torch.div(x, self.dx)).int()
        if lc[0] == self.N or lc[0] == 0:
           return self.f[lc[0]][lc[1]]
        else:
            return 0.0

        if lc[1] == self.N or lc[1] == 0:
           return self.f[lc[0]][lc[1]]
        else:
            return 0.0

        return ()

    def psy_trial(self,x):
        return self.A(x) + x[0] * (self.Lx - x[0]) * x[1] * (self.Lv - x[1]) * self.forward(x)

    def f(x):
        return 0.

pde = PDEnet(0.0)
from loss import loss_function1
lf = loss_function1(pde.times,pde.x,pde.v,pde,pde.forward,pde.f_sonn,pde.v_sonn,pde.E_sonn)

y = pde(torch.tensor([5*1.39,8*1.39]))
qq = 0