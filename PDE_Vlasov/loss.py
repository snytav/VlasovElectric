import torch

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import shutil

from jacobian_py import get_jacobian

from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
from torch.autograd import grad
from torch.autograd import Variable





def loss_function1(t,x, y,pde,psy_trial,f,v,E):
    loss_sum = 0.
    rhs = torch.zeros_like(f)

    for i,ti in enumerate(t):
          for j,xi in enumerate(x):
              for k,yi in enumerate(y):
                  input_point = torch.tensor([ti,xi, yi])
                  net_out = pde.forward(input_point)[0]
                  inputs = (input_point, net_out)
                  net_out_jacobian = jacobian(pde.forward, input_point, create_graph=True)
                  psy_t = psy_trial(input_point, net_out)
                  inputs = (input_point, net_out)

                  psy_t_jacobian = jacobian(psy_trial, inputs, create_graph=True)[0]
                  gradient_of_trial_dt = psy_t_jacobian[0]
                  gradient_of_trial_dx = psy_t_jacobian[1]
                  gradient_of_trial_dv = psy_t_jacobian[2]

                  func = 0.0 #f(input_point) # right part function

                  rhs[i][j][k] = ((gradient_of_trial_dt - v[k]*gradient_of_trial_dx
                                             - E[j]*gradient_of_trial_dv) - func)**2


    return rhs

