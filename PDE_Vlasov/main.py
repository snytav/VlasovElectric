import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import numpy as np
import torch
from visual_PDE import plot_3Dsurface
from PDE_loss import loss_function
# from PDE_aux import  f,analytic_solution
from PDE import PDEnet
from visual_PDE import get_analytic_and_trial_solution_2D
from convection_basic import linear_convection_solve
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

c = 1.0
Lx = 1.0

nx = 40
ny = 5
Lt = ny*0.025

u,u2D = linear_convection_solve(c,Lx,nx+1,Lt,ny)
u2D = torch.from_numpy(u2D)

dx = Lx / nx
dy = Lt / ny
x_space = torch.linspace(0, Lx, nx)
y_space = torch.linspace(0, Lt, ny)

def f(x):
    return 0.

def analytic_solution(x):
    ix = int(torch.where(x_space == x[0])[0])
    iy = int(torch.where(y_space == x[1])[0])
    ansol = u2D[iy][ix]
    # if not isinstance(t,np.float64):
    #     qq = 0
    return ansol

def A(x):
    return analytic_solution(x)

def psy_trial(x, net_out):
    return A(x) + x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) * net_out

def psy_trial1(x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) * net_out

def psy_trial2(x, net_out):
    return x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) #* net_out

pde = PDEnet(10)

xt = torch.tensor([x_space[3],y_space[3]])
yt = pde(xt)

def loss_function1(pde, x, y):

    #loss_basic = np.loadtxt('penta_loss_arr.txt')
    # dpsy_dx = loss_basic[:, 2]
    # dpsy_dy = loss_basic[:, 3]
    loss_sum = 0.

    loss_list = []
    for xi in x:
        for yi in y:

            input_point = torch.tensor([xi, yi])

            net_out = pde.forward(input_point)[0]

            net_out_jacobian = jacobian(pde.forward,input_point,create_graph=True)
            net_out_hessian = hessian(pde.forward,input_point,create_graph=True)

            psy_t = psy_trial(input_point, pde.forward(input_point))
            psy_t_jacobian = jacobian(psy_trial,inputs=(input_point,pde.forward(input_point)), create_graph=True)
            psy_t_jacobian1 = jacobian(psy_trial1,inputs=(input_point,pde.forward(input_point)), create_graph=True)
            psy_t_jacobian2 = jacobian(psy_trial2,inputs=(input_point,pde.forward(input_point)), create_graph=True)
            psy_t_hessian  = hessian(psy_trial,inputs=(input_point,pde.forward(input_point)), create_graph=True)
            #psy_t_hessian = hessian(psy_trial1, inputs=(input_point, pde.forward(input_point)), create_graph=True)

            gradient_of_trial_dx = psy_t_jacobian[0][0][0][0]
            n = len(loss_list)
            if np.abs(dpsy_dx[n]-gradient_of_trial_dx.item()) > 1e-3:
                qq = 0
            gradient_of_trial_dy = psy_t_jacobian[0][0][0][1]
            if np.abs(dpsy_dy[n]-gradient_of_trial_dy.item()) > 1e-3:
                qq = 0

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]


            func = f(input_point) # right part function

            err_sqr = ((gradient_of_trial_dx + gradient_of_trial_dy) - func)**2
            loss_sum += err_sqr
            print(xi.item(),yi.item(),err_sqr.item(), loss_sum.item())
            loss_list.append([xi.item(),yi.item(),gradient_of_trial_dx.item(),gradient_of_trial_dy.item(),err_sqr.item(), loss_sum.item()])
        print(xi.item(), loss_sum.item())

    loss_arr = np.array(loss_list)
    np.savetxt('loss_arr.txt', loss_arr,fmt='%15.5e')
    return loss_sum


# from PDE_aux import A,psy_trial

x = torch.tensor([x_space[-1],y_space[-1]])
y = A(x)

x = torch.tensor([x_space[3],y_space[3]])
net_out = pde.forward(x)
y = psy_trial(x,net_out)
qq = 0

# psy_trial ?

#loss ?

loss = loss_function1(pde, x_space, y_space)

qq = 0
lmb = 0.01
optimizer = torch.optim.Adam(pde.parameters(), lr=lmb)
loss = 1e6*torch.ones(1)
i = 0

#initial condition
surface,surface2 = get_analytic_and_trial_solution_2D(x_space,y_space,analytic_solution,psy_trial,pde)

while loss.item() > 1e2: #for i in range(100):
    optimizer.zero_grad()
    loss = loss_function(x_space, y_space,pde,psy_trial,f)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(i,loss.item())
    i = i + 1

surface,surface2 = get_analytic_and_trial_solution_2D(x_space,y_space,analytic_solution,psy_trial,pde)

plot_3Dsurface(x_space,y_space,surface,'X','Y','Finite-Difference solution')
plot_3Dsurface(x_space,y_space,surface2,'X','Y','Neural network solution')
qq = 0


