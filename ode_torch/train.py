
import torch
from torch.optim import SGD
from auxiliary_functions import loss_function

def trainODE(pde,lrate,eps,x_space):

    optimizer = SGD(pde.parameters(),lr=lrate)
    loss = 1e6*torch.ones(1)
    n = 0
    while loss.item() > eps:
        optimizer.zero_grad()
        loss = loss_function(pde,x_space)
        loss.backward()
        print(n,loss.item())
        n = n + 1
        optimizer.step()

    return pde