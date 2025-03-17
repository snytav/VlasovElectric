import torch

def optimizer_step(pde,lmb):
    DL_DW1 = pde.fc1.weight.grad
    DL_DW2 = pde.fc2.weight.grad

    pde.fc1.weight = torch.nn.Parameter(pde.fc1.weight - DL_DW1 * lmb)
    pde.fc2.weight = torch.nn.Parameter(pde.fc2.weight - DL_DW2 * lmb)