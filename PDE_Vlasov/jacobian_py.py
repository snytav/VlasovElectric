import torch

def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    # x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data