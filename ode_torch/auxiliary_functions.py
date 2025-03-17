import torch
from torch.autograd.functional import jacobian

from equation import LEFT,RIGHT

def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return RIGHT(x) - psy * LEFT(x)

def loss_function(pde, x):
    loss_sum = 0.
    for xi in x:
        xi = xi.reshape(1)
        input_point = xi

        # net_out = neural_network(W, xi)[0][0]
        net_out = pde.forward(xi)
        net_out_jacobian = jacobian(pde.forward, input_point)
        psy_t = 1. + xi * net_out
        d_net_out = net_out_jacobian.item()

        # d_net_out = d_net_out[0]
        # d_net_out = d_neural_network_dx(pde, xi)
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)
        err_sqr = (d_psy_t - func)**2

        loss_sum += err_sqr
    return loss_sum
