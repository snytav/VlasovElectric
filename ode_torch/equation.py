import numpy as np
import torch


def LEFT(x):
    '''
        Left part of initial equation
    '''
    return 0.0 #x + (1. + 3.*x**2) / (1. + x + x**3)


def RIGHT(x):
    '''
        Right part of initial equation
    '''
    return 1.0+ torch.cos(x*0.5) #1.0 + 0.05*torch.cos(0.5*x)              #x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return x + 2*np.sin(x*0.5)  #x + 0.1*np.sin(0.5*x) # (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2
