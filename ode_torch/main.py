import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt

from ODE import ODEnet
results = []
plt.figure()
plt.show()
eqname = '1_p_cos_05x_4pi'  #05x_p1'
for nx in [10,20,40]:
    names = ['x_space','nx','neural_solution','y_space','mae','mape']
    d = dict.fromkeys(names)
    x_space = np.linspace(0, 4*np.pi, nx)
    x_space = torch.from_numpy(x_space).to(torch.float)
    x_space.requires_grad = True
    np.savetxt(eqname+'x_space_'+str(nx)+'.txt',x_space.detach().numpy())

    d['nx'] = nx

    d['x_space'] = x_space

    pde = ODEnet(nx)
    x = torch.zeros(nx)
    y = pde.forward(x_space[0].reshape(1))

    from auxiliary_functions import loss_function
    lf = loss_function(pde,x_space)

    from train import trainODE
    pde = trainODE(pde,0.001,1e-1,x_space)

    neural_solution = [xi * pde.forward(xi.reshape(1)) for xi in x_space]
    neural_solution = torch.tensor(neural_solution)
    d['neural_solution'] = neural_solution #res.append(neural_solution)
    np.savetxt(eqname + 'neural_' + str(nx) + '.txt',x_space.detach().numpy())


    from equation import psy_analytic

    y_space = psy_analytic(x_space.detach().numpy())
    d['y_space']= y_space
    d['function'] = eqname
    np.savetxt(eqname + '_analytic_' + str(nx) + '.txt',x_space.detach().numpy())

    plt.plot(x_space.detach().numpy(), neural_solution.detach().numpy(),'o',label='neural '+ 'nx = '+str(nx))

    #plt.plot(x_space.detach().numpy(), neural_solution.detach().numpy(), 'o', label='neural ' + 'nx = ' + str(nx))
    plt.plot(x_space.detach().numpy(), y_space, label='analytic',color='red')
    plt.legend()
    plt.savefig(eqname + '_analytic_' + str(nx) + '.png')

    from sklearn.metrics import  mean_absolute_error,mean_absolute_percentage_error
    mae  = mean_absolute_error(y_space,neural_solution)
    mape = mean_absolute_percentage_error(y_space, neural_solution)
    d['mae'] = mae
    d['mape'] = mape

    results.append(d)
plt.plot(x_space.detach().numpy(), y_space, label='analytic')
plt.legend()
plt.title('Right hand side = '+d['function'])
plt.show()
plt.savefig(eqname+'_convergency.png')
nxx   = [r['nx'] for r in results]
maes  = [r['mae'] for r in results]
mapes = [r['mape'] for r in results]
import pandas as pd
df2 = pd.DataFrame(np.array([nxx,maes,mapes]).T,
                   columns=['nx', 'mae', 'mape'])
df2['nx'].astype('int32')
df2.to_csv(eqname+'_errors.csv')

qq = 0


qq = 0




# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
