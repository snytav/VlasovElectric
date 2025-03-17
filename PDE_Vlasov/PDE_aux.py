import torch
import numpy as np

from convection_basic import linear_convection_solve

c = 1.0
Lx = 1.0

nx = 40
ny = 5
Lt = ny*0.025


u,u2D = linear_convection_solve(c,Lx,nx+1,Lt,ny)

dx = Lx / nx
dy = Lt / ny
x_space = np.linspace(0, Lx, nx)
y_space = np.linspace(0, Lt, ny)


# def analytic_solution(x):
#     ix = int(np.where(x_space == x[0])[0])
#     iy = int(np.where(y_space == x[1])[0])
#     ansol = u2D[iy][ix]
#     # if not isinstance(t,np.float64):
#     #     qq = 0
#     return ansol








