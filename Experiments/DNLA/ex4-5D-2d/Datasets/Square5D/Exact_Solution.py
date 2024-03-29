import torch
import numpy as np

# exact solution
def u_Exact_Square5D(x, dim_prob=5):
    temp = (torch.sin(2 * np.pi * x[:,0]))
    for i in range(1, dim_prob):
        temp += torch.sin(2 * np.pi * x[:,i])
    return temp

# right hand side
def f_Exact_Square5D(x, dim_prob=5):
    temp = (torch.sin(2 * np.pi * x[:,0]))
    for i in range(1, dim_prob):
        temp += torch.sin(2 * np.pi * x[:,i])
    temp = 4 * np.pi * np.pi * temp
    return temp  

# Dirichlet boundary condition
def g_Exact_Square5D(x, dim_prob=5):
    temp = (torch.sin(2 * np.pi * x[:,0]))
    for i in range(1, dim_prob):
        temp += torch.sin(2 * np.pi * x[:,i])
    return temp

# the index-th component of gradient
def Gradu_x_Exact_Square5D(x, index=0, dim_prob=5):
    temp = 2 * np.pi * (torch.cos(2 * np.pi * x[:,index]))
    temp = temp.reshape(-1,1)
    return temp
# gradient of exact solution
def Gradu_Exact_Square5D(x, dim_prob=5):
    dim_prob = x.size()[1]
    temp = Gradu_x_Exact_Square5D(x)
    for i in range(1, dim_prob):
        temp0 = Gradu_x_Exact_Square5D(x, i, dim_prob)
        temp = torch.cat([temp, temp0], dim = 1)

    return temp

