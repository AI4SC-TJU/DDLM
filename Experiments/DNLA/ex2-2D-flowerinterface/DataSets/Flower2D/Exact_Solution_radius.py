import torch
import numpy as np

a = 0.500
b = 0.250
m = 0
n = 12

def radius(x):

    theta = torch.atan(x[:,1]/x[:,0])
    radius = a + b * torch.cos(m * theta) * torch.sin(n * theta)
    return radius

def u_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        temp = torch.norm(x, dim = 1) * torch.norm(x, dim = 1) * (torch.norm(x, dim = 1) - radius(x)) + 1
    else:
        temp = torch.norm(x, dim = 1) * torch.norm(x, dim = 1) * (torch.norm(x, dim = 1) - radius(x)) + 1 
    return temp
    

# right hand side
def f_Exact(x, alphain, alphaout, sub_dom):

    if sub_dom == 1:
        temp = -35 * torch.sin(12 * torch.atan(x[:, 1]/x[:, 0])) + 2 - 9 * torch.norm(x, dim = 1)      
        temp = temp * alphain
    else:
        temp = -35 * torch.sin(12 * torch.atan(x[:, 1]/x[:, 0])) + 2 - 9 * torch.norm(x, dim = 1)  
        temp = temp * alphaout
    return temp

# Dirichlet boundary condition
def g_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        temp = torch.norm(x, dim = 1) * torch.norm(x, dim = 1) * (torch.norm(x, dim = 1) - radius(x)) + 1
    else:
        temp = torch.norm(x, dim = 1) * torch.norm(x, dim = 1) * (torch.norm(x, dim = 1) - radius(x)) + 1
    return temp


# grad_x of exact solution
def Grad_u_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        grad_u_x = 3 * x[:, 0] * torch.norm(x, dim = 1) - x[:, 0] + 3 * x[:, 1] * torch.cos(12 * torch.atan(x[:, 1]/x[:, 0])) - x[:, 0] * torch.sin(12 * torch.atan(x[: ,1]/x[:, 0]))/2      
        grad_u_y = 3 * x[:, 1] * torch.norm(x, dim = 1) - x[:, 1] - 3 * x[:, 0] * torch.cos(12 * torch.atan(x[:, 1]/x[:, 0])) - x[:, 1] * torch.sin(12 * torch.atan(x[: ,1]/x[:, 0]))/2      
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
        
    else:
        grad_u_x = 3 * x[:, 0] * torch.norm(x, dim = 1) - x[:, 0] + 3 * x[:, 1] * torch.cos(12 * torch.atan(x[:, 1]/x[:, 0])) - x[:, 0] * torch.sin(12 * torch.atan(x[: ,1]/x[:, 0]))/2      
        grad_u_y = 3 * x[:, 1] * torch.norm(x, dim = 1) - x[:, 1] - 3 * x[:, 0] * torch.cos(12 * torch.atan(x[:, 1]/x[:, 0])) - x[:, 1] * torch.sin(12 * torch.atan(x[: ,1]/x[:, 0]))/2      
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
    return grad_temp 

