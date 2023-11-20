import torch
import numpy as np

# exact solution
def u_Exact(x, alphain, alphaout, sub_dom):
    
    if sub_dom in ['B', 2 ,4, '1E2','1E4', '3E2','3E4']:
        return  torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphaout
    if sub_dom in ['R', 1 ,3, '2E1','2E3', '4E1','4E3']:
        return  torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphain
    
# right hand side
def f_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom in ['B', 2 ,4, '1E2','1E4', '3E2','3E4']:
        return  32  * np.pi * np.pi * torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])
    if sub_dom in ['R', 1 ,3, '2E1','2E3', '4E1','4E3']:
        return  32  * np.pi * np.pi * torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])
# Dirichlet boundary condition
def g_Exact(x, alphain, alphaout, sub_dom):

    if sub_dom in ['B', 2 ,4, '1E2','1E4', '3E2','3E4']:
        return  torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphaout
    if sub_dom in ['R', 1 ,3, '2E1','2E3', '4E1','4E3']:
        return  torch.sin(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphain
# The initial guess of u
def u_0(x, alphain, alphaout, sub_dom):

    return 10**0 * torch.cos(100*np.pi*x[:,0]) * torch.cos(100*np.pi*x[:,1])
# grad_x of exact solution
def Grad_u_Exact(x, alphain, alphaout, sub_dom):
    
    if sub_dom in ['B', 2 ,4, '1E2','1E4', '3E2','3E4']:
        grad_u_x = 10**0 * 4*np.pi *  torch.cos(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphaout
        grad_u_y = 10**0 * 4*np.pi *  torch.sin(4*np.pi*x[:,0]) * torch.cos(4*np.pi*x[:,1])/alphaout
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
        return grad_temp  
    if sub_dom in ['R', 1 ,3, '2E1','2E3', '4E1','4E3']:
        grad_u_x = 10**0 * 4*np.pi *  torch.cos(4*np.pi*x[:,0]) * torch.sin(4*np.pi*x[:,1])/alphain
        grad_u_y = 10**0 * 4*np.pi *  torch.sin(4*np.pi*x[:,0]) * torch.cos(4*np.pi*x[:,1])/alphain
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
        return grad_temp  
    
