import torch
import numpy as np

# exact solution
def u_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        
        return torch.sin(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * x[:, 0] * x[:, 1] * (x[:, 1] - 1) * (x[:, 0] -1)
    else:
        return torch.sin(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * x[:, 0] * x[:, 1] * (x[:, 1] - 1) * (x[:, 0] -1)

# right hand side
def f_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        return 4 * alphain * np.pi * np.pi * torch.sin(2*np.pi*x[:,0]) * (2 * torch.cos(2*np.pi*x[:,1]) - 1) \
            - 20 * alphain *(x[:,0]- 1) * x[:,0] - 20 * alphain *(x[:,1]- 1) * x[:,1] 
    else:
        return 4 * alphaout * np.pi * np.pi * torch.sin(2*np.pi*x[:,0]) * (2 * torch.cos(2*np.pi*x[:,1]) - 1)\
            - 20 * alphaout *(x[:,0]- 1) * x[:,0] - 20 * alphaout *(x[:,1]- 1) * x[:,1] 

# Dirichlet boundary condition
def g_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        return torch.sin(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * x[:, 0] * x[:, 1] * (x[:, 1] - 1) * (x[:, 0] -1)
    else:
        return torch.sin(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * x[:, 0] * x[:, 1] * (x[:, 1] - 1) * (x[:, 0] -1)


# grad_x of exact solution
def Grad_u_Exact(x, alphain, alphaout, sub_dom):
    if sub_dom == 1:
        grad_u_x = 2*np.pi *  torch.cos(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * (x[:,1]-1)*x[:,1] *(2*x[:,0]-1)
        grad_u_y = -2*np.pi * torch.sin(2*np.pi*x[:,0]) * torch.sin(2*np.pi*x[:,1]) + 10 * (x[:,0]-1)*x[:,0] *(2*x[:,1]-1)
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
        return grad_temp  
    else:
        grad_u_x = 2*np.pi * torch.cos(2*np.pi*x[:,0]) * (torch.cos(2*np.pi*x[:,1]) - 1) + 10 * (x[:,1]-1)*x[:,1] *(2*x[:,0]-1)
        grad_u_y = -2*np.pi * torch.sin(2*np.pi*x[:,0]) * torch.sin(2*np.pi*x[:,1]) + 10 * (x[:,0]-1)*x[:,0] *(2*x[:,1]-1)
        grad_temp = torch.cat([grad_u_x.reshape(-1,1), grad_u_y.reshape(-1,1)], dim = 1)
        return grad_temp  
