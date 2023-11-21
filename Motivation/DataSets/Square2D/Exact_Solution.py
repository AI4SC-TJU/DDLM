import torch
import numpy as np

# exact solution
def u_Exact_Square2D(x, y):
    
    return torch.sin(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1)

# right hand side
def f_Exact_Square2D(x, y):
    
    return 4 * np.pi * np.pi * torch.sin(2*np.pi*x) * (2 * torch.cos(2*np.pi*y) - 1)  

# grad_x of exact solution
def Gradu_x_Exact_Square2D(x, y):

	return 2*np.pi * torch.cos(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1)  

# grad_y of exact solution	 
def Gradu_y_Exact_Square2D(x, y):

	return - 2*np.pi * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)     





# Dirichlet boundary condition (for verification of DtN map)
def g_Exact_Square2D(x, y):
    
    return torch.sin(2*np.pi*x) * (torch.cos(2*np.pi*y) - 1) 

# Robin boundary condition on the interface (for verification of weight imbalance)
def h_Exact_Square2D(x, y, alpha):

	return ( 2*np.pi * torch.cos(2*np.pi*x) + alpha * torch.sin(2*np.pi*x) ) * (torch.cos(2*np.pi*y) - 1) 


