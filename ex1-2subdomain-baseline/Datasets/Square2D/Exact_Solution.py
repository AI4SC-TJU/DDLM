import torch
import numpy as np

# exact solution
def u_Exact_Square2D(x, y):
    
    return 10*x*(x-1)*y*(y-1)
# right hand side
def f_Exact_Square2D(x, y):
    
    return -2*10*x*(x-1)-20*y*(y-1)

# Dirichlet boundary condition
def g_Exact_Square2D(x, y):
    
    return 10*x*(x-1)*y*(y-1)

# Boundary condition on Gamma
def h_Exact_Square2D(x, y, alpha):

	# Robin boundary condition
	return 10*alpha*x*(x-1)*y*(y-1)+10*(2*x-1)*y*(y-1)

# grad_x of exact solution
def Gradu_x_Exact_Square2D(x, y):

	return 10*(2*x-1)*y*(y-1)

# grad_y of exact solution	 
def Gradu_y_Exact_Square2D(x, y):

	return 10*(2*y-1)*x*(x-1)