import torch
import math
import numpy as np
import scipy.io as io
##############################################################################################

#set the parameters of the curve
a = 0.50
b = 0.25
m = 0
n = 12

def radius(x):
    theta = torch.atan(x[1]/x[0])
    radius = a + b * torch.cos(m * theta) * torch.sin(n * theta)
    return radius

def SmpPts_Interior(num_intrr_pts, sub_dom, dim_prob=2): 
    temp = torch.rand([num_intrr_pts, dim_prob])
    
    if sub_dom == 1:
        """ num_intrr_pts = total number of sampling points inside the domain
            dim_prob  = dimension of sampling domain """
        '''
        # subdomain 1 interior of the flower 
        r = a + b * cos(m * theta) * sin(n * theta)
        theta = 2* np.pi * torch.rand(num_intrr_pts,1)            
        X = r * torch.cos(theta).reshape(-1,1)
        Y = r * torch.sin(theta).reshape(-1,1)
        rho = torch.cat([X,Y], dim=1)

        return rho
        '''
        i = 0
        while i < num_intrr_pts:
            if torch.norm(temp[i]) < radius(temp[i]):
                i = i + 1
            else:
                temp[i] = torch.rand([1, dim_prob])
                

    if sub_dom==2:
        # outside the flower but in the square [-1,1] * [-1,1]
        i = 0
        while i < num_intrr_pts:
            if torch.norm(temp[i]) > radius(temp[i]):
                i = i + 1
            else:
                temp[i] = torch.rand([1, dim_prob])
                
    temp[:, 0] = temp[:, 0] + 0.01    
    return temp

def Sample_intrr_plus(num_intrr_pts, dim_prob):
    temp = torch.rand([num_intrr_pts, dim_prob])
    i = 0
    while i < num_intrr_pts:
        if (torch.norm(temp[i]) > radius(temp[i])) and (a > torch.norm(temp[i])):
            i = i + 1
        else:
            temp[i] = torch.rand([1, dim_prob])
                
    temp[:, 0] = temp[:, 0] + 0.01    
    return temp    




# draw sample points uniformly at random from the Dirichlet boundaries (not including Gamma)
def SmpPts_Boundary(num_bndry_pts, sub_dom,dim_prob=2):
    if sub_dom==1:
        """ num_bndry_pts = total number of sampling points at each boundary
            dim_prob  = dimension of sampling domain """ 
        X_bottom_l = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2 , torch.zeros(num_bndry_pts, 1)], dim=1)
        X_left_d  = torch.cat([torch.zeros(num_bndry_pts, dim_prob - 1), torch.rand(num_bndry_pts, dim_prob-1)/2], dim=1) 
        temp = torch.cat((X_bottom_l, X_left_d), dim=0)

 
    if sub_dom==2:
        """ num_bndry_pts = total number of sampling points at each boundary
            dim_prob  = dimension of sampling domain """ 
        
        temp1 = torch.ones(num_bndry_pts, 1)
        #  the random sample from the interval [-1,1]
        tempr =  torch.rand(num_bndry_pts, dim_prob-1)
        # boundary of domain (-1,1) * (-1,1), not including Gamma x^2 + y^2 = 0.25          
        X_right = torch.cat([temp1, tempr], dim=1)
        X_bottom_r = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2 + 0.5 , torch.zeros(num_bndry_pts, 1)], dim=1)
        X_top = torch.cat([tempr, temp1], dim=1)
        X_left_u = torch.cat([torch.zeros(num_bndry_pts, dim_prob - 1), torch.rand(num_bndry_pts, dim_prob-1)/2 + 0.5], dim=1)
        temp = torch.cat((X_right, X_left_u, X_bottom_r, X_top), dim=0) 
    temp[:, 0] = temp[:, 0] + 0.01
    return temp 
# draw sample points uniformly at random from Gamma {0.5} * (0,1)
def SmpPts_Interface(num_bndry_pts, dim_prob=2):    
    """ num_bndry_pts = total number of boundary sampling points at Gamma
        dim_prob  = dimension of sampling domain """

    theta =  math.pi/2 * torch.rand(num_bndry_pts,1)            
    X = (a + b * torch.cos(m * theta) * torch.sin(n * theta) ) * torch.cos(theta).reshape(-1,1) + 0.01
    Y = (a + b * torch.cos(m * theta) * torch.sin(n * theta) ) * torch.sin(theta).reshape(-1,1)
    rho = torch.cat([X,Y], dim=1)

    return rho

# draw equidistant sample points from the subdomain [0,1/2] * [0,1] and subdomain [1/2,1]*[0,1]
def SmpPts_Test(num_test_pts,sub_dom):
    """ num_test_pts = total number of sampling points for model testing """
    temp = torch.empty([1,2])
    #return SmpPts_Interior(num_test_pts * num_test_pts, sub_dom, dim_prob=2)
    if sub_dom == 1:
        mesh_in = io.loadmat("flower-quarter.mat")
        x = torch.from_numpy(mesh_in["node"][0,:]).float().reshape(-1,1)
        y = torch.from_numpy(mesh_in["node"][1,:]).float().reshape(-1,1)  
        temp = torch.hstack([x, y])

    if sub_dom == 2:
        mesh_out = io.loadmat("flower-quarter-out.mat")
        x = torch.from_numpy(mesh_out["node"][0,:]).float().reshape(-1,1)
        y = torch.from_numpy(mesh_out["node"][1,:]).float().reshape(-1,1)  
        temp = torch.hstack([x, y])
    return temp
def arclength(num_bndry_pts, dim_prob=2):
    #return the arclength of curve with flower shape
    theta =  math.pi/2 * torch.rand(num_bndry_pts,1) 
    DX = 3 * torch.cos(12 * theta) * torch.cos(theta) - torch.sin(theta) * (torch.sin(12 * theta)/4 + 1/2)
    DY = torch.cos(theta) * (torch.sin(12 * theta)/4 + 1/2) + 3 * torch.cos(12 * theta) * torch.sin(theta)
    rho = torch.cat([DX,DY], dim=1)
    return torch.norm(rho, dim = 1)
##############################################################################################