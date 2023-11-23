import torch
##############################################################################################
# draw sample points uniformly at random inside the domain
def SmpPts_Interior_Square5D(num_intrr_pts, sub_dom, dim_prob = 5):    
    if sub_dom == 1:
        """ num_intrr_pts = total number of sampling points inside the domain
            dim_prob  = dimension of sampling domain """

        # subdomain 1 x_1 in (0,1/2) , x_2 ... x_n in (0,1)                
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0] / 2
        if X_d.min() == 0:
            print('Error! Boundary nodes in the interior domain!')

        return X_d
    if sub_dom==2:
        # subdomain2 x_1 in (1/2,1) , x_2 ... x_n in (0,1)                 
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0] * 0.5 + 0.5
        
        if X_d.min() == 0:
            print('Error! Boundary nodes in the interior domain!')

        return X_d        

# draw sample points uniformly at random from the Dirichlet boundaries (not including Interface)
def SmpPts_Boundary_Square5D(num_bndry_pts, sub_dom, dim_prob = 5):
    if sub_dom==1:
        """ num_bndry_pts = total number of sampling points at each boundary
            dim_prob  = dimension of sampling domain """ 
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)

        X_1 = torch.cat([temp0,torch.rand(num_bndry_pts, dim_prob-1)], dim = 1)

        X_bndry = X_1
        for i in range(2, dim_prob+1):
            X_temp_1 = torch.cat([torch.rand(num_bndry_pts, 1) / 2, torch.rand(num_bndry_pts, i-2), temp0, torch.rand(num_bndry_pts, dim_prob-i)], dim = 1)
            X_temp_2 = torch.cat([torch.rand(num_bndry_pts, 1) / 2, torch.rand(num_bndry_pts, i-2), temp1, torch.rand(num_bndry_pts, dim_prob-i)], dim = 1)
            X_temp = torch.cat([X_temp_1, X_temp_2], dim = 0)
            X_bndry = torch.cat([X_bndry, X_temp], dim = 0)
        
        return X_bndry 
    if sub_dom==2:
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)

        X_1 = torch.cat([temp1,torch.rand(num_bndry_pts, dim_prob-1)], dim = 1)
        X_bndry = X_1
        for i in range(2, dim_prob+1):
            X_temp_1 = torch.cat([torch.rand(num_bndry_pts, 1) / 2 + 0.5, torch.rand(num_bndry_pts, i-2), temp0, torch.rand(num_bndry_pts, dim_prob-i)], dim = 1)
            X_temp_2 = torch.cat([torch.rand(num_bndry_pts, 1) / 2 + 0.5, torch.rand(num_bndry_pts, i-2), temp1, torch.rand(num_bndry_pts, dim_prob-i)], dim = 1)
            X_temp = torch.cat([X_temp_1, X_temp_2], dim = 0)
            X_bndry = torch.cat([X_bndry, X_temp], dim = 0)
        
        return X_bndry 
# draw sample points uniformly at random from the Interface x_1=1/2, x_2... x_n in (0,1)
def SmpPts_Interface_Square5D(num_bndry_pts, dim_prob = 5):    
    """ num_bndry_pts = total number of boundary sampling points at Gamma
        dim_prob  = dimension of sampling domain """

    temp0 = torch.ones(num_bndry_pts, 1) / 2 
    temp1=torch.rand(num_bndry_pts, dim_prob-1)
    X_gam=torch.cat([temp0, temp1],dim=1)
    return X_gam
# draw equidistant sample points from the subdomain [0,1/2] * [0,1] and subdomain [1/2,1]*[0,1]
def SmpPts_Test_Square5D(num_test_pts,sub_dom, dim_prob=10):
    """ num_test_pts = total number of sampling points for model testing """
    # subdomain 1 x_1 in (0,1/2) , x_2 ... x_n in (0,1)  
    if sub_dom==1:
        xs = torch.linspace(0, 1/2, steps=num_test_pts)
        ys = torch.linspace(0, 1, steps=num_test_pts)
        x1, x2, x3, x4, x5  = torch.meshgrid(xs, ys, ys, ys, ys)
        temp = torch.stack([x1.reshape(1,num_test_pts**5), x2.reshape(1,num_test_pts**5), x3.reshape(1,num_test_pts**5), x4.reshape(1,num_test_pts**5), x5.reshape(1,num_test_pts**5)], dim=-1)
        return torch.squeeze(temp)                
    # subdomain 1 x_1 in (1/2,1) , x_2 ... x_n in (0,1)  
    if sub_dom==2:
        xs = torch.linspace(1/2, 1, steps=num_test_pts)
        ys = torch.linspace(0, 1, steps=num_test_pts)
        x1, x2, x3, x4, x5  = torch.meshgrid(xs, ys, ys, ys, ys)
        temp = torch.stack([x1.reshape(1,num_test_pts**5), x2.reshape(1,num_test_pts**5), x3.reshape(1,num_test_pts**5), x4.reshape(1,num_test_pts**5), x5.reshape(1,num_test_pts**5)], dim=-1)
        return torch.squeeze(temp)     
     

##############################################################################################