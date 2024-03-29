import torch
##############################################################################################
# draw sample points uniformly at random inside the domain
def SmpPts_Interior_Square2D(num_intrr_pts, sub_dom, dim_prob=2):  
    """ sub_dom = 1 refers to the Dirichlet subproblem
        sub_dom = 2 refers to the Neumann subproblem
        sub_dom = 0 refers to the entire domain

        num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain """

    if sub_dom == 1:
        # subdomain (0,1/2) * (0,1)                
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0]/2
        if X_d.min() == 0:
            print('Error! Boundary nodes in the interior domain!')

        return X_d

    if sub_dom == 2:
        # subdomain (1/2,1) * (0,1)                
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0]*0.5+0.5
        
        if X_d.min() == 0.5:
            print('Error! Boundary nodes in the interior domain!')

        return X_d 

    if sub_dom == 0:
        # entire domain (0,1) * (0,1)                
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        
        if X_d.min() == 0:
            print('Error! Boundary nodes in the interior domain!')

        return X_d     

# draw sample points uniformly at random from the Dirichlet boundaries (not including the interface)
def SmpPts_Boundary_Square2D(num_bndry_pts, sub_dom,dim_prob=2):
    """ sub_dom = 1 refers to the Dirichlet subproblem
        sub_dom = 2 refers to the Neumann subproblem

        num_bndry_pts = total number of sampling points at each boundary
        dim_prob  = dimension of sampling domain """

    if sub_dom == 1:
        # boundaries of domain (0,1/2) * (0,1), not including {1/2} * (0,1)
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)

        X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob-1)], dim=1)
        X_bottom = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2, temp0], dim=1)
        X_top = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2, temp1], dim=1)
        
        return torch.cat((X_left, X_bottom, X_top), dim=0)  

    if sub_dom == 2:
        # boundaries of domain (1/2,1) * (0,1), not including {1/2} * (0,1)
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)

        X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob-1)], dim=1)
        X_bottom = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2+0.5, temp0], dim=1)
        X_top = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2+0.5, temp1], dim=1)
        
        return torch.cat((X_right, X_bottom, X_top), dim=0)  
 

# draw sample points uniformly at random from the interface {0.5} * (0,1)
def SmpPts_Interface_Square2D(num_intfc_pts, dim_prob=2):    
    """ num_intfc_pts = total number of boundary sampling points at interface
        dim_prob  = dimension of sampling domain """

    temp0 = torch.ones(num_intfc_pts, 1)/2 
    temp1 = torch.rand(num_intfc_pts, dim_prob-1)
    temp2 = torch.unique(temp1)

    while(len(temp1) != len(temp2)):
        temp1 = torch.rand(num_intfc_pts, dim_prob-1)
        temp2 = torch.unique(temp1)
    X_intfc = torch.cat([temp0,temp1],dim=1)

    return X_intfc


# draw equidistant sample points from the domain
def SmpPts_Test_Square2D(num_test_pts, sub_dom):
    """ sub_dom = 1 refers to the Dirichlet subproblem
        sub_dom = 2 refers to the Neumann subproblem
        sub_dom = 0 refers to the entire domain

        num_test_pts = total number of sampling points 
        dim_prob  = dimension of sampling domain """

    if sub_dom == 1:
        xs = torch.linspace(0, 1/2, steps=num_test_pts)
        ys = torch.linspace(0, 1, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)

        return torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                

    if sub_dom == 2:
        xs = torch.linspace(1/2, 1, steps=num_test_pts)
        ys = torch.linspace(0, 1, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)

        return torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                

    if sub_dom == 0:
        xs = torch.linspace(0, 1, steps=num_test_pts)
        ys = torch.linspace(0, 1, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)

        return torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                

##############################################################################################