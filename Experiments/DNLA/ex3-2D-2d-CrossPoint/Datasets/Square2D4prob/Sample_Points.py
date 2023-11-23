import torch
##############################################################################################
# draw sample points uniformly at random inside the domain
def SmpPts_Interior_Square2D(num_intrr_pts, sub_dom, dim_prob=2):   
    """ num_intrr_pts = total number of sampling points inside the domain
        dim_prob  = dimension of sampling domain 
        sub_dom == R represent the Red block (the westnorth block and eastsouth block)
        sub_dom == B represent the Black block (the westsouth block and eastnorth block)""" 

    if sub_dom=='R':
        # subdomain 1 (0,1/2) * (1/2,1) westnorth 
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0] * 0.5
        X_d[:,1] = X_d[:,1] / 2 + 0.5
        X_d_1 = X_d
        # subdomain 3 (1/2,1) * (0,1/2) eastsouth              
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0] * 0.5 + 0.5
        X_d[:,1] = X_d[:,1] / 2
        X_d_3 = X_d      
        temp = torch.cat((X_d_1, X_d_3), dim = 0)
    if sub_dom=='B':
        # subdomain 2 (0,1/2) * (0,1/2) westsouth
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0] = X_d[:,0] / 2
        X_d[:,1] = X_d[:,1] / 2
        X_d_2 = X_d
        # subdomain 4 (1/2,1) * (1/2,1) eastnorth               
        X_d = torch.rand(num_intrr_pts, dim_prob)  
        X_d[:,0]=X_d[:,0] * 0.5 + 0.5
        X_d[:,1] = X_d[:,1] / 2 + 0.5
        X_d_4 = X_d      
        temp = torch.cat((X_d_2, X_d_4), dim = 0)
    return temp

# draw sample points uniformly at random from the Dirichlet boundaries (not including interface)
def SmpPts_Boundary_Square2D(num_bndry_pts, sub_dom, dim_prob=2):
    if sub_dom=='R':
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)

        # boundary of eastsouth domain (1/2,1) * (0,1/2), not including Interface           
        X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob-1)/2], dim=1)
        X_bottom = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2+0.5, temp0], dim=1)
        X_bndry_3 = torch.cat((X_right, X_bottom), dim=0)
        X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob-1)/2+0.5], dim=1)
        X_top = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2, temp1], dim=1)
        X_bndry_1 = torch.cat((X_left, X_top), dim=0)  
        return torch.cat((X_bndry_1, X_bndry_3),dim=0)
    if sub_dom == 'B':
        temp0 = torch.zeros(num_bndry_pts, 1)
        temp1 = torch.ones(num_bndry_pts, 1)
        # boundary of westsouth domain (0,1/2) * (0,1/2), not including Interface          
        X_left = torch.cat([temp0, torch.rand(num_bndry_pts, dim_prob-1)/2], dim=1)
        X_bottom = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2, temp0], dim=1)          
        X_bndry_2 = torch.cat((X_left, X_bottom), dim=0) 
        X_right = torch.cat([temp1, torch.rand(num_bndry_pts, dim_prob-1)/2+0.5], dim=1)
        X_top = torch.cat([torch.rand(num_bndry_pts, dim_prob-1)/2+0.5, temp1], dim=1)        
        X_bndry_4 = torch.cat((X_right, X_top), dim=0) 
        return torch.cat([X_bndry_2, X_bndry_4], dim=0)

# draw sample points uniformly at random from Interface

def SmpPts_Interface_Square2D(num_bndry_pts, item, dim_prob=2):    
    """ num_bndry_pts = total number of boundary sampling points at Gamma
        dim_prob  = dimension of sampling domain
        sub_dom == R represent the Red block (the westnorth block and eastsouth block)
        sub_dom == B represent the Black block (the westsouth block and eastnorth block)"""
    
    temp0 = torch.ones(num_bndry_pts, 1)/2 
    temp1=torch.rand(num_bndry_pts, dim_prob-1)
    temp2=torch.unique(temp1).reshape(-1,1)
    while(len(temp1)!=len(temp2)):
        temp1=torch.rand(num_bndry_pts, dim_prob-1)
        temp2=torch.unique(temp1).reshape(-1,1)
    X_gam_1=torch.cat([temp0,temp2/2+0.5],dim=1)
    X_gam_2=torch.cat([temp2/2,temp0],dim=1)
    X_gam_3=torch.cat([temp0,temp2/2],dim=1)
    X_gam_4=torch.cat([temp2/2+0.5,temp0],dim=1)
    if item==1:
        # Interface {1/2} * (1/2, 1)
        return X_gam_1
    if item==2:
        # Interface (0, 1/2) * {1/2}
        return X_gam_2
    if item==3:
        # Interface {1/2} * (0, 1/2)
        return X_gam_3
    if item==4:
        # Interface (1/2, 1) * {1/2}
        return X_gam_4

# draw equidistant sample points from the subdomain [0,1/2] * [0,1] and subdomain [1/2,1]*[0,1]
def SmpPts_Test_Square2D(num_test_pts, sub_dom):
    """ num_test_pts = total number of sampling points for model testing 
        sub_dom == R represent the Red block (the westnorth block and eastsouth block)
        sub_dom == B represent the Black block (the westsouth block and eastnorth block)    
    """      
    if sub_dom=='R':        
        xs = torch.linspace(0, 1/2, steps=num_test_pts)
        ys = torch.linspace(1/2, 1, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)
        testdata_1 =  torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                
        xs = torch.linspace(1/2, 1, steps=num_test_pts)
        ys = torch.linspace(0, 1/2, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)
        testdata_3 = torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                
        return torch.cat((testdata_1, testdata_3), dim = 0)
    if sub_dom=='B':        
        xs = torch.linspace(0, 1/2, steps=num_test_pts)
        ys = torch.linspace(0, 1/2, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)
        testdata_2 =  torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                
        xs = torch.linspace(1/2, 1, steps=num_test_pts)
        ys = torch.linspace(1/2, 1, steps=num_test_pts)
        x, y = torch.meshgrid(xs, ys)
        testdata_4 = torch.squeeze(torch.stack([x.reshape(1,num_test_pts*num_test_pts), y.reshape(1,num_test_pts*num_test_pts)], dim=-1))                
        return torch.cat((testdata_2, testdata_4), dim = 0)
##############################################################################################