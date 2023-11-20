##############################################################################################
import matplotlib
import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import argparse 
import scipy.io as io
import math
from torch import optim, autograd
from matplotlib import projections, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from DataSets.Flower2D import Exact_Solution_radius, Sample_Points_quarter
from Utils import helper
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')
# load data from two datasets within the same loop
from itertools import cycle
from DirichletSolverPINNflower2d import DirichletSolverPINNflower2d
from NeumannSolverPINNflower2d import NeumannSolverPINNflower2d

############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='Physics-Informed Neural Network for Poisson Subproblem with Robin Interface Condition')

# path for saving results
parser.add_argument('-r', '--result', default='Results/2_2Prob-2D-flower/DN-PINNs/R=1-B=1000/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=5000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[1500, 3000, 4000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=100, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts_D', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling point  s at inteface')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Dirichlet-Neumann algorithm setting    
parser.add_argument('--alpha_R', type=float, default=1, help='alpha of the inner subproblem')
parser.add_argument('--alpha_B', type=float, default=1000, help='alpha of the outer subproblem')
parser.add_argument('--r0', type=float, default=0.5, help='radius of the sphere in the square')
parser.add_argument('--max_ite_num', type=int, default=16, help='maximum number of outer iterations')
parser.add_argument('--dim_prob', type=int, default=2, help='dimension of the sub-problem to be solved')
parser.add_argument('--theta', type=float, default=0.5, help='relaxation parameter')
args = parser.parse_args()
##############################################################################################
## problem setting
## ----------------------------- ##
batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_bndry_pts_D = (2*args.dim_prob-1)*args.num_bndry_pts_D // args.num_batches
dim_prob = args.dim_prob
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##

def normalvector(smppts, sub_dom):
    theta = torch.atan(smppts[:,1]/smppts[:,0])
    x = -(6996 * torch.cos(23 * theta) - 7020 * torch.cos(25 * theta) - 9225 * torch.sin(11 * theta) + 15375 * torch.sin(13 * theta) - 1573 * torch.sin(35 * theta) + 1859 * torch.sin(37 * theta) + 4696 * torch.cos(theta))/(8 * (143 * torch.cos(24 * theta)/2 + 4 * torch.sin(12 * theta) + 153/2) ** 1.5)
    y = (9225 * torch.cos(11 * theta) + 15375 * torch.cos(13 * theta) + 1573 * torch.cos(35 * theta) + 1859 * torch.cos(37 * theta) + 6996 * torch.sin(23 * theta) + 7020 * torch.sin(25 * theta) - 4696 * torch.sin(theta))/(8 * (143 * torch.cos(24 * theta)/2 + 4 * torch.sin(12 * theta) + 153/2) ** 1.5)
    temp = torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])
    temp = temp/torch.norm(temp, dim = 1).reshape(-1,1)
    return temp

class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, sub_dom, alphain, alphaout, dim_prob): 
        
        self.SmpPts_Interior = Sample_Points_quarter.SmpPts_Interior(num_intrr_pts, sub_dom, dim_prob)
        self.f_Exact_SmpPts = Exact_Solution_radius.f_Exact(self.SmpPts_Interior, alphain, alphaout, sub_dom)        
    def __len__(self):
        return len(self.SmpPts_Interior)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Interior[idx]
        f_SmpPt = self.f_Exact_SmpPts[idx]

        return [SmpPt, f_SmpPt]
    
# training dataset for sample points at the Dirichlet boundary
class TraindataBoundaryDirichlet(Dataset):    
    def __init__(self, num_bndry_pts_D, sub_dom, alphain, alphaout, dim_prob):         
        
        self.SmpPts_Bndry_D = Sample_Points_quarter.SmpPts_Boundary(num_bndry_pts_D, sub_dom, dim_prob)
        self.g_SmpPts = Exact_Solution_radius.g_Exact(self.SmpPts_Bndry_D, alphain, alphaout, sub_dom)        
        
    def __len__(self):
        return len(self.SmpPts_Bndry_D)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_D[idx]
        g_SmpPt = self.g_SmpPts[idx]

        return [SmpPt, g_SmpPt]    
    

class TraindataIntfc(Dataset):    
    def __init__(self, num_bndry_pts_G, dim_prob):         
        
        self.SmpPts_Bndry_G = Sample_Points_quarter.SmpPts_Interface(num_bndry_pts_G, dim_prob)       
        self.g_SmpPts = Exact_Solution_radius.g_Exact(self.SmpPts_Bndry_G, 1, 1, 1).reshape(-1,1)        

    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        g_SmpPt = self.g_SmpPts[idx]
        return [SmpPt,g_SmpPt]

class Testdata(Dataset):    
    def __init__(self, num_test_pts,sub_dom, alphain, alphaout, dim_prob): 
        
        self.SmpPts_Test = Sample_Points_quarter.SmpPts_Test(num_test_pts, sub_dom)
        self.u_Exact_SmpPts = Exact_Solution_radius.u_Exact(self.SmpPts_Test, alphain, alphaout, sub_dom)        
        self.Grad_u_Exact_SmpPts = Exact_Solution_radius.Grad_u_Exact(self.SmpPts_Test, alphain, alphaout, sub_dom)
                         
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Grad_u_Exact_SmpPt = self.Grad_u_Exact_SmpPts[idx]

        return [SmpPt, u_Exact_SmpPt, Grad_u_Exact_SmpPt]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traindata_bndry_G = TraindataIntfc(args.num_bndry_pts_G, dim_prob)
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dataloader_bndry_G = DataLoader(traindata_bndry_G, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
SmpPts_Intfc = traindata_bndry_G.SmpPts_Bndry_G  



File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)


# prepare testing data over the entire domain

##############################################################################################
##############################################################################################
# step 1. generate initial guess of interface condition (in subproblem)
## ----------------------------- ##
h_exact = Exact_Solution_radius.u_Exact(SmpPts_Intfc, args.alpha_R, args.alpha_B, 1).reshape(-1,1)

h_diff = SmpPts_Intfc[:,0] * (SmpPts_Intfc[:,0] - 1) * (SmpPts_Intfc[:,1]) * (SmpPts_Intfc[:,1] - 1)

g_in = h_exact - 1000*h_diff.reshape(-1,1)

g_in = g_in.reshape(-1,1).to(device).detach()

traindata_bndry_G.g_SmpPts = g_in
# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLM-2Prob-2D-PINN-hole')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
SmpPts_Intfc = SmpPts_Intfc.to(device)
ite_index = 1

while((ite_index < args.max_ite_num)):
    # in subproblem-solving
    args.beta = 1000    
    errorL2_in, errorH1_in, model_in = DirichletSolverPINNflower2d(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, ite_index, 1)
    # update Robin boundary condition for out subproblem
    # out subproblem-solving

    args.beta = 1000 * args.alpha_B
    errorL2_out, errorH1_out, model_out = NeumannSolverPINNflower2d(args, traindata_bndry_G, dataloader_bndry_G, model_in, ite_index, 2)
    # update Robin boundary condition for in subproblem
    g_in_temp =  model_out(SmpPts_Intfc)
    g_in = args.theta * g_in_temp + (1-args.theta)* g_in
    g_in = g_in.detach()
    traindata_bndry_G.g_SmpPts = g_in
    # compute testing errors over entire domain

    time_elapse = time.time()
    time_ite = time_elapse - since
    since = time_elapse
    errorL2 = errorL2_in + errorL2_out
    errorH1 = errorH1_in + errorH1_out
    logger.append([ite_index, errorL2, errorH1, time_ite])
    ErrL2.append(errorL2.item())
    ErrH1.append(errorH1.item())
    torch.save(model_in.state_dict(), args.result + "/mode_in_DN-PINNS_flower_c1=1_c2=1-%d.pth"%ite_index)
    torch.save(model_out.state_dict(), args.result + "/mode_out_DN-PINNS_flower_c1=1_c2=1-%d.pth"%ite_index)
    ite_index += 1
  



ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since




##############################################################################################