##############################################################################################
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
from Datasets.Square5Dcos import Exact_Solutionplus, Sample_Points
from DirichletSolverPINN5d import DirichletSolverPINN
from NeumannSolverPINN5d import NeumannSolverPINN
from Utils import helper
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')
# load data from two datasets within the same loop
from itertools import cycle

##############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='DN-PINNs method for 5d Poisson problem(ex4)')

# path for saving results
parser.add_argument('-r', '--result', default='Results/4_2Prob-5D/DN-PINNs/G1e_2-N2e4-baseline/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[600, 1500], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=50, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts_D', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=10, help='number of sampling points for each dimension during testing')

# Dirichlet-Neumann algorithm setting    

parser.add_argument('--max_ite_num', type=int, default=11, help='maximum number of outer iterations')
parser.add_argument('--dim_prob', type=int, default=5, help='dimension of the sub-problem to be solved')

args = parser.parse_args()
##############################################################################################
## problem setting
## ----------------------------- ##
dim_prob = args.dim_prob
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataIntfc(Dataset):    
    def __init__(self, num_bndry_pts_G, dim_prob):         
        
        self.SmpPts_Bndry_G = Sample_Points.SmpPts_Interface_Square10D(num_bndry_pts_G, dim_prob)       
        self.g_1_smppts = Exact_Solutionplus.u_Exact_Square10D(self.SmpPts_Bndry_G, dim_prob).reshape(-1,1)
    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        g_1_smppts = self.g_1_smppts[idx]
        return [SmpPt, g_1_smppts]

class Testdata(Dataset):    
    def __init__(self, num_test_pts, dim_prob): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square10D(num_test_pts, dim_prob)
        self.u_Exact_SmpPts = Exact_Solutionplus.u_Exact_Square10D(self.SmpPts_Test, dim_prob)        
        self.Gradu_x_Exact_SmpPts = Exact_Solutionplus.Gradu_x_Exact_Square10D(self.SmpPts_Test, dim_prob)
        self.Gradu_y_Exact_SmpPts = Exact_Solutionplus.Gradu_y_Exact_Square10D(self.SmpPts_Test, dim_prob)         
        self.Grad_u_Exact_SmpPts = Exact_Solutionplus.Grad_u_Exact_Square10D(self.SmpPts_Test, dim_prob)         
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Gradu_x_Exact_SmpPt = self.Gradu_x_Exact_SmpPts[idx]
        Gradu_y_Exact_SmpPt = self.Gradu_y_Exact_SmpPts[idx]
        Grad_u_Exact_SmpPts = self.Grad_u_Exact_SmpPts[idx]
        return [SmpPt, u_Exact_SmpPt, Gradu_x_Exact_SmpPt, Gradu_y_Exact_SmpPt, Grad_u_Exact_SmpPts]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# traindata loader (interface)
## ----------------------------- ##
traindata_bndry_G = TraindataIntfc(args.num_bndry_pts_G, dim_prob)
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dataloader_bndry_G = DataLoader(traindata_bndry_G, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
SmpPts_Intfc = traindata_bndry_G.SmpPts_Bndry_G.to(device)     

# prepare testing data for left subproblem
## ----------------------------- ##


# prepare testing data for right subproblem
## ----------------------------- ##


# prepare testing data over the entire domain
## ----------------------------- ##
#test_smppts = Testdata(args.num_test_pts,0,dim_prob)
##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact = Exact_Solutionplus.u_Exact_Square10D(SmpPts_Intfc, dim_prob).reshape(-1,1)
h_diff = SmpPts_Intfc[:,0]
for i in range(1,dim_prob):    
    h_diff = h_diff * SmpPts_Intfc[:,i]*(SmpPts_Intfc[:,i]-1)
SmpPts_Intfc.requires_grad = True
g_left = h_exact - 5000*h_diff.reshape(-1,1)
g_left = g_left.reshape(-1,1)
traindata_bndry_G.g_1_smppts = g_left

# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLM-2Prob-10D-PINN')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
time_ite = since

ite_index = 1
while((ite_index < args.max_ite_num)):

    # left subproblem-solving
    model_left, error_L2_left, error_H1_left = DirichletSolverPINN(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, g_left, ite_index, 1)
    # update Robin boundary condition for right subproblem
    
    g_right_temp = model_left(SmpPts_Intfc)
    #g_right_temp = Exact_Solutionplus.u_Exact_Square10D(SmpPts_Intfc, dim_prob).reshape(-1,1)
    g_right =  torch.autograd.grad(outputs=g_right_temp, inputs=SmpPts_Intfc, grad_outputs=torch.ones_like(g_right_temp), retain_graph=True, create_graph=True, only_inputs=True)[0][:,0].reshape(-1,1)        
    traindata_bndry_G.g_1_smppts = g_right.detach()
    # right subproblem-solving

    model_right, error_L2_right, error_H1_right = NeumannSolverPINN(args, traindata_bndry_G, dataloader_bndry_G, ite_index, 2)
    # update Robin boundary condition for left subproblem
    g_left_temp =  model_right(SmpPts_Intfc)
    g_left = 1/2 * g_left_temp + (1-1/2) * g_left
    g_left = g_left.detach()
    traindata_bndry_G.g_1_smppts = g_left
    # compute testing errors over entire domain
    error_L2 = float(error_L2_left + error_L2_right)
    error_H1 = float(error_H1_left + error_H1_right)

    ErrL2.append(error_L2)
    ErrH1.append(error_H1)
    time_temp =time.time()
    training_time =time_temp - time_ite
    time_ite = time_temp
    logger.append([ite_index,error_L2,error_H1,training_time])
    ite_index += 1




ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################