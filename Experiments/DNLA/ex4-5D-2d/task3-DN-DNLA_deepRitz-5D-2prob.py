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
from Datasets.Square5D import Exact_Solution, Sample_Points
from Solver_Dirichlet_Ritz import SolverDirichletDeepRitz
from Solver_Neumann_DNLA import SolverNeumannDNLA
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
parser = argparse.ArgumentParser(description='DNLM(Ritz) method for 5d Poisson problem(ex4)')

# path for saving results
parser.add_argument('-r', '--result', default='Results/', type=str, metavar='PATH', help='path to save checkpoint')

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
parser.add_argument('--num_bndry_pts', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_intfc_pts', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=10, help='number of sampling points for each dimension during testing')

# Dirichlet-Neumann algorithm setting    

parser.add_argument('--max_ite_num', type=int, default=30, help='maximum number of outer iterations')
parser.add_argument('--dim_prob', type=int, default=5, help='dimension of the sub-problem to be solved')
# set the stop criteria
parser.add_argument('--tol', type=float, default=0.01, help='tolerance of stopping criteria')

args = parser.parse_args()
##############################################################################################
## problem setting
## ----------------------------- ##
dim_prob = args.dim_prob
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataIntfc(Dataset):    
    def __init__(self, num_intfc_pts, dim_prob):         
        
        self.SmpPts_intfc = Sample_Points.SmpPts_Interface_Square5D(num_intfc_pts, dim_prob)       
        self.h_smppts = Exact_Solution.u_Exact_Square5D(self.SmpPts_intfc, dim_prob).reshape(-1,1)
    def __len__(self):
        return len(self.SmpPts_intfc)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_intfc[idx]
        h_smppts = self.h_smppts[idx]
        return [SmpPt, h_smppts]

class Testdata(Dataset):    
    def __init__(self, num_test_pts, sub_dom): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square5D(num_test_pts, sub_dom, dim_prob)
        self.u_Exact_SmpPts = Exact_Solution.u_Exact_Square5D(self.SmpPts_Test, dim_prob)        
        self.Gradu_Exact_SmpPts = Exact_Solution.Gradu_Exact_Square5D(self.SmpPts_Test, dim_prob)   
                
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Gradu_Exact_SmpPt = self.Gradu_Exact_SmpPts[idx]


        return [SmpPt, u_Exact_SmpPt, Gradu_Exact_SmpPt] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# traindata loader (interface)
## ----------------------------- ##
traindata_intfc = TraindataIntfc(args.num_intfc_pts, dim_prob)
SmpPts_Intfc = traindata_intfc.SmpPts_intfc.to(device)     

# prepare testing data over the entire domain

##############################################################################################
testdata_left = Testdata(args.num_test_pts, 1)
testdata_right = Testdata(args.num_test_pts, 2)

Smppts_left = testdata_left.SmpPts_Test.to(device)
Smppts_right = testdata_right.SmpPts_Test.to(device)



##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact = Exact_Solution.u_Exact_Square5D(SmpPts_Intfc, dim_prob).reshape(-1,1)
h_diff = SmpPts_Intfc[:,0] * (SmpPts_Intfc[:,0] - 1)
for i in range(1,dim_prob):    
    h_diff = h_diff * SmpPts_Intfc[:,i]*(SmpPts_Intfc[:,i]-1)
SmpPts_Intfc.requires_grad = True
g_left = h_exact - 5000*h_diff.reshape(-1,1)
g_left = g_left.reshape(-1,1)
traindata_intfc.h_smppts = g_left

u_left = Exact_Solution.u_Exact_Square5D(Smppts_left, dim_prob).reshape(-1,1)
u_right = Exact_Solution.u_Exact_Square5D(Smppts_right, dim_prob).reshape(-1,1)

# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLM-2Prob-5D-PINN')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
time_ite = since

ite_index = 1
while((ite_index < args.max_ite_num)):

    # left subproblem-solving
    model_left, error_L2_left, error_H1_left = SolverDirichletDeepRitz(args, traindata_intfc, ite_index, 1)
    # right subproblem-solving
    model_right, error_L2_right, error_H1_right = SolverNeumannDNLA(args, model_left, ite_index, 2)
    # compute testing errors over entire domain
    error_L2 = float(error_L2_left + error_L2_right)
    error_H1 = float(error_H1_left + error_H1_right)

    ErrL2.append(error_L2)
    ErrH1.append(error_H1)
    time_temp =time.time()
    training_time =time_temp - time_ite
    time_ite = time_temp
    logger.append([ite_index,error_L2,error_H1,training_time])
    

    # check if the stop criteria is satisfied
    g_left_temp =  model_right(SmpPts_Intfc)
    u_left_temp = model_left(Smppts_left)
    u_right_temp = model_right(Smppts_right)
    if torch.norm(g_left - g_left_temp).item()/torch.norm(g_left_temp).item() < args.tol:
        break
    if (torch.norm(u_left_temp - u_left).item()/torch.norm(u_left).item()< args.tol) or (torch.norm(u_right_temp - u_right).item()/torch.norm(u_right_temp).item() < args.tol):
        break 
    # update Dirichlet boundary condition for left subproblem
    g_left_temp =  model_right(SmpPts_Intfc)
    g_left = 1/2 * g_left_temp + (1-1/2) * g_left
    g_left = g_left.detach()
    u_left = u_left_temp
    u_right = u_right_temp
    traindata_intfc.h_smppts = g_left
    ite_index += 1


ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################
