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
from DataSets.Square2D import Sample_Points, Exact_Solution
from Utils import helper
from Solver_Robin_PINN import SolverRobinPINN
# create neural network surrogate model
from Models.FcNet import FcNet

# load data from two datasets within the same loop
from itertools import cycle


##############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='RR-PINNs method for Poisson equation')

# path for saving results
parser.add_argument('-r', '--result', default='Results/', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[600, 800], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=40, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_intfc_pts', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Robin-Robin algorithm setting    
parser.add_argument('--alpha_left', type=float, default=1, help='alpha of the left subproblem')
parser.add_argument('--alpha_right', type=float, default=1000, help='alpha of the right subproblem')
parser.add_argument('--max_ite_num', type=int, default=30, help='maximum number of outer iterations')
# set the stop criteria
parser.add_argument('--tol', type=float, default=0.01, help='tolerance of stopping criteria')
args = parser.parse_known_args()[0]
##############################################################################################
## problem setting
## ----------------------------- ##
dim_prob = 2
##############################################################################################


##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataInterface(Dataset):    
    def __init__(self, num_intfc_pts, dim_prob):         
        
        self.SmpPts_intfc = Sample_Points.SmpPts_Interface_Square2D(num_intfc_pts, dim_prob)       
        self.h_smppts = Exact_Solution.g_Exact_Square2D(self.SmpPts_intfc[:, 0], self.SmpPts_intfc[:, 1])
    def __len__(self):
        return len(self.SmpPts_intfc)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_intfc[idx]
        h_Smppt = self.h_smppts[idx]
        return [SmpPt, h_Smppt]


class Testdata(Dataset):    
    def __init__(self, args, sub_dom): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(args.num_test_pts,sub_dom)
        self.u_Exact_SmpPts = Exact_Solution.u_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])        
        self.Gradu_x_Exact_SmpPts = Exact_Solution.Gradu_x_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])
        self.Gradu_y_Exact_SmpPts = Exact_Solution.Gradu_y_Exact_Square2D(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])         
                 
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Gradu_x_Exact_SmpPt = self.Gradu_x_Exact_SmpPts[idx]
        Gradu_y_Exact_SmpPt = self.Gradu_y_Exact_SmpPts[idx]
 
        return [SmpPt, u_Exact_SmpPt, Gradu_x_Exact_SmpPt, Gradu_y_Exact_SmpPt]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# traindata loader (interface)
## ----------------------------- ##
traindata_intfc = TraindataInterface(args.num_intfc_pts, dim_prob)
SmpPts_Intfc = traindata_intfc.SmpPts_intfc.to(device)

# prepare testing data for checking stopping criteria
## ----------------------------- ##
test_smppts_left = Testdata(args, 1).SmpPts_Test.to(device)
test_smppts_right = Testdata(args, 2).SmpPts_Test.to(device)
##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact = Exact_Solution.u_Exact_Square2D(SmpPts_Intfc[:,0], SmpPts_Intfc[:,1]).reshape(-1,1)
h_diff = SmpPts_Intfc[:,1] * (SmpPts_Intfc[:,1]-1) * SmpPts_Intfc[:,0] * (SmpPts_Intfc[:,0] - 1) # represent the term x * (x - 1) * y * (y - 1)
SmpPts_Intfc.requires_grad = True
g_left = h_exact - 50*h_diff.reshape(-1,1) # the initial guess h^{[0]} = u - 50 * x * (x - 1) * y * (y - 1)
g_left = g_left.reshape(-1,1)
traindata_intfc.h_smppts = g_left
# initial the variable u_left_temp and u_right_temp, which is used to store the solution of previous step
u_left_temp = Exact_Solution.u_Exact_Square2D(test_smppts_left[:, 0], test_smppts_left[:, 1]).reshape(-1,1)
u_right_temp = Exact_Solution.u_Exact_Square2D(test_smppts_right[:, 0], test_smppts_right[:, 1]).reshape(-1,1)
# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DN-PINNs-2D-2prob-line')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
time_ite = since
ite_index = 1

while((ite_index < args.max_ite_num)):

    # left subproblem-solving
    model_left, error_L2_left, error_H1_left = SolverRobinPINN(args, traindata_intfc, ite_index, 1)
    # update Robin boundary condition for right subproblem
    g_right = (args.alpha_right + args.alpha_left) * model_left(SmpPts_Intfc) - g_left
    g_right = g_right.detach()
    # right subproblem-solving
    model_right, error_L2_right, error_H1_right = SolverRobinPINN(args, traindata_intfc, ite_index, 2)


    # compute testing errors over entire domain
    error_L2 = float(error_L2_left + error_L2_right)
    error_H1 = float(error_H1_left + error_H1_right)

    ErrL2.append(error_L2)
    ErrH1.append(error_H1)

    time_temp =time.time()
    training_time =time_temp - time_ite
    time_ite = time_temp
    logger.append([ite_index, error_L2, error_H1, training_time])

    # check if the stop criteria is satisfied
    u_NN_left = model_left(test_smppts_left)
    u_NN_right = model_right(test_smppts_right)
    g_left_temp = (args.alpha_left + args.alpha_right) * model_right(SmpPts_Intfc) - g_right
    if torch.norm(g_left - g_left_temp).item()/torch.norm(g_left_temp).item() < args.tol:
        break
    if (torch.norm(u_left_temp - u_NN_left).item()/torch.norm(u_NN_left).item()< args.tol) or (torch.norm(u_right_temp - u_NN_right).item()/torch.norm(u_NN_right).item() < args.tol):
        break 
    # update Robin boundary condition for left subproblem
    g_left = 1/2 * g_left_temp + (1-1/2) * g_left
    g_left = g_left.detach()
    u_left_temp = u_NN_left
    u_right_temp = u_NN_right
    traindata_intfc.h_smppts = g_left
    ite_index += 1
ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################
