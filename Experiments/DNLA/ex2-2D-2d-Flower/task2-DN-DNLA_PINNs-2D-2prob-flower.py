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
from DataSets.Flower2D import Exact_Solution, Sample_Points
from Utils import helper
# load data from two datasets within the same loop
from itertools import cycle
from Solver_Dirichlet_PINN import SolverDirichletPINN
from Solver_Neumann_DNLA import SolverNeumannDNLA

############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='DNLA (PINNs) method for Poisson problem with flower-shape interface (ex2)')

# path for saving results
parser.add_argument('-r', '--result', default='Results/', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=1000, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[600, 800], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=100, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_intfc_pts', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Dirichlet-Neumann algorithm setting    
parser.add_argument('--max_ite_num', type=int, default=16, help='maximum number of outer iterations')
parser.add_argument('--dim_prob', type=int, default=2, help='dimension of the sub-problem to be solved')
parser.add_argument('--theta', type=float, default=0.5, help='relaxation parameter')
# set the stop criteria
parser.add_argument('--tol', type=float, default=0.01, help='tolerance of stopping criteria')

args = parser.parse_args()
##############################################################################################
## problem setting
## ----------------------------- ##
dim_prob = 2
ite_index = 1 # number of outer iteration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataIntfc(Dataset):    
    def __init__(self, num_intfc_pts, dim_prob):         
        
        self.SmpPts_intfc = Sample_Points.SmpPts_Interface(num_intfc_pts, dim_prob)       
        self.g_SmpPts = Exact_Solution.g_Exact(self.SmpPts_intfc).reshape(-1,1)        

    def __len__(self):
        return len(self.SmpPts_intfc)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_intfc[idx]
        g_SmpPt = self.g_SmpPts[idx]
        return [SmpPt,g_SmpPt]

class Testdata(Dataset):    
    def __init__(self, num_test_pts,sub_dom): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test(num_test_pts, sub_dom)
        self.u_Exact_SmpPts = Exact_Solution.u_Exact(self.SmpPts_Test)        
        self.Grad_u_Exact_SmpPts = Exact_Solution.Grad_u_Exact(self.SmpPts_Test)
                         
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Grad_u_Exact_SmpPt = self.Grad_u_Exact_SmpPts[idx]

        return [SmpPt, u_Exact_SmpPt, Grad_u_Exact_SmpPt]

##############################################################################################
# traindata loader (interface)
## ----------------------------- ##
traindata_intfc = TraindataIntfc(args.num_intfc_pts, dim_prob)
SmpPts_Intfc = traindata_intfc.SmpPts_intfc  

# create the directory if the path is not exist
## ----------------------------- ##
File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)


# prepare testing data for checking stopping criteria
## ----------------------------- ##
testdata_in = Testdata(args.num_test_pts, 1)
testdata_out = Testdata(args.num_test_pts, 2)

Smppts_in = testdata_in.SmpPts_Test.to(device)
Smppts_out = testdata_out.SmpPts_Test.to(device)

# step 1. generate initial guess of interface condition (Dirichlet subproblem)
## ----------------------------- ##
h_exact = Exact_Solution.u_Exact(SmpPts_Intfc).reshape(-1,1)
h_diff = SmpPts_Intfc[:,0] * (SmpPts_Intfc[:,0] - 1) * (SmpPts_Intfc[:,1]) * (SmpPts_Intfc[:,1] - 1) # represent the term x * (x - 1) * y * (y - 1)
g_in = h_exact - 1000*h_diff.reshape(-1,1)  # the initial guess h^{[0]} = u - 1000 * x * (x - 1) * y * (y - 1)
g_in = g_in.reshape(-1,1).to(device).detach()
# initial the variable u_in and u_out, which is used to store the solution of previous step
u_in = Exact_Solution.u_Exact(Smppts_in).reshape(-1,1)
u_out = Exact_Solution.u_Exact(Smppts_out).reshape(-1,1)

traindata_intfc.g_SmpPts = g_in
# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLA-PINNs-2Prob-2D-flower')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
SmpPts_Intfc = SmpPts_Intfc.to(device)
ite_index = 1

while((ite_index < args.max_ite_num)):
    # inner subproblem-solving
    errorL2_in, errorH1_in, model_in = SolverDirichletPINN(args, traindata_intfc, ite_index, 1)
    # outer subproblem-solving
    errorL2_out, errorH1_out, model_out = SolverNeumannDNLA(args, model_in, ite_index, 2)
    # save the testing errors over entire domain
    time_elapse = time.time()
    time_ite = time_elapse - since
    since = time_elapse
    errorL2 = errorL2_in + errorL2_out
    errorH1 = errorH1_in + errorH1_out
    logger.append([ite_index, errorL2, errorH1, time_ite])
    ErrL2.append(errorL2.item())
    ErrH1.append(errorH1.item())
    torch.save(model_in.state_dict(), args.result + "/mode_in_DNLM(PINN)_flower_c1=1_c2=1-%d.pth"%ite_index)
    torch.save(model_out.state_dict(), args.result + "/mode_out_DNLM(PINN)_flower_c1=1_c2=1-%d.pth"%ite_index)

    # check if the stop criteria is satisfied
    g_in_temp =  model_out(SmpPts_Intfc)
    u_in_temp = model_in(Smppts_in)
    u_out_temp = model_out(Smppts_out)

    if torch.norm(g_in - g_in_temp).item()/torch.norm(g_in_temp).item() < args.tol:
        break
    if (torch.norm(u_in_temp - u_in).item()/torch.norm(u_in).item()< args.tol) or (torch.norm(u_out_temp - u_out).item()/torch.norm(u_out_temp).item() < args.tol):
        break 
    # update Dirichlet boundary condition for Dirichlet subproblem
    g_in = args.theta * g_in_temp + (1-args.theta) * g_in
    g_in = g_in.detach()
    u_in = u_in_temp
    u_out = u_out_temp
    traindata_intfc.g_SmpPts = g_in
    ite_index += 1
ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################
