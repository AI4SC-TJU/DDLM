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
from DataSets.Square2Dcos import Sample_Points, Exact_Solution
from Utils import helper
from RobinSolverPINN import RobinSolverPINN
from CompensentedSolver import CompensentedSolver
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')
# load data from two datasets within the same loop
from itertools import cycle


##############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='RRLM(PINN) method for Poisson equation')

# path for saving results
parser.add_argument('-r', '--result', default='Results/1_2Prob-2D/RRLM(PINN)/G1e_2-N2e4-baseline/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

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
parser.add_argument('--num_bndry_pts_D', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Robin-Robin algorithm setting    
parser.add_argument('--alpha_left', type=float, default=1, help='alpha of the left subproblem')
parser.add_argument('--alpha_right', type=float, default=1000, help='alpha of the right subproblem')
parser.add_argument('--max_ite_num', type=int, default=30, help='maximum number of outer iterations')

args = parser.parse_args()
##############################################################################################
## problem setting
## ----------------------------- ##
dim_prob = 2
##############################################################################################


##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataGamma(Dataset):    
    def __init__(self, num_bndry_pts_G, dim_prob):         
        
        self.SmpPts_Bndry_G = Sample_Points.SmpPts_Interface_Square2D(num_bndry_pts_G, dim_prob)       
        
    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        
        return [SmpPt]

class Testdata(Dataset):    
    def __init__(self, num_test_pts,sub_dom): 
        
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
traindata_bndry_G = TraindataGamma(args.num_bndry_pts_G, dim_prob)
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dataloader_bndry_G = DataLoader(traindata_bndry_G, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
SmpPts_Intfc = traindata_bndry_G.SmpPts_Bndry_G.to(device)     

# prepare testing data for left subproblem
## ----------------------------- ##
xs = torch.linspace(0, 1/2, steps=args.num_test_pts//2)
ys = torch.linspace(0, 1, steps=args.num_test_pts)
x, y = torch.meshgrid(xs, ys)

test_smppts_left = torch.squeeze(torch.stack([x.reshape(1,args.num_test_pts*args.num_test_pts//2), y.reshape(1,args.num_test_pts*args.num_test_pts//2)], dim=-1)).to(device)                
test_smppts_left.requires_grad = True

# prepare testing data for right subproblem
## ----------------------------- ##
xs = torch.linspace(1/2, 1, steps=args.num_test_pts//2)
ys = torch.linspace(0, 1, steps=args.num_test_pts)
x, y = torch.meshgrid(xs, ys)

test_smppts_right= torch.squeeze(torch.stack([x.reshape(1,args.num_test_pts*args.num_test_pts//2), y.reshape(1,args.num_test_pts*args.num_test_pts//2)], dim=-1)).to(device)                
test_smppts_right.requires_grad = True

# prepare testing data over the entire domain
## ----------------------------- ##
test_smppts = Testdata(args.num_test_pts,0)
##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact = Exact_Solution.h_Exact_Square2D(SmpPts_Intfc[:,0],SmpPts_Intfc[:,1],args.alpha_left).reshape(-1,1)
h_diff = SmpPts_Intfc[:,1]*(SmpPts_Intfc[:,1]-1)*SmpPts_Intfc[:,0]
g_left = h_exact - 50*h_diff.reshape(-1,1)
g_left = g_left.reshape(-1,1)

# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger_temp = helper.Logger(os.path.join(args.result, 'logtemp.txt'), title='RRLM-2Prob-2D-PINN')
logger_temp.set_names(['ite_index', 'error_L2', 'error_H1'])  

since = time.time()

ite_index = 1
while((ite_index < args.max_ite_num)):

    # left subproblem-solving
    model_left, error_L2_left, error_H1_left = RobinSolverPINN(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, g_left, ite_index, 1)
    # update Robin boundary condition for right subproblem
    g_right = (args.alpha_right + args.alpha_left) * model_left(SmpPts_Intfc) - g_left
    g_right = g_right.detach()
    # right subproblem-solving
    model_right, error_L2_right, error_H1_right = CompensentedSolver(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, model_left, ite_index, sub_dom=2)
    # update Robin boundary condition for left subproblem
    g_left_temp = (args.alpha_left + args.alpha_right) * model_right(SmpPts_Intfc) - g_right
    g_left = 1/2 * g_left_temp + (1-1/2) * g_left
    g_left = g_left.detach()

    # compute testing errors over entire domain
    error_L2 = float(error_L2_left + error_L2_right)
    error_H1 = float(error_H1_left + error_H1_right)

    ErrL2.append(error_L2)
    ErrH1.append(error_H1)
    logger_temp.append([ite_index,error_L2,error_H1])

    ite_index += 1

    # compute solution over entire domain (for the ease of plotting figures)
    u_NN_left = model_left(test_smppts_left)
    u_NN_right = model_right(test_smppts_right)
    u_NN_test = torch.cat((u_NN_left,u_NN_right),dim=0).detach().cpu().numpy()    

    grad_u_NN_left = torch.autograd.grad(outputs=u_NN_left, inputs=test_smppts_left, grad_outputs=torch.ones_like(u_NN_left), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_u_NN_right = torch.autograd.grad(outputs=u_NN_right, inputs=test_smppts_right, grad_outputs=torch.ones_like(u_NN_right), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_u_NN = torch.cat([grad_u_NN_left,grad_u_NN_right],dim=0).detach().cpu().numpy()
    
    u_Exact = test_smppts.u_Exact_SmpPts.reshape(-1,1).to(device)
    u_Exact = u_Exact.cpu().detach().numpy()

    err = u_NN_test - u_Exact

    gradu_x_Exact = test_smppts.Gradu_x_Exact_SmpPts.reshape(-1,1).to(device)
    gradu_y_Exact = test_smppts.Gradu_y_Exact_SmpPts.reshape(-1,1).to(device)
    grad_u_Exact = torch.cat([gradu_x_Exact,gradu_y_Exact],dim=1).detach().cpu().numpy()

    io.savemat(args.result+"/u_exact.mat",{"u_exact": u_Exact})
    io.savemat(args.result+"/u_NN_ite%d_test.mat"%ite_index,{"u_NN_test": u_NN_test})
    io.savemat(args.result+"/gradu_exact.mat",{"grad_u_exact": grad_u_Exact})    
    io.savemat(args.result+"/gradu_NN_ite%d_test.mat"%ite_index,{"grad_u_NN_test": grad_u_NN})
    io.savemat(args.result+"/err_test_ite%d.mat"%ite_index, {"pointerr": err})

ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger_temp.close()
time_elapsed = time.time() - since
##############################################################################################