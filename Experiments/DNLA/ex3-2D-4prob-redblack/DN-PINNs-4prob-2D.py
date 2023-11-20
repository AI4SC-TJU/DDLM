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
import matplotlib

from torch import optim, autograd

from matplotlib import pyplot as plt

# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from Datasets.Square2D4prob import Sample_Points, Exact_Solution
from Utils import helper
from DirichletSolverPINN4probR import DirichletSolverPINN
from NeumannSolverPINN4probB import NeumannSolverPINN
# create neural network surrogate model
from Models.FcNet import FcNet


# load data from two datasets within the same loop
from itertools import cycle
## parser arguments
parser = argparse.ArgumentParser(description='DN-PINNs method for Poisson problem with 4 sub-domains(ex2)')

# path for saving results
parser.add_argument('-r', '--result', default='Results/2_4Prob-2D/DN-PINNs/G1e_2-N2e4-baseline/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--mu', default=1000, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[1000, 1500], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=50, help='network width')


# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts_D', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at inteface')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Robin-Robin algorithm setting    
parser.add_argument('--alpha_R', type=float, default=1, help='alpha of the left subproblem')
parser.add_argument('--alpha_B', type=float, default=1, help='alpha of the right subproblem')
parser.add_argument('--max_ite_num', type=int, default=21, help='maximum number of outer iterations')
parser.add_argument('--learning_rate', type=float, default=0.001, help='the initial learning rate')

args = parser.parse_known_args()[0]

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
## problem setting
## ----------------------------- ##
sub_dom = 2
dim_prob = 2
## hyperparameter configuration
## ------------------------- ##
batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_bndry_pts_D = 2*args.num_bndry_pts_D // args.num_batches
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
##############################################################################################
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataIntfc(Dataset):    
    def __init__(self,  num_intfc_pts, item, dim_prob=2):         
        
        self.SmpPts_Bndry_G = Sample_Points.SmpPts_Interface_Square2D(num_intfc_pts, item, dim_prob)
        self.g_1_SmpPts = Exact_Solution.u_Exact_Square2D(self.SmpPts_Bndry_G[:,0],self.SmpPts_Bndry_G[:,1]).reshape(-1,1)       
        
    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        g_1_SmpPts = self.g_1_SmpPts[idx]
        return [SmpPt,g_1_SmpPts]

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
traindata_bndry_G_1 = TraindataIntfc(args.num_bndry_pts_G, 1)
traindata_bndry_G_2 = TraindataIntfc(args.num_bndry_pts_G, 2)
traindata_bndry_G_3 = TraindataIntfc(args.num_bndry_pts_G, 3)
traindata_bndry_G_4 = TraindataIntfc(args.num_bndry_pts_G, 4)
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dataloader_bndry_G_1 = DataLoader(traindata_bndry_G_1, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)
dataloader_bndry_G_2 = DataLoader(traindata_bndry_G_2, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
dataloader_bndry_G_3 = DataLoader(traindata_bndry_G_3, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
dataloader_bndry_G_4 = DataLoader(traindata_bndry_G_4, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)      
SmpPts_Intfc_1 = traindata_bndry_G_1.SmpPts_Bndry_G.to(device)     
SmpPts_Intfc_2 = traindata_bndry_G_2.SmpPts_Bndry_G.to(device)     
SmpPts_Intfc_3 = traindata_bndry_G_3.SmpPts_Bndry_G.to(device)     
SmpPts_Intfc_4 = traindata_bndry_G_4.SmpPts_Bndry_G.to(device)     


##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact_1 = Exact_Solution.u_Exact_Square2D(SmpPts_Intfc_1[:,0],SmpPts_Intfc_1[:,1]).reshape(-1,1)
h_exact_2 = Exact_Solution.u_Exact_Square2D(SmpPts_Intfc_2[:,0],SmpPts_Intfc_2[:,1]).reshape(-1,1)
h_exact_3 = Exact_Solution.u_Exact_Square2D(SmpPts_Intfc_3[:,0],SmpPts_Intfc_3[:,1]).reshape(-1,1)
h_exact_4 = Exact_Solution.u_Exact_Square2D(SmpPts_Intfc_4[:,0],SmpPts_Intfc_4[:,1]).reshape(-1,1)
h_diff_1 =  SmpPts_Intfc_1[:,1]*(SmpPts_Intfc_1[:,1]-1)*SmpPts_Intfc_1[:,0]* (SmpPts_Intfc_1[:,0]-1)
h_diff_2 =  SmpPts_Intfc_2[:,1]*(SmpPts_Intfc_2[:,1]-1)*SmpPts_Intfc_2[:,0]* (SmpPts_Intfc_2[:,0]-1)
h_diff_3 =  SmpPts_Intfc_3[:,1]*(SmpPts_Intfc_3[:,1]-1)*SmpPts_Intfc_3[:,0]* (SmpPts_Intfc_3[:,0]-1)
h_diff_4 =  SmpPts_Intfc_4[:,1]*(SmpPts_Intfc_4[:,1]-1)*SmpPts_Intfc_4[:,0]* (SmpPts_Intfc_4[:,0]-1)
SmpPts_Intfc_1.requires_grad = True
SmpPts_Intfc_2.requires_grad = True
SmpPts_Intfc_3.requires_grad = True
SmpPts_Intfc_4.requires_grad = True

cross_point = 0.5* torch.ones(100,2).to(device)

u_exact_cross = Exact_Solution.u_Exact_Square2D(cross_point[:,0],cross_point[:,1]).reshape(-1,1)
cross_diff = cross_point[:,1]*(cross_point[:,1]-1)*cross_point[:,0]*(cross_point[:,0]-1)
u_cross = u_exact_cross - 1000 * cross_diff.reshape(-1,1)

g_1 = h_exact_1 - 1000*h_diff_1.reshape(-1,1)
g_2 = h_exact_2 - 1000*h_diff_2.reshape(-1,1)
g_3 = h_exact_3 - 1000*h_diff_3.reshape(-1,1)
g_4 = h_exact_4 - 1000*h_diff_4.reshape(-1,1)
g_1 = g_1.reshape(-1,1)
g_2 = g_2.reshape(-1,1)
g_3 = g_3.reshape(-1,1)
g_4 = g_4.reshape(-1,1)
traindata_bndry_G_1.g_1_SmpPts = g_1.detach()
traindata_bndry_G_2.g_1_SmpPts = g_2.detach()
traindata_bndry_G_3.g_1_SmpPts = g_3.detach()
traindata_bndry_G_4.g_1_SmpPts = g_4.detach()

u_cross = u_cross.detach()

# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLM-4Prob-2D-PINN')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
time_ite = since
ite_index = 1

while((ite_index < args.max_ite_num)):

    # Red subproblem-solving

    model_R, error_L2_R, error_H1_R = DirichletSolverPINN(args, traindata_bndry_G_2, dataloader_bndry_G_1, dataloader_bndry_G_2, dataloader_bndry_G_3, dataloader_bndry_G_4, ite_index, u_cross, sub_dom='R')
    '''
    model_R.load_state_dict(torch.load('Results/2_4Prob-2D/DNLM(PINN)/baseline/simulation-test-1/model_ite-1-R.pth'))
    model_R.eval()
    '''
    u_cross = model_R(cross_point)
    u_cross = u_cross.detach()

    # update Neumann boundary condition for Black subproblem
    g_1_temp =  model_R(SmpPts_Intfc_1)
    g_2_temp =  model_R(SmpPts_Intfc_2)
    g_3_temp =  model_R(SmpPts_Intfc_3)
    g_4_temp =  model_R(SmpPts_Intfc_4)

    grad_u_1 =  torch.autograd.grad(outputs=g_1_temp, inputs=SmpPts_Intfc_1, grad_outputs=torch.ones_like(g_1_temp), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_2 =  torch.autograd.grad(outputs=g_2_temp, inputs=SmpPts_Intfc_2, grad_outputs=torch.ones_like(g_2_temp), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_3 =  torch.autograd.grad(outputs=g_3_temp, inputs=SmpPts_Intfc_3, grad_outputs=torch.ones_like(g_3_temp), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_4 =  torch.autograd.grad(outputs=g_4_temp, inputs=SmpPts_Intfc_4, grad_outputs=torch.ones_like(g_4_temp), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    # choose the normal direction
    grad_u_1 = grad_u_1[:,0].detach().reshape(-1,1)
    grad_u_2 = grad_u_2[:,1].detach().reshape(-1,1)
    grad_u_3 = grad_u_3[:,0].detach().reshape(-1,1)
    grad_u_4 = grad_u_4[:,1].detach().reshape(-1,1)

    traindata_bndry_G_1.g_1_SmpPts = grad_u_1
    traindata_bndry_G_2.g_1_SmpPts = grad_u_2
    traindata_bndry_G_3.g_1_SmpPts = grad_u_3
    traindata_bndry_G_4.g_1_SmpPts = grad_u_4
    # Black subproblem-solving

    model_B, error_L2_B, error_H1_B = NeumannSolverPINN(args, traindata_bndry_G_2, dataloader_bndry_G_1, dataloader_bndry_G_2, dataloader_bndry_G_3, dataloader_bndry_G_4, ite_index, u_cross, sub_dom='B')


    # update Dirichlet boundary condition for Black subproblem
    g_1_temp =  model_B(SmpPts_Intfc_1)
    g_2_temp =  model_B(SmpPts_Intfc_2)
    g_3_temp =  model_B(SmpPts_Intfc_3)
    g_4_temp =  model_B(SmpPts_Intfc_4)
        
    g_1 = 1/2 * g_1_temp + (1-1/2) * g_1
    g_2 = 1/2 * g_2_temp + (1-1/2) * g_2
    g_3 = 1/2 * g_3_temp + (1-1/2) * g_3
    g_4 = 1/2 * g_4_temp + (1-1/2) * g_4
    g_1 = g_1.detach()
    g_2 = g_2.detach()
    g_3 = g_3.detach()
    g_4 = g_4.detach()
    u_cross = model_B(cross_point)
    u_cross = u_cross.detach()
    traindata_bndry_G_1.g_1_SmpPts = g_1
    traindata_bndry_G_2.g_1_SmpPts = g_2
    traindata_bndry_G_3.g_1_SmpPts = g_3
    traindata_bndry_G_4.g_1_SmpPts = g_4
    # compute testing errors over entire domain
    error_L2 = float(error_L2_R + error_L2_B)
    error_H1 = float(error_H1_R + error_H1_B)

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
