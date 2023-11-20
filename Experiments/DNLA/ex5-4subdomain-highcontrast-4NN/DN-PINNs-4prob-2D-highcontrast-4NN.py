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
from Datasets.Square2D4prob import Sample_Points, Exact_Solution_gradcontrast
from Utils import helper
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')
# load data from two datasets within the same loop
from itertools import cycle
from DirichletSolver4probhighcontrast import DirichletSolverPINN
from NeumannSolver4probhighcontrast import NeumannSolverPINN

############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='DN-PINNs method for Poisson problem with 4 high-contrast coefficient(ex5)')

# path for saving results
parser.add_argument('-r', '--result', default='Results/6_4Prob-2D-highcontrast-4NN/DN-PINNs/G1e_2-N2e4-baseline/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

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
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at inteface')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Dirichlet-Neumann algorithm setting    
parser.add_argument('--alpha_R', type=float, default=1, help='alpha of the red subproblem')
parser.add_argument('--alpha_B', type=float, default=100, help='alpha of the black subproblem')
parser.add_argument('--r0', type=float, default=1, help='radius of the sphere in the square')
parser.add_argument('--max_ite_num', type=int, default=21, help='maximum number of righter iterations')
parser.add_argument('--dim_prob', type=int, default=2, help='dimension of the sub-problem to be solved')

args = parser.parse_args()
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
batchsize_bndry_pts_D = (2*args.dim_prob-1)*args.num_bndry_pts_D // args.num_batches
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dim_prob = args.dim_prob
##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataIntfc(Dataset):    
    def __init__(self, num_bndry_pts_G, item):         
        
        self.SmpPts_Bndry_G = Sample_Points.SmpPts_Interface_Square2D(num_bndry_pts_G, item)      
        self.g_1_SmpPts = Exact_Solution_gradcontrast.u_Exact(self.SmpPts_Bndry_G, args.alpha_R, args.alpha_B, 'R').reshape(-1,1) 
        
    def __len__(self):
        return len(self.SmpPts_Bndry_G)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Bndry_G[idx]
        g_1_SmpPts = self.g_1_SmpPts[idx]
        return [SmpPt,g_1_SmpPts]

class Testdata(Dataset):    
    def __init__(self, num_test_pts, sub_dom, alphain, alpharight, dim_prob): 
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(num_test_pts, sub_dom)
        self.u_Exact_SmpPts = Exact_Solution_gradcontrast.u_Exact(self.SmpPts_Test, alphain, alpharight, sub_dom)        
        self.Grad_u_Exact_SmpPts = Exact_Solution_gradcontrast.Grad_u_Exact(self.SmpPts_Test, alphain, alpharight, sub_dom)
                         
    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Test[idx]
        u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
        Grad_u_Exact_SmpPt = self.Grad_u_Exact_SmpPts[idx]
 
        return [SmpPt, u_Exact_SmpPt, Grad_u_Exact_SmpPt]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# traindata loader (interface)
## ----------------------------- ##
traindata_bndry_G_1 = TraindataIntfc(args.num_bndry_pts_G, 1)
traindata_bndry_G_2 = TraindataIntfc(args.num_bndry_pts_G, 2)
traindata_bndry_G_3 = TraindataIntfc(args.num_bndry_pts_G, 3)
traindata_bndry_G_4 = TraindataIntfc(args.num_bndry_pts_G, 4)
batchsize_bndry_pts_G = 2 * args.num_bndry_pts_G // args.num_batches
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
h_exact_1 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_1, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
h_exact_2 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_2, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
h_exact_3 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_3, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
h_exact_4 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_4, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
h_diff_1 =  SmpPts_Intfc_1[:,0]* (SmpPts_Intfc_1[:,1]-1)
h_diff_2 =  SmpPts_Intfc_2[:,0]* (SmpPts_Intfc_2[:,1]-1)
h_diff_3 =  SmpPts_Intfc_3[:,0]* (SmpPts_Intfc_3[:,1]-1)
h_diff_4 =  SmpPts_Intfc_4[:,0]* (SmpPts_Intfc_4[:,1]-1)
SmpPts_Intfc_1.requires_grad = True
SmpPts_Intfc_2.requires_grad = True
SmpPts_Intfc_3.requires_grad = True
SmpPts_Intfc_4.requires_grad = True




g_1 =   100 * Exact_Solution_gradcontrast.u_0(SmpPts_Intfc_1, args.alpha_R, args.alpha_B, 'R').reshape(-1,1).detach() + 100 * h_diff_1.reshape(-1,1).detach()
g_2 =   100 * Exact_Solution_gradcontrast.u_0(SmpPts_Intfc_2, args.alpha_R, args.alpha_B, 'R').reshape(-1,1).detach() + 100 * h_diff_2.reshape(-1,1).detach()
g_3 =   100 * Exact_Solution_gradcontrast.u_0(SmpPts_Intfc_3, args.alpha_R, args.alpha_B, 'R').reshape(-1,1).detach() + 100 * h_diff_3.reshape(-1,1).detach()
g_4 =   100 * Exact_Solution_gradcontrast.u_0(SmpPts_Intfc_4, args.alpha_R, args.alpha_B, 'R').reshape(-1,1).detach() + 100 * h_diff_4.reshape(-1,1).detach()

'''
g_1 =   h_exact_1
g_2 =   h_exact_2
g_3 =   h_exact_3
g_4 =   h_exact_4
'''
g_1 = g_1.detach()
g_2 = g_2.detach()
g_3 = g_3.detach()
g_4 = g_4.detach()

cross_point = 0.5* torch.ones(100,2).to(device)

u_exact_cross = Exact_Solution_gradcontrast.u_Exact(cross_point, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
cross_diff = cross_point[:,1]*(cross_point[:,1]-1)*cross_point[:,0]*(cross_point[:,0]-1)
u_cross = 100 * Exact_Solution_gradcontrast.u_0(cross_point, args.alpha_R, args.alpha_B, 'R').reshape(-1,1) + 100 * cross_diff.reshape(-1,1)



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
    
    # left subproblem-solving
    model_1, error_L2_1, error_H1_1 = DirichletSolverPINN(args, traindata_bndry_G_2, dataloader_bndry_G_1, dataloader_bndry_G_2, ite_index, u_cross,  sub_dom=1)    
    '''
    model_1.load_state_dict(torch.load('Results/6_4Prob-2D-highcontrast-4NN/DN-PINNs/G1e_2-N2e4-baseline/simulation-1/model_ite-1-1.pth'))
    model_1.eval()
    '''
    u_cross = model_1(cross_point)
    u_cross = u_cross.detach() 
    model_3, error_L2_3, error_H1_3 = DirichletSolverPINN(args, traindata_bndry_G_2, dataloader_bndry_G_3, dataloader_bndry_G_4, ite_index, u_cross,  sub_dom=3)    
    '''
    model_3.load_state_dict(torch.load('Results/6_4Prob-2D-highcontrast-4NN/DN-PINNs/G1e_2-N2e4-baseline/simulation-1/model_ite-1-3.pth'))
    model_3.eval()
    '''
    u_cross = model_3(cross_point)
    u_cross = u_cross.detach()  
    
    u_NN_Intfc_1 = model_1(SmpPts_Intfc_1)
    u_NN_Intfc_2 = model_1(SmpPts_Intfc_2)
    u_NN_Intfc_3 = model_3(SmpPts_Intfc_3)
    u_NN_Intfc_4 = model_3(SmpPts_Intfc_4)
    '''
    u_NN_Intfc_1 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_1, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
    u_NN_Intfc_2 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_2, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
    u_NN_Intfc_3 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_3, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
    u_NN_Intfc_4 = Exact_Solution_gradcontrast.u_Exact(SmpPts_Intfc_4, args.alpha_R, args.alpha_B, 'R').reshape(-1,1)
    '''
    grad_u_1 =  torch.autograd.grad(outputs=u_NN_Intfc_1, inputs=SmpPts_Intfc_1, grad_outputs=torch.ones_like(u_NN_Intfc_1), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_2 =  torch.autograd.grad(outputs=u_NN_Intfc_2, inputs=SmpPts_Intfc_2, grad_outputs=torch.ones_like(u_NN_Intfc_2), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_3 =  torch.autograd.grad(outputs=u_NN_Intfc_3, inputs=SmpPts_Intfc_3, grad_outputs=torch.ones_like(u_NN_Intfc_3), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    grad_u_4 =  torch.autograd.grad(outputs=u_NN_Intfc_4, inputs=SmpPts_Intfc_4, grad_outputs=torch.ones_like(u_NN_Intfc_4), retain_graph=True, create_graph=True, only_inputs=True)[0]        
    
    grad_u_1 = grad_u_1[:,0].detach() * args.alpha_R / args.alpha_B
    grad_u_2 = grad_u_2[:,1].detach() * args.alpha_R / args.alpha_B   
    grad_u_3 = grad_u_3[:,0].detach() * args.alpha_R / args.alpha_B
    grad_u_4 = grad_u_4[:,1].detach() * args.alpha_R / args.alpha_B
    traindata_bndry_G_1.g_1_SmpPts = grad_u_1.reshape(-1,1)
    traindata_bndry_G_2.g_1_SmpPts = grad_u_2.reshape(-1,1)
    traindata_bndry_G_3.g_1_SmpPts = grad_u_3.reshape(-1,1)
    traindata_bndry_G_4.g_1_SmpPts = grad_u_4.reshape(-1,1)


    # right subproblem-solving

    
    model_2, error_L2_2, error_H1_2 = NeumannSolverPINN(args, traindata_bndry_G_1, dataloader_bndry_G_3, dataloader_bndry_G_2, ite_index, u_cross, sub_dom=2)

    model_4, error_L2_4, error_H1_4 = NeumannSolverPINN(args, traindata_bndry_G_1, dataloader_bndry_G_1, dataloader_bndry_G_4, ite_index, u_cross, sub_dom=4)

    # update Dirichlet boundary condition for South-East and North-West subproblem
    g_1_temp =  model_4(SmpPts_Intfc_1)
    g_2_temp =  model_2(SmpPts_Intfc_2)
    g_3_temp =  model_2(SmpPts_Intfc_3)
    g_4_temp =  model_4(SmpPts_Intfc_4)
        
    g_1 = 0.5 * g_1_temp + 0.5 * g_1
    g_2 = 0.5 * g_2_temp + 0.5 * g_2
    g_3 = 0.5 * g_3_temp + 0.5 * g_3
    g_4 = 0.5 * g_4_temp + 0.5 * g_4
    g_1 = g_1.detach()
    g_2 = g_2.detach()
    g_3 = g_3.detach()
    g_4 = g_4.detach()
    traindata_bndry_G_1.g_1_SmpPts = g_1
    traindata_bndry_G_2.g_1_SmpPts = g_2
    traindata_bndry_G_3.g_1_SmpPts = g_3
    traindata_bndry_G_4.g_1_SmpPts = g_4
    # compute testing errors over entire domain
    error_L2 = float(error_L2_1 + error_L2_2 + error_L2_3 + error_L2_4)
    error_H1 = float(error_H1_1 + error_H1_2 + error_H1_3 + error_H1_4)

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