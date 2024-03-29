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
from RobinSolverPINN import *
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')
# load data from two datasets within the same loop
from itertools import cycle


##############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='PINN method for Poisson problem (motivation) with mixed Robin and Dirichlet boundary condition')

# path for saving results
parser.add_argument('-r', '--result', default='Results/Overfit/Robin/2D-squaure/simulation-test', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[2000, 4000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=50, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts_D', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Robin problem setting    
parser.add_argument('--alpha_left', type=float, default=1, help='kappa of the left subproblem')
parser.add_argument('--max_ite_num', type=int, default=2, help='maximum number of outer iterations')

args = parser.parse_args()
##############################################################################################

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
        
        self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(num_test_pts,sub_dom)
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

# traindata loader (interface) with Robin boundary condition
## ----------------------------- ##
traindata_bndry_G = TraindataGamma(args.num_bndry_pts_G, dim_prob)
batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
dataloader_bndry_G = DataLoader(traindata_bndry_G, batch_size=batchsize_bndry_pts_G, shuffle=True, num_workers=0)   
SmpPts_Intfc = traindata_bndry_G.SmpPts_Bndry_G.to(device)     




# prepare testing data over the entire domain
test_smppts = Testdata(args.num_test_pts, 1)
## ----------------------------- ##
##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
# step 1. generate initial guess of interface condition (left subproblem)
## ----------------------------- ##
h_exact = Exact_Solution.h_Exact_Square2D(SmpPts_Intfc[:,0],SmpPts_Intfc[:,1], args.alpha_left).reshape(-1,1)
h_diff = SmpPts_Intfc[:,1]*(SmpPts_Intfc[:,1]-1)*SmpPts_Intfc[:,0]
SmpPts_Intfc.requires_grad = True
g = h_exact - 0*h_diff.reshape(-1,1)
g = g.reshape(-1,1)

# step 2. loop over DDM outer iterations
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DNLM-2Prob-2D-PINN')
logger.set_names(['ite_index', 'error_L2', 'error_H1','time'])  

since = time.time()
time_ite = since
ite_index = 1


# left subproblem-solving
model, error_L2, error_H1 =RobinSolverPINN(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, g, ite_index, 1)


# compute testing errors over entire domain
error_L2 = float(error_L2)
error_H1 = float(error_H1)

ErrL2.append(error_L2)
ErrH1.append(error_H1)

time_temp =time.time()
training_time =time_temp - time_ite
time_ite = time_temp
logger.append([ite_index,error_L2,error_H1,training_time])

smppts_test = test_smppts.SmpPts_Test.to(device)
smppts_test.requires_grad = True
# compute solution over entire domain (for the ease of plotting figures)
u_NN_test = model(smppts_test)  

grad_u_NN = torch.autograd.grad(outputs=u_NN_test, inputs=smppts_test, grad_outputs=torch.ones_like(u_NN_test), retain_graph=True, create_graph=True, only_inputs=True)[0]

u_Exact = test_smppts.u_Exact_SmpPts.reshape(-1,1).to(device)
u_Exact = u_Exact.cpu().detach().numpy()

u_NN_test = u_NN_test.detach().cpu().numpy()
err = u_NN_test - u_Exact

gradu_x_Exact = test_smppts.Gradu_x_Exact_SmpPts.reshape(-1,1).to(device)
gradu_y_Exact = test_smppts.Gradu_y_Exact_SmpPts.reshape(-1,1).to(device)
grad_u_Exact = torch.cat([gradu_x_Exact,gradu_y_Exact],dim=1).detach().cpu().numpy()


## Save the data into ".mat" files and use matlab to plot the figure 
io.savemat(args.result+"/u_exact.mat",{"u_exact": u_Exact})
io.savemat(args.result+"/u_NN_ite%d_test.mat"%ite_index,{"u_NN_test": u_NN_test})
io.savemat(args.result+"/gradu_exact.mat",{"grad_u_exact": grad_u_Exact})    
io.savemat(args.result+"/gradu_NN_ite%d_test.mat"%ite_index,{"grad_u_NN_test": grad_u_NN})
io.savemat(args.result+"/err_test_ite%d.mat"%ite_index, {"pointerr": err})
torch.save(model.state_dict(), args.result + "/mode_Robin_PINN_ite-%d.pth"%ite_index)
ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################