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

# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from DataSets.Square2D import Sample_Points, Exact_Solution
from itertools import cycle # load data from two datasets within the same loop

# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')

from Solver_Dirichlet_FCNN import SolverDirichletFCNN
from Utils import helper


##############################################################################################
## hyperparameter configuration
## ----------------------------- ##
## parser arguments
parser = argparse.ArgumentParser(description='DtN map for Dirichlet subproblem')

# path for saving results
parser.add_argument('-r', '--result', default='Results', type=str, metavar='PATH', help='path to save checkpoint')

# optimization options
parser.add_argument('--num_epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[500 ,5000, 8000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=3, help='network depth')
parser.add_argument('--width', type=int, default=50, help='network width')

# datasets options 
parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
parser.add_argument('--num_bndry_pts', type=int, default=5000, help='total number of sampling points at each line segment of Dirichlet boundary')
parser.add_argument('--num_intfc_pts', type=int, default=5000, help='total number of sampling points at intefae')
parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

# Dirichlet problem setting    
parser.add_argument('--max_ite_num', type=int, default=2, help='maximum number of outer iterations')

args = parser.parse_args()
## ----------------------------- ##

## ----------------------------- ##
dim_prob = 2
ite_index = 1 # number of outer iteration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## ----------------------------- ##
##############################################################################################


##############################################################################################
# generate training and testing datasets
## ----------------------------- ##
class TraindataInterfaceDirichlet(Dataset):    
    def __init__(self, num_intfc_pts, dim_prob):         
        
        self.smppts_intfc = Sample_Points.SmpPts_Interface_Square2D(num_intfc_pts, dim_prob)       
        self.h_smppts = Exact_Solution.g_Exact_Square2D(self.smppts_intfc[:,0], self.smppts_intfc[:,1])        
            
    def __len__(self):
        return len(self.smppts_intfc)
        
    def __getitem__(self, idx):
        SmpPt = self.smppts_intfc[idx]
        h_SmpPt = self.h_SmpPts[idx]

        return [SmpPt, h_SmpPt]  

# traindata loader (interface)
## ----------------------------- ##
traindata_intfc = TraindataInterfaceDirichlet(args.num_intfc_pts, dim_prob)
##############################################################################################

File_Path = args.result
if not os.path.exists(File_Path):
    os.makedirs(File_Path)

##############################################################################################
## ----------------------------- ##
ErrL2 = [] 
ErrH1 = []

logger = helper.Logger(os.path.join(args.result, 'log.txt'), title='DtN-Map')
logger.set_names(['ite_index', 'error_L2', 'error_H1', 'time'])  

since = time.time()
time_ite = since

# left subproblem-solving
model, error_L2, error_H1 = SolverDirichletFCNN(args, traindata_intfc, ite_index, 1)

# compute testing errors over entire domain
error_L2 = float(error_L2)
error_H1 = float(error_H1)

ErrL2.append(error_L2)
ErrH1.append(error_H1)

time_temp =time.time()
training_time =time_temp - time_ite
time_ite = time_temp
logger.append([ite_index,error_L2,error_H1,training_time])

ErrL2 = np.asarray(ErrL2).reshape(-1,1)
ErrH1 = np.asarray(ErrH1).reshape(-1,1)
io.savemat(args.result+"/ErrL2.mat",{"ErrL2": ErrL2})
io.savemat(args.result+"/ErrH1.mat",{"ErrH1": ErrH1})

logger.close()
time_elapsed = time.time() - since
##############################################################################################