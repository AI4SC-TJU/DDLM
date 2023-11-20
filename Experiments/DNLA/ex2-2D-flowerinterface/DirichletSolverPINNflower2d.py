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
from matplotlib import pyplot as plt

# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from DataSets.Flower2D import Sample_Points_quarter, Exact_Solution_radius
from Utils import helper

# create neural network surrogate model
from Models.FcNet import FcNet
from Models.FcNet import Resnet
plt.switch_backend('agg')

# load data from two datasets within the same loop
from itertools import cycle



# used for solution exchange between subdomains
def findidx(a,b):
    n = a.shape[0] # 3
    m = b.shape[0] # 2
    _a = a.unsqueeze(0).repeat(m, 1, 1) 
    _b = b.unsqueeze(1).repeat(1, n, 1) 

    match = (_a - _b).sum(-1) # m x n
    indices = (match == 0).nonzero()
    if indices.nelement() > 0 and indices.nelement()<10**-1: # empty tensor check
        row_indices = indices[:, 1]
    else:
        row_indices = []

    return row_indices  


def DirichletSolverPINNflower2d(args, traindata_bndry_G, dataloader_bndry_G, SmpPts_Intfc, iter_num, sub_dom=1):
    print("pytorch version", torch.__version__, "\n")

    dim_prob = 2

    ##############################################################################################
    ## hyperparameter configuration
    ## ------------------------- ##
    batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
    batchsize_bndry_pts_D = 3*args.num_bndry_pts_D // args.num_batches
    batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
    ## ------------------------- ##
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> preparing training and testing datasets ...')
    print('*', '-' * 45, '*')

    # training dataset for sample points inside the domain
    # training dataset for sample points inside the domain
    class TraindataInterior(Dataset):    
        def __init__(self, args, sub_dom, dim_prob ): 
            
            self.SmpPts_Interior = Sample_Points_quarter.SmpPts_Interior(args.num_intrr_pts, sub_dom)
            self.f_Exact_SmpPts = Exact_Solution_radius.f_Exact(self.SmpPts_Interior, args.alpha_R, args.alpha_B, sub_dom)        
        def __len__(self):
            return len(self.SmpPts_Interior)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Interior[idx]
            f_SmpPt = self.f_Exact_SmpPts[idx]

            return [SmpPt, f_SmpPt]
        
    # training dataset for sample points at the Dirichlet boundary
    class TraindataBoundaryDirichlet(Dataset):    
        def __init__(self, args, sub_dom, dim_prob):         
            
            self.SmpPts_Bndry_D = Sample_Points_quarter.SmpPts_Boundary(args.num_bndry_pts_D, sub_dom, dim_prob)
            self.g_SmpPts = Exact_Solution_radius.g_Exact(self.SmpPts_Bndry_D, args.alpha_R, args.alpha_B, sub_dom)        
            
        def __len__(self):
            return len(self.SmpPts_Bndry_D)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Bndry_D[idx]
            g_SmpPt = self.g_SmpPts[idx]

            return [SmpPt, g_SmpPt]    
        

    class TraindataIntfc(Dataset):    
        def __init__(self, args, dim_prob=2):         
            
            self.SmpPts_Bndry_G = Sample_Points_quarter.SmpPts_Interface(args.num_intfc_pts, dim_prob)
            self.h_exact_SmpPts = Exact_Solution_radius.f_Exact(self.SmpPts_Bndry_G, args.alpha_R, args.alpha_B, 1)        
            
        def __len__(self):
            return len(self.SmpPts_Bndry_G)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Bndry_G[idx]
            h_exact_SmpPts = self.h_exact_SmpPts[idx]
            return [SmpPt,h_exact_SmpPts]


    class Testdata(Dataset):    
        def __init__(self, args, sub_dom): 
            
            self.SmpPts_Test = Sample_Points_quarter.SmpPts_Test(args.num_test_pts, sub_dom)
            self.u_Exact_SmpPts = Exact_Solution_radius.u_Exact(self.SmpPts_Test, args.alpha_R, args.alpha_B, sub_dom)        
            self.Grad_u_Exact_SmpPts = Exact_Solution_radius.Grad_u_Exact(self.SmpPts_Test, args.alpha_R, args.alpha_B, sub_dom)
                            
        def __len__(self):
            return len(self.SmpPts_Test)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Test[idx]
            u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
            Grad_u_Exact_SmpPt = self.Grad_u_Exact_SmpPts[idx]
    
            return [SmpPt, u_Exact_SmpPt, Grad_u_Exact_SmpPt]
            
    # create training and testing datasets
    traindata_intrr = TraindataInterior(args, sub_dom, dim_prob)
    traindata_bndry_D = TraindataBoundaryDirichlet(args, sub_dom, dim_prob)
    testdata = Testdata(args, sub_dom)
    # define dataloader 
    dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D = DataLoader(traindata_bndry_D, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(testdata, batch_size=args.num_test_pts*args.num_test_pts, shuffle=False, num_workers=0)
    ##############################################################################################
    plt.figure()
    plt.scatter(traindata_intrr.SmpPts_Interior[:,0], traindata_intrr.SmpPts_Interior[:,1], c = 'red' , label = 'interior points sub_dom1' )
    plt.scatter(traindata_bndry_D.SmpPts_Bndry_D[:,0],  traindata_bndry_D.SmpPts_Bndry_D[:,1], c = 'black', label = 'boundary of sub_dom1' )
    plt.scatter(traindata_bndry_G.SmpPts_Bndry_G[:,0],  traindata_bndry_G.SmpPts_Bndry_G[:,1], c = 'blue', label = 'interface' )

    plt.title('Sample Points during Training \n Square with a hole')
    plt.legend(loc = 'lower right')
    plt.savefig(os.path.join(args.result,'SamplePointsfortraining_D.png')) 

    plt.figure()
    plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'red',   label = 'test points sub_dom1' )

    plt.title('Sample Points during Testing')
    plt.legend(loc = 'lower right')
    plt.savefig(os.path.join(args.result,'SamplePointsfortesting_D.png'))              

    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating training model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def train_epoch(epoch, model, optimizer, device, sub_dom):
        # set model to training mode
        model.train()

        loss_epoch, loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch = 0, 0, 0, 0

        # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
        # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)
        for i, (data_intrr, data_bndry_D, data_bndry_G) in enumerate(zip(dataloader_intrr, cycle(dataloader_bndry_D), cycle(dataloader_bndry_G))):

            smppts_intrr, f_smppts = data_intrr
            smppts_bndry_D, g_D_smppts = data_bndry_D
            smppts_bndry_G, g_1_smppts = data_bndry_G
            



            # send training data to device
            smppts_intrr = smppts_intrr.to(device)
            f_smppts = f_smppts.to(device).reshape(-1,1)
            smppts_bndry_D = smppts_bndry_D.to(device)
            g_D_smppts = g_D_smppts.to(device)
            smppts_bndry_G = smppts_bndry_G.to(device)
            smppts_bndry_G.requires_grad = True
            g_1_smppts = g_1_smppts.to(device)



            smppts_intrr.requires_grad = True
            smppts_bndry_G = smppts_bndry_G.float()
            # forward pass to obtain NN prediction of u(x)
            u_NN_interior = model(smppts_intrr)
            u_NN_bndry_D = model(smppts_bndry_D)
            u_NN_bndry_R = model(smppts_bndry_G.float())
            # zero parameter gradients and then compute NN prediction of gradient u(x)
            model.zero_grad()
            grad_u_NN_intrr = torch.autograd.grad(outputs=u_NN_interior, inputs=smppts_intrr, grad_outputs=torch.ones_like(u_NN_interior), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            
            laplace_u = torch.zeros([len(grad_u_NN_intrr), 1]).float().to(device)
            for index in range(2):
                #p_temp =dfdx[:, index].detach().clone().reshape([len(grad_u_hat[0]), 1]).float().to(device)
                p_temp = grad_u_NN_intrr[:, index].reshape([len(grad_u_NN_intrr), 1])
                #grad_u_hat = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)
                temp = torch.autograd.grad(p_temp,smppts_intrr,grad_outputs=torch.ones_like(p_temp),create_graph=True)[0]
                laplace_u = temp[:, index].reshape([len(grad_u_NN_intrr), 1]) + laplace_u

            # construct mini-batch loss function and then perform backward pass
            loss_intrr = torch.mean((-args.alpha_R * torch.squeeze(laplace_u) - torch.squeeze(f_smppts))**2)
            arclen = Sample_Points_quarter.arclength(len(smppts_bndry_G), dim_prob).to(device)
            loss_bndry_D = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D) - torch.squeeze(g_D_smppts), 2))
    #         loss_bndry_N = torch.mean(torch.pow(- grad_u_NN_bndry_N[:,0] - g_N_smppts, 2))

            loss_bndry_R = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_R) - torch.squeeze(g_1_smppts), 2) * arclen)   
    #         loss_minibatch = loss_intrr + beta * (loss_bndry_N + loss_bndry_D)
    #       loss_minibatch = loss_intrr + beta * loss_bndry_D
            loss_minibatch = loss_intrr + args.beta * (loss_bndry_D + loss_bndry_R)
            #loss_minibatch = loss_intrr + args.beta * loss_bndry_D
            # zero parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss_minibatch.backward()
            # parameter update
            optimizer.step()     

            # integrate loss over the entire training datset
            loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
            loss_bndry_D_epoch += loss_bndry_D.item() * smppts_bndry_D.size(0) / traindata_bndry_D.SmpPts_Bndry_D.shape[0]
            loss_bndry_G_epoch += loss_bndry_R.item() * smppts_bndry_G.size(0) / traindata_bndry_G.SmpPts_Bndry_G.shape[0]
            loss_epoch += loss_intrr_epoch + args.beta * ( loss_bndry_G_epoch + loss_bndry_D_epoch )  
        return loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch, loss_epoch
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating testing model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def test_epoch(epoch, model, optimizer, device):
        
        # set model to testing mode
        model.eval()

        epoch_loss_u, epoch_loss_grad_u = 0, 0
        for smppts_test, u_exact_smppts, grad_u_exact_smppts in dataloader_test:
            
            # send inputs, outputs to device
            smppts_test = smppts_test.to(device)
            u_exact_smppts = u_exact_smppts.to(device)  
            grad_u_exact_smppts = grad_u_exact_smppts.to(device)
            
            smppts_test.requires_grad = True
            
            # forward pass and then compute loss function for approximating u by u_NN
            u_NN_smppts = model(smppts_test) 
            
            loss_u = torch.mean(torch.pow(torch.squeeze(u_NN_smppts) - u_exact_smppts, 2))         
          
            # backward pass to obtain gradient and then compute loss function for approximating grad_u by grad_u_NN
            model.zero_grad()
            gradu_NN_smppts = torch.autograd.grad(outputs=u_NN_smppts, inputs=smppts_test, grad_outputs=torch.ones_like(u_NN_smppts), retain_graph=True, create_graph=True, only_inputs=True)[0]
            
            loss_grad_u = torch.mean(torch.pow(gradu_NN_smppts - grad_u_exact_smppts, 2)) 
                    
            # integrate loss      
            epoch_loss_u += loss_u.item()         
            epoch_loss_grad_u += loss_grad_u.item()  
        
        return epoch_loss_u, epoch_loss_grad_u
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> training neural network ...')
    if not os.path.isdir(args.result):
        helper.mkdir_p(args.result)

    # create model
    model = FcNet.FcNet(dim_prob, args.width, 1, args.depth)
    model.Xavier_initi()
    print(model)
    print('Total number of trainable parameters = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # create optimizer and learning rate schedular
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: {}'.format(device), "\n")
    model = model.to(device)

    # train and test 
    train_loss, test_loss_u, test_loss_grad_u = [], [], []
    trainloss_best = 1e10
    since = time.time()
    for epoch in range(args.num_epochs):        
        print('Epoch {}/{}'.format(epoch, args.num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))              
        # execute training and testing
        trainloss_intrr_epoch, trainloss_bndry_D_epoch, trainloss_bndry_G_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device, sub_dom)
        testloss_u_epoch, testloss_grad_u_epoch = test_epoch(epoch, model, optimizer, device)     
        
        # save current and best models to chec
        # kpoint
        is_best = trainloss_epoch < trainloss_best
        if is_best:
            print('==> Saving best model ...')
        trainloss_best = min(trainloss_epoch, trainloss_best)
        helper.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'trainloss_intrr_epoch': trainloss_intrr_epoch,
                                'trainloss_bndry_D_epoch': trainloss_bndry_D_epoch,
                                'trainloss_bndry_G_epoch': trainloss_bndry_G_epoch,
                                'trainloss_epoch': trainloss_epoch,
                                'testloss_u_epoch': testloss_u_epoch,
                                'testloss_grad_u_epoch': testloss_grad_u_epoch,
                                'trainloss_best': trainloss_best,
                                'optimizer': optimizer.state_dict(),
                            }, is_best, checkpoint=args.result)   
        # save training process to log file
                
        # adjust learning rate according to predefined schedule
        schedular.step()

        # print results
        train_loss.append(trainloss_epoch)
        test_loss_u.append(testloss_u_epoch)
        test_loss_grad_u.append(testloss_grad_u_epoch)
            # # save learning curves


        print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
        print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad(u-u_NN) = {:.4e}'.format(testloss_grad_u_epoch), "\n")
    io.savemat(args.result+"/trainloss%d-sub_%d.mat"%(iter_num,sub_dom),{"trainloss": train_loss})
    io.savemat(args.result+"/testloss%d-sub_%d.mat"%(iter_num,sub_dom),{"testloss": test_loss_u})
    io.savemat(args.result+"/testloss_gradu%d-sub_%d.mat"%(iter_num,sub_dom),{"test_loss_grad_u": test_loss_grad_u})        
    time_elapsed = time.time() - since

    # # save learning curves

    if sub_dom == 1:
        mesh = io.loadmat("flower-quarter-fig.mat")
        x = torch.from_numpy(mesh["node"][0,:]).float().reshape(-1,1)
        y = torch.from_numpy(mesh["node"][1,:]).float().reshape(-1,1)  
    else:
        mesh = io.loadmat("flower-quarter-out-fig.mat")
        x = torch.from_numpy(mesh["node"][0,:]).float().reshape(-1,1)
        y = torch.from_numpy(mesh["node"][1,:]).float().reshape(-1,1)  
    SmpPts_Test = torch.hstack([x, y])     
    SmpPts_Test = SmpPts_Test.to(device)
    SmpPts_Test.requires_grad = True

    u_Exact = Exact_Solution_radius.u_Exact(SmpPts_Test, args.alpha_R, args.alpha_B, sub_dom)
    gradu = Exact_Solution_radius.Grad_u_Exact(SmpPts_Test, args.alpha_R, args.alpha_B, sub_dom)
    gradu_x_Exact = gradu[:,0].reshape(-1,1)
    gradu_y_Exact = gradu[:,1].reshape(-1,1)     
    
    u_NN_Test = model(SmpPts_Test)
    model.zero_grad()
    gradu_NN_test = torch.autograd.grad(outputs=u_NN_Test, inputs=SmpPts_Test, grad_outputs=torch.ones_like(u_NN_Test), retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradu_NN_x_test = gradu_NN_test[:,0].reshape(-1,1)
    gradu_NN_y_test = gradu_NN_test[:,1].reshape(-1,1)

    err = u_NN_Test.squeeze()  - u_Exact.squeeze() 
    errorL2 = torch.norm(err)
    errorH1 = torch.norm(gradu_y_Exact - gradu_NN_y_test) + torch.norm(gradu_x_Exact - gradu_NN_x_test)
    err = err.cpu().detach().numpy()

    u_NN_Test = u_NN_Test.cpu().detach().numpy()
    u_Exact = u_Exact.cpu().detach().numpy()
    grad_u_Exact = torch.cat([gradu_x_Exact,gradu_y_Exact],dim=1).detach().cpu().numpy()
    gradu_NN_test = gradu_NN_test.cpu().detach().numpy()
    gradu_x_Exact = gradu_x_Exact.cpu().detach().numpy()
    gradu_y_Exact = gradu_y_Exact.cpu().detach().numpy()
    

    print('*', '-' * 45, '*', "\n", "\n")
    io.savemat(args.result+"/u_exact_sub%d.mat"%sub_dom,{"u_ex%d"%sub_dom: u_Exact})
    io.savemat(args.result+"/gradu_exact_sub%d.mat"%sub_dom,{"gradu_Exact%d"%sub_dom: grad_u_Exact})
    io.savemat(args.result+"/u_NN_test_ite%d_sub%d.mat"%(iter_num, sub_dom), {"u_NN_sub%d"%sub_dom: u_NN_Test})    
    io.savemat(args.result+"/gradu_NN_test_ite%d_sub%d.mat"%(iter_num, sub_dom),{"grad_u_test%d"%sub_dom: gradu_NN_test})
    io.savemat(args.result+"/err_test_ite%d_sub%d.mat"%(iter_num, sub_dom), {"pointerr%d"%sub_dom: err})    
    torch.save(model.state_dict(), args.result + "/model_ite-%d-%s.pth"%(iter_num, sub_dom))
    ##############################################################################################

    return errorL2, errorH1, model






    
