##############################################################################################
import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import argparse 
import scipy.io as io

from torch import optim, autograd
from matplotlib import pyplot as plt

# create training and testing datasets
from torch.utils.data import Dataset, DataLoader
from Datasets.Square2Dcos import Sample_Points, Exact_Solution
from Utils import helper

# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')

# load data from two datasets within the same loop
from itertools import cycle



def CompensentedSolver(args, model_left, iter_num, sub_dom=1):
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
    class TraindataInterior(Dataset):    
        def __init__(self, num_intrr_pts, sub_dom, dim_prob): 
            
            self.SmpPts_Interior = Sample_Points.SmpPts_Interior_Square2D(num_intrr_pts, sub_dom, dim_prob)
            self.f_Exact_SmpPts = Exact_Solution.f_Exact_Square2D(self.SmpPts_Interior[:,0], self.SmpPts_Interior[:,1])        
                
        def __len__(self):
            return len(self.SmpPts_Interior)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Interior[idx]
            f_SmpPt = self.f_Exact_SmpPts[idx]

            return [SmpPt, f_SmpPt]
        
    # training dataset for sample points at the Dirichlet boundary
    class TraindataBoundaryDirichlet(Dataset):    
        def __init__(self, num_bndry_pts_D, sub_dom, dim_prob):         
            
            self.SmpPts_Bndry_D = Sample_Points.SmpPts_Boundary_Square2D(num_bndry_pts_D, sub_dom, dim_prob)
            self.g_SmpPts = Exact_Solution.g_Exact_Square2D(self.SmpPts_Bndry_D[:,0], self.SmpPts_Bndry_D[:,1])        
            
        def __len__(self):
            return len(self.SmpPts_Bndry_D)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Bndry_D[idx]
            g_SmpPt = self.g_SmpPts[idx]

            return [SmpPt, g_SmpPt]    
        

        
    # testing dataset for equidistant sample points over the entire domain
    class Testdata(Dataset):    
        def __init__(self, num_test_pts, sub_dom): 
            
            self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(num_test_pts, sub_dom)
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
            
    # create training and testing datasets
    sub_prob = 2       
    traindata_intrr_N = TraindataInterior(args.num_intrr_pts, sub_prob, dim_prob)
    traindata_bndry_N = TraindataBoundaryDirichlet(args.num_bndry_pts_D, sub_prob, dim_prob)
    testdata_N = Testdata(args.num_test_pts,2)
    
    dataloader_intrr_N = DataLoader(traindata_intrr_N, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_N = DataLoader(traindata_bndry_N, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
    dataloader_test_N = DataLoader(testdata_N, batch_size=args.num_test_pts*args.num_test_pts, shuffle=False, num_workers=0)

    # create training dataset, and then define dataloader for Dirichlet subproblem 
    sub_prob = 1       
    traindata_intrr_D = TraindataInterior(args.num_intrr_pts, sub_prob, dim_prob)
    traindata_bndry_D = TraindataBoundaryDirichlet(args.num_bndry_pts_D,sub_prob, dim_prob)
    
    dataloader_intrr_D = DataLoader(traindata_intrr_D, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D = DataLoader(traindata_bndry_D, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
    ############################################################################################## 
    fig = plt.figure()
    plt.scatter(traindata_intrr_N.SmpPts_Interior[:,0], traindata_intrr_N.SmpPts_Interior[:,1], c = 'red', label = 'interior points' )
    plt.scatter(traindata_bndry_N.SmpPts_Bndry_D[:,0], traindata_bndry_N.SmpPts_Bndry_D[:,1], c = 'blue', label = 'Dirichlet boundary points' )
    
    plt.title('Sample Points during Training')
    plt.legend(loc = 'lower right')
    # plt.show()
    plt.savefig(os.path.join(args.result,'TrainSmpPts_N.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testdata_N.SmpPts_Test[:,0], testdata_N.SmpPts_Test[:,1], c = 'black')
    plt.title('Sample Points during Testing')
    # plt.show()
    plt.savefig(os.path.join(args.result,'TestSmpPts_N.png'))
    plt.close(fig)        

    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating training model ...')
    print('*', '-' * 45, '*', "\n", "\n")
    logger = helper.Logger(os.path.join(args.result, 'log-ite-%d-%s.txt'%(iter_num, sub_dom)), title='RRLM-2Prob-2D-PINN')
    logger.set_names(['epoch', 'loss_N_Ritz','loss_N_Ritz_ex', 'loss_intrr_N_PINN', 'loss_intrr_D', 'loss_intrr_D_ex','loss_bndry_D'])  

    def train_epoch(epoch, model, optimizer, device):
        
        # set model to training mode
        model.train()
        loss_epoch, loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch = 0, 0, 0, 0
        loss_epoch, loss_intrr_N_epoch, loss_intrr_D_epoch, loss_bndry_N_epoch, loss_bndry_D_epoch = 0, 0, 0, 0, 0

        # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
        # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)
        for i, (data_intrr_N, data_bndry_N, data_intrr_D, data_bndry_D) in enumerate(zip(dataloader_intrr_N, cycle(dataloader_bndry_N), cycle(dataloader_intrr_D), cycle(dataloader_bndry_D))):
            
            # get mini-batch training data        
            smppts_intrr_N, f_smppts_N = data_intrr_N
            smppts_bndry_N, g_smppts_N = data_bndry_N

            smppts_intrr_D, f_smppts_D = data_intrr_D
            smppts_bndry_D, g_smppts_D = data_bndry_D

            # send training data to device
            smppts_intrr_N = smppts_intrr_N.to(device)
            f_smppts_N = f_smppts_N.to(device)
            smppts_bndry_N = smppts_bndry_N.to(device)
            g_smppts_N = g_smppts_N.to(device)

            smppts_intrr_D = smppts_intrr_D.to(device)
            f_smppts_D = f_smppts_D.to(device)
            smppts_bndry_D = smppts_bndry_D.to(device)
            g_smppts_D = g_smppts_D.to(device)        
        
            # set requires_grad 
            smppts_intrr_N.requires_grad = True
            smppts_intrr_D.requires_grad = True
            
            # forward pass to obtain NN prediction of u(x) in the domian of Neumann subproblem
            u_NN_intrr_N = model(smppts_intrr_N)
            u_NN_bndry_N = model(smppts_bndry_N)

            # zero parameter gradients and then compute NN prediction of gradient u(x) for Neumann subproblem
            model.zero_grad()
            gradu_NN_intrr_N = torch.autograd.grad(outputs=u_NN_intrr_N, inputs=smppts_intrr_N, grad_outputs = torch.ones_like(u_NN_intrr_N), retain_graph=True, create_graph=True, only_inputs=True)[0]                                    
            
            laplace_u_N = torch.zeros([len(gradu_NN_intrr_N), 1]).float().to(device)
            for index in range(2):
                #p_temp =dfdx[:, index].detach().clone().reshape([len(grad_u_hat[0]), 1]).float().to(device)
                p_temp = gradu_NN_intrr_N[:, index].reshape([len(gradu_NN_intrr_N), 1])
                #grad_u_hat = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)
                temp = torch.autograd.grad(p_temp,smppts_intrr_N,grad_outputs=torch.ones_like(p_temp),create_graph=True)[0]
                laplace_u_N = temp[:, index].reshape([len(gradu_NN_intrr_N), 1]) + laplace_u_N
           
            # forward pass to obtain NN prediction of u(x) in the domian of Dirichlet subproblem
            u_NN_intrr_D = model(smppts_intrr_D)
            u_NN_bndry_D = model(smppts_bndry_D)

            # zero parameter gradients and then compute NN prediction of gradient u(x) for Dirichlet subproblem
            model.zero_grad()
            gradu_NN_intrr_D = torch.autograd.grad(outputs=u_NN_intrr_D, inputs=smppts_intrr_D, grad_outputs = torch.ones_like(u_NN_intrr_D), retain_graph=True, create_graph=True, only_inputs=True)[0]                

            # construct mini-batch loss function and then perform backward pass
            loss_intrr_N = torch.mean(0.5 * torch.sum(torch.pow(gradu_NN_intrr_N, 2), dim=1) - f_smppts_N * torch.squeeze(u_NN_intrr_N))
            loss_intrr_N_PI =  torch.mean(torch.pow(-torch.squeeze(laplace_u_N) - torch.squeeze(f_smppts_N), 2))
            loss_bndry_N = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_N) - g_smppts_N, 2))
            u_NN_left = model_left(smppts_intrr_D)
            u_exact_N = Exact_Solution.u_Exact_Square2D(smppts_intrr_N[:,0],smppts_intrr_N[:,1])
            
            gradu_exact_N = torch.autograd.grad(outputs=u_exact_N, inputs=smppts_intrr_N, grad_outputs = torch.ones_like(u_exact_N), retain_graph=True, create_graph=True, only_inputs=True)[0]                                    
            loss_intrr_N_ex = torch.mean(0.5 * torch.sum(torch.pow(gradu_exact_N, 2), dim=1) - f_smppts_N * torch.squeeze(u_exact_N))

            u_exact_D = Exact_Solution.u_Exact_Square2D(smppts_intrr_D[:,0],smppts_intrr_D[:,1])
            u_NN_left = u_exact_D.reshape(-1,1)
            gradu_NN_left_intrr_D = torch.autograd.grad(outputs=u_NN_left, inputs=smppts_intrr_D, grad_outputs = torch.ones_like(u_NN_left), retain_graph=True, create_graph=True, only_inputs=True)[0]                
            gradu_exact_D = torch.autograd.grad(outputs=u_exact_D, inputs=smppts_intrr_D, grad_outputs = torch.ones_like(u_exact_D), retain_graph=True, create_graph=True, only_inputs=True)[0]                

            loss_intrr_D = torch.mean(f_smppts_D * torch.squeeze(u_NN_intrr_D)) - torch.mean( gradu_NN_left_intrr_D[:,0] * torch.squeeze(gradu_NN_intrr_D[:,0]) + gradu_NN_left_intrr_D[:,1] * torch.squeeze(gradu_NN_intrr_D[:,1]) )
            loss_intrr_D_ex = torch.mean(f_smppts_D * torch.squeeze(u_exact_D)) - torch.mean( gradu_exact_D[:,0] * torch.squeeze(gradu_exact_D[:,0]) + gradu_exact_D[:,1] * torch.squeeze(gradu_exact_D[:,1]) )

            loss_bndry_D = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D) - g_smppts_D, 2))         
            
            loss_minibatch = loss_intrr_N + loss_intrr_N_PI - loss_intrr_D + (args.beta + 10 * epoch) * (loss_bndry_N + loss_bndry_D)
            logger.append([epoch, loss_intrr_N.item(), loss_intrr_N_ex.item(), loss_intrr_N_PI.item(), loss_intrr_D.item(), loss_intrr_D_ex.item(), loss_bndry_N.item()+loss_bndry_D.item()])
            # zero parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss_minibatch.backward()
            # parameter update
            optimizer.step()     

            # integrate loss over the entire training datset
            loss_intrr_N_epoch += loss_intrr_N.item() * smppts_intrr_N.size(0) / traindata_intrr_N.SmpPts_Interior.shape[0]
            loss_intrr_D_epoch += loss_intrr_D.item() * smppts_intrr_D.size(0) / traindata_intrr_D.SmpPts_Interior.shape[0]
            loss_bndry_N_epoch += loss_bndry_N.item() * smppts_bndry_N.size(0) / traindata_bndry_N.SmpPts_Bndry_D.shape[0]
            loss_bndry_D_epoch += loss_bndry_D.item() * smppts_bndry_D.size(0) / traindata_bndry_D.SmpPts_Bndry_D.shape[0]
            loss_epoch += loss_intrr_N_epoch - loss_intrr_D_epoch + args.beta * (loss_bndry_N_epoch + loss_bndry_D_epoch)                    
            loss_intrr_epoch = loss_intrr_N_epoch + loss_intrr_D_epoch
        return loss_intrr_epoch,loss_bndry_D_epoch, loss_bndry_G_epoch, loss_epoch
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating testing model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def test_epoch(epoch, model, optimizer, device):
        
        # set model to testing mode
        model.eval()

        loss_epoch_u, loss_epoch_gradu_x, loss_epoch_gradu_y = 0, 0, 0
        for smppts_test, u_exact_smppts, gradu_x_exact_smppts, gradu_y_exact_smppts in dataloader_test_N:
            
            # send inputs, outputs to device
            smppts_test = smppts_test.to(device)
            u_exact_smppts = u_exact_smppts.to(device)  
            gradu_x_exact_smppts = gradu_x_exact_smppts.to(device)
            gradu_y_exact_smppts = gradu_y_exact_smppts.to(device)
            
            smppts_test.requires_grad = True
            
            # forward pass and then compute loss function for approximating u by u_NN
            u_NN_smppts = model(smppts_test) 
            
            loss_u = torch.mean(torch.pow(torch.squeeze(u_NN_smppts) - u_exact_smppts, 2))         
            
            # backward pass to obtain gradient and then compute loss function for approximating grad_u by grad_u_NN
            model.zero_grad()
            gradu_NN_smppts = torch.autograd.grad(outputs=u_NN_smppts, inputs=smppts_test, grad_outputs=torch.ones_like(u_NN_smppts), retain_graph=True, create_graph=True, only_inputs=True)[0]
            
            loss_gradu_x = torch.mean(torch.pow(torch.squeeze(gradu_NN_smppts[:,0]) - gradu_x_exact_smppts, 2)) 
            loss_gradu_y = torch.mean(torch.pow(torch.squeeze(gradu_NN_smppts[:,1]) - gradu_y_exact_smppts, 2))
                    
            # integrate loss      
            loss_epoch_u += loss_u.item()         
            loss_epoch_gradu_x += loss_gradu_x.item()  
            loss_epoch_gradu_y += loss_gradu_y.item()  
        
        return loss_epoch_u, loss_epoch_gradu_x, loss_epoch_gradu_y
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
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.rate_decay)

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: {}'.format(device), "\n")
    model = model.to(device)

    # train and test 
    train_loss, test_loss_u, test_loss_gradu_x, test_loss_gradu_y = [], [], [], []
    trainloss_best = 1e10
    since = time.time()
    for epoch in range(args.num_epochs):        
        print('Epoch {}/{}'.format(epoch, args.num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))              
        # execute training and testing
        trainloss_intrr_epoch, trainloss_bndry_D_epoch, trainloss_bndry_G_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
        testloss_u_epoch, testloss_gradu_x_epoch, testloss_gradu_y_epoch = test_epoch(epoch, model, optimizer, device)    
        
        # save current and best models to checkpoint
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
                                'testloss_gradu_x_epoch': testloss_gradu_x_epoch,
                                'testloss_gradu_y_epoch': testloss_gradu_y_epoch,
                                'trainloss_best': trainloss_best,
                                'optimizer': optimizer.state_dict(),
                            }, is_best, checkpoint=args.result)   
        # save training process to log file
              
        # adjust learning rate according to predefined schedule
        schedular.step()

        # print results
        train_loss.append(trainloss_epoch)
        test_loss_u.append(testloss_u_epoch)
        test_loss_gradu_x.append(testloss_gradu_x_epoch)
        test_loss_gradu_y.append(testloss_gradu_y_epoch)
        print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
        print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad_x(u-u_NN) = {:.4e}'.format(testloss_gradu_x_epoch), '  Grad_y(u-u_NN) = {:.4e}'.format(testloss_gradu_y_epoch), "\n")
        
    time_elapsed = time.time() - since

    # # save learning curves
    # helper.save_learncurve({'train_curve': train_loss, 'test_curve': test_loss}, curve=args.image)   

    print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
    print('*', '-' * 45, '*', "\n", "\n")
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> loading trained model for inference ...')
    logger.close()
    # load trained model
    # compute NN predicution of u and gradu
    SmpPts_Test = testdata_N.SmpPts_Test    
    SmpPts_Test = SmpPts_Test.to(device)
    SmpPts_Test.requires_grad = True

    u_Exact = testdata_N.u_Exact_SmpPts.reshape(-1,1).to(device)
    gradu_x_Exact = testdata_N.Gradu_x_Exact_SmpPts.reshape(-1,1).to(device)
    gradu_y_Exact = testdata_N.Gradu_y_Exact_SmpPts.reshape(-1,1).to(device)    
    
    u_NN_Test = model(SmpPts_Test)
    model.zero_grad()
    gradu_NN_test = torch.autograd.grad(outputs=u_NN_Test, inputs=SmpPts_Test, grad_outputs=torch.ones_like(u_NN_Test), retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradu_NN_x_test = gradu_NN_test[:,0].reshape(-1,1)
    gradu_NN_y_test = gradu_NN_test[:,1].reshape(-1,1)

    err = u_NN_Test - u_Exact
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
    io.savemat(args.result+"/testloss%d_sub%d.mat"%(iter_num, sub_dom), {"testloss%d"%sub_dom: test_loss_u})
    io.savemat(args.result+"/testloss%d_sub%d_x.mat"%(iter_num, sub_dom), {"testloss%d_x"%sub_dom: test_loss_gradu_x})
    io.savemat(args.result+"/testloss%d_sub%d_y.mat"%(iter_num, sub_dom), {"testloss%d_y"%sub_dom: test_loss_gradu_y})
    io.savemat(args.result+"/trainloss%d_sub%d.mat"%(iter_num, sub_dom), {"trainloss%d"%sub_dom: train_loss})
    torch.save(model.state_dict(), args.result + "/model_ite-%d-%s.pth"%(iter_num, sub_dom))
    return model, errorL2, errorH1
    ##############################################################################################
    
