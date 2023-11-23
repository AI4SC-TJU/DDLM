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
from Datasets.Square2D4prob import Sample_Points, Exact_Solution
from Utils import helper

# create neural network surrogate model
from Models.FcNet import Resnet

# load data from two datasets within the same loop
from itertools import cycle

def SolverNeumannDNLA(args, model_R, iter_num, sub_dom = 'B'):
    print("pytorch version", torch.__version__, "\n")

    dim_prob = 2

    ##############################################################################################
    ## hyperparameter configuration
    ## ------------------------- ##
    batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
    batchsize_bndry_pts = 2*args.num_bndry_pts // args.num_batches
    ## ------------------------- ##
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> preparing training and testing datasets ...')
    print('*', '-' * 45, '*')

    # training dataset for sample points inside the domain
    class TraindataInterior(Dataset):    
        def __init__(self, num_intrr_pts, sub_dom, dim_prob): 
            
            self.SmpPts_intrr = Sample_Points.SmpPts_Interior_Square2D(num_intrr_pts, sub_dom, dim_prob)
            self.f_SmpPts = Exact_Solution.f_Exact_Square2D(self.SmpPts_intrr[:,0], self.SmpPts_intrr[:,1])

        def __len__(self):
            return len(self.SmpPts_intrr)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_intrr[idx]
            f_SmpPt = self.f_SmpPts[idx]

            return [SmpPt, f_SmpPt]
        
    # training dataset for sample points at the Dirichlet boundary
    class TraindataBoundaryDirichlet(Dataset):    
        def __init__(self, num_bndry_pts, sub_dom, dim_prob):         
            
            self.SmpPts_bndry = Sample_Points.SmpPts_Boundary_Square2D(num_bndry_pts, sub_dom, dim_prob)
            self.g_SmpPts = Exact_Solution.g_Exact_Square2D(self.SmpPts_bndry[:,0], self.SmpPts_bndry[:,1])        
            
        def __len__(self):
            return len(self.SmpPts_bndry)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_bndry[idx]
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
    # create training and testing datasets,  and then define dataloader for Neumann subproblem 
    sub_prob = 'B'
    traindata_intrr_B = TraindataInterior(args.num_intrr_pts, sub_prob, dim_prob)
    traindata_bndry_D_B = TraindataBoundaryDirichlet(args.num_bndry_pts, sub_prob, dim_prob)
    
    testdata = Testdata(args.num_test_pts, sub_prob)
    # define dataloader 
    dataloader_intrr_B = DataLoader(traindata_intrr_B, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D_B = DataLoader(traindata_bndry_D_B, batch_size=batchsize_bndry_pts, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(testdata, batch_size=args.num_test_pts * args.num_test_pts, shuffle=False, num_workers=0)
    
    # create training dataset, and then define dataloader for Dirichlet subproblem 
    sub_prob = 'R'
    traindata_intrr_R = TraindataInterior(args.num_intrr_pts, sub_prob, dim_prob)
    traindata_bndry_D_R = TraindataBoundaryDirichlet(args.num_bndry_pts, sub_prob, dim_prob)
        
    dataloader_intrr_R = DataLoader(traindata_intrr_R, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D_R = DataLoader(traindata_bndry_D_R, batch_size=batchsize_bndry_pts, shuffle=True, num_workers=0)
    ##############################################################################################
    # plot the points during training    
    plt.figure()

    plt.scatter(traindata_intrr_R.SmpPts_intrr[:,0], traindata_intrr_R.SmpPts_intrr[:,1], c = 'red' , label = 'interior points red block' )
    plt.scatter(traindata_intrr_B.SmpPts_intrr[:,0], traindata_intrr_B.SmpPts_intrr[:,1], c = 'blue', label = 'interior points black block' )
    plt.scatter(traindata_bndry_D_R.SmpPts_bndry[:,0],  traindata_bndry_D_R.SmpPts_bndry[:,1], c = 'green', label = 'boundary of red block' )
    plt.scatter(traindata_bndry_D_B.SmpPts_bndry[:,0],  traindata_bndry_D_B.SmpPts_bndry[:,1], c = 'black', label = 'boundary of black block' )

    plt.title('Sample Points during Training \n Square with a hole')
    plt.legend(loc = 'lower right')
    str1='SamplePointsfortraining-%s.png'%sub_dom 
    plt.savefig(os.path.join(args.result,str1)) 
    plt.close()
    plt.figure()
    plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'red',   label = 'test points ' )
    # plot the points during testing
    plt.title('Sample Points during Testing')
    plt.legend(loc = 'lower right')
    str1='SamplePointsfortesting-%s.png'%sub_dom 
    plt.savefig(os.path.join(args.result,str1)) 
    plt.close()
    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating training model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def train_epoch(epoch, model, optimizer, device):
        # set model to training mode
        model.train()

        loss_intrr_B_epoch, loss_bndry_B_epoch, loss_intrr_R_epoch, loss_bndry_R_epoch, loss_epoch= 0, 0, 0, 0, 0

        # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
        # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)
        for i, (data_intrr_B, data_intrr_R, data_bndry_D_B, data_bndry_D_R) in enumerate(zip(dataloader_intrr_B, cycle(dataloader_intrr_R), cycle(dataloader_bndry_D_B), cycle(dataloader_bndry_D_R))):
            # get mini-batch training data
            smppts_intrr_B, f_smppts_B = data_intrr_B
            smppts_intrr_R, f_smppts_R = data_intrr_R

            smppts_bndry_D_B, g_D_smppts_B = data_bndry_D_B            
            smppts_bndry_D_R, g_D_smppts_R = data_bndry_D_R


            # send training data to device
            smppts_intrr_B = smppts_intrr_B.to(device)
            f_smppts_B = f_smppts_B.to(device)
            smppts_bndry_D_B = smppts_bndry_D_B.to(device)
            g_D_smppts_B = g_D_smppts_B.to(device)

            smppts_intrr_R = smppts_intrr_R.to(device)
            f_smppts_R = f_smppts_R.to(device)
            smppts_bndry_D_R = smppts_bndry_D_R.to(device)
            g_D_smppts_R = g_D_smppts_R.to(device)
            # set requires_grad
            smppts_intrr_B.requires_grad = True
            smppts_intrr_R.requires_grad = True
            

            # forward pass to obtain NN prediction of u(x) for Black problem
            u_NN_intrr_B = model(smppts_intrr_B)
            u_NN_bndry_D_B = model(smppts_bndry_D_B)

           # zero parameter gradients and then compute NN prediction of gradient u(x) for Black problem
            model.zero_grad()
            grad_u_NN_intrr_B = torch.autograd.grad(outputs=u_NN_intrr_B, inputs=smppts_intrr_B, grad_outputs = torch.ones_like(u_NN_intrr_B), retain_graph=True, create_graph=True, only_inputs=True)[0]        

            # foward pass to obtain NN prediction of u(x) in the domain of Red problem 
            u_NN_intrr_R = model(smppts_intrr_R)
            u_NN_bndry_D_R = model(smppts_bndry_D_R)

            laplace_u = torch.zeros([len(grad_u_NN_intrr_B), 1]).float().to(device)
            for index in range(2):
                p_temp = grad_u_NN_intrr_B[:, index].reshape([len(grad_u_NN_intrr_B), 1])
                temp = torch.autograd.grad(p_temp,smppts_intrr_B,grad_outputs=torch.ones_like(p_temp),create_graph=True)[0]
                laplace_u = temp[:, index].reshape([len(grad_u_NN_intrr_B), 1]) + laplace_u
            grad_u_NN_intrr_R = torch.autograd.grad(outputs=u_NN_intrr_R, inputs=smppts_intrr_R, grad_outputs = torch.ones_like(u_NN_intrr_R), retain_graph=True, create_graph=True, only_inputs=True)[0]        

            # construct mini-batch loss function and then perform backward pass
            loss_intrr_B_PINN = torch.mean((-laplace_u-f_smppts_B.reshape(-1,1))**2)
            loss_intrr_B = torch.mean(0.5  * torch.sum(torch.pow(grad_u_NN_intrr_B, 2), dim=1) - f_smppts_B.reshape(1,-1) * torch.squeeze(u_NN_intrr_B))
            loss_bndry_B = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D_B) - g_D_smppts_B.reshape(1,-1), 2))
            u_R = model_R(smppts_intrr_R)
            grad_u_R = torch.autograd.grad(outputs=u_R, inputs=smppts_intrr_R, grad_outputs = torch.ones_like(u_R), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            loss_intrr_R = torch.mean(-f_smppts_R * torch.squeeze(u_NN_intrr_R)) +  torch.mean( grad_u_R[:,0] * torch.squeeze(grad_u_NN_intrr_R[:,0]) + grad_u_R[:,1] * torch.squeeze(grad_u_NN_intrr_R[:,1]))
            loss_bndry_R = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D_R) - g_D_smppts_R, 2))                 
            loss_minibatch =  loss_intrr_B_PINN + loss_intrr_B + loss_intrr_R + args.beta  * (loss_bndry_B + loss_bndry_R)
            # zero parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss_minibatch.backward()
            # parameter update
            optimizer.step()     

            # integrate loss over the entire training datset
            loss_intrr_B_epoch += loss_intrr_B_PINN.item() * smppts_intrr_B.size(0) / traindata_intrr_B.SmpPts_intrr.shape[0]
            loss_intrr_B_epoch += loss_intrr_B.item() * smppts_intrr_B.size(0) / traindata_intrr_B.SmpPts_intrr.shape[0]
            loss_bndry_B_epoch += loss_bndry_B.item() * smppts_bndry_D_B.size(0) / traindata_bndry_D_B.SmpPts_bndry.shape[0]
            loss_intrr_R_epoch += loss_intrr_R.item() * smppts_intrr_R.size(0) / traindata_intrr_R.SmpPts_intrr.shape[0]
            loss_bndry_R_epoch += loss_bndry_R.item() * smppts_bndry_D_R.size(0) / traindata_bndry_D_R.SmpPts_bndry.shape[0]
            loss_epoch += loss_intrr_B_epoch + args.beta * (loss_bndry_B_epoch + loss_bndry_R_epoch)  
            loss_intrr_epoch = loss_intrr_B_epoch + loss_intrr_R_epoch
            loss_bndry_epoch = loss_bndry_B_epoch + loss_bndry_R_epoch
        return loss_intrr_epoch, loss_bndry_epoch, loss_epoch
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating testing model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def test_epoch(epoch, model, optimizer, device):
        
        # set model to testing mode
        model.eval()

        epoch_loss_u, epoch_loss_gradu_x, epoch_loss_gradu_y = 0, 0, 0
        for smppts_test, u_exact_smppts, gradu_x_exact_smppts, gradu_y_exact_smppts in dataloader_test:
            
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
            epoch_loss_u += loss_u.item()         
            epoch_loss_gradu_x += loss_gradu_x.item()  
            epoch_loss_gradu_y += loss_gradu_y.item()  
        
        return epoch_loss_u, epoch_loss_gradu_x, epoch_loss_gradu_y
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> training neural network ...')

    if not os.path.isdir(args.result):
        helper.mkdir_p(args.result)

    # create model
    model = Resnet.ResNet(dim_prob, args.width, 1, args.depth)
    model.Xavier_initi()
    print(model)
    print('Total number of trainable parameters = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # create optimizer and learning rate schedular
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

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
        trainloss_intrr_epoch, trainloss_bndry_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
        testloss_u_epoch, testloss_gradu_x_epoch, testloss_gradu_y_epoch = test_epoch(epoch, model, optimizer, device)    
        
        # save current and best models to checkpoint
        is_best = trainloss_epoch < trainloss_best
        if is_best:
            print('==> Saving best model ...')
        trainloss_best = min(trainloss_epoch, trainloss_best)
        helper.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'trainloss_intrr_epoch': trainloss_intrr_epoch,
                                'trainloss_bndry_epoch': trainloss_bndry_epoch,
                                'trainloss_epoch': trainloss_epoch,
                                'testloss_u_epoch': testloss_u_epoch,
                                'testloss_gradu_x_epoch': testloss_gradu_x_epoch,
                                'testloss_gradu_y_epoch': testloss_gradu_y_epoch,
                                'trainloss_best': trainloss_best,
                                'optimizer': optimizer.state_dict(),
                            }, is_best, checkpoint=args.result)   
        # adjust learning rate according to predefined schedule
        schedular.step()

        # print results
        train_loss.append(trainloss_epoch)
        test_loss_u.append(testloss_u_epoch)
        test_loss_gradu_x.append(testloss_gradu_x_epoch)
        test_loss_gradu_y.append(testloss_gradu_y_epoch)
        print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
        print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad_x(u-u_NN) = {:.4e}'.format(testloss_gradu_x_epoch), '  Grad_y(u-u_NN) = {:.4e}'.format(testloss_gradu_y_epoch), "\n")
    # save the learning curves
    io.savemat(args.result+"/trainloss%d-sub_%s.mat"%(iter_num, sub_dom),{"trainloss": train_loss})
    io.savemat(args.result+"/testloss%d-sub_%s.mat"%(iter_num, sub_dom),{"testloss": test_loss_u})
    time_elapsed = time.time() - since

    print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
    print('*', '-' * 45, '*', "\n", "\n")
    ##############################################################################################
    # plot learning curves
    fig = plt.figure()
    plt.plot(torch.tensor(train_loss), c = 'red', label = 'training loss' )
    plt.title('Learning Curve during Training')
    plt.legend(loc = 'upper right')
    # plt.show()
    str0='TrainCurve(sub_dom_N_%s)%d.png'%(sub_dom,iter_num)    
    fig.savefig(os.path.join(args.result,str0))
    plt.close()
    fig = plt.figure()
    plt.plot(torch.log10(torch.tensor(test_loss_u)), c = 'red', label = 'testing loss (u)' )
    plt.plot(torch.log10(torch.tensor(test_loss_gradu_x)), c = 'black', label = 'testing loss (gradu_x)' )
    plt.plot(torch.log10(torch.tensor(test_loss_gradu_y)), c = 'green', label = 'testing loss (gradu_y)' )
    plt.title('Learning Curve during Testing')
    plt.legend(loc = 'upper right')
    # plt.show()
    str1='TestCurve(sub_dom_N_%s)%d.png'%(sub_dom,iter_num)    
    fig.savefig(os.path.join(args.result,str1))
    plt.close()
    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> loading trained model for inference ...')

    # save the data predict by neural network
    SmpPts_Test = testdata.SmpPts_Test    
    SmpPts_Test = SmpPts_Test.to(device)
    SmpPts_Test.requires_grad = True

    u_Exact = testdata.u_Exact_SmpPts.reshape(-1,1).to(device)
    gradu_x_Exact = testdata.Gradu_x_Exact_SmpPts.reshape(-1,1).to(device)
    gradu_y_Exact = testdata.Gradu_y_Exact_SmpPts.reshape(-1,1).to(device)    
    
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
    io.savemat(args.result+"/u_exact_sub%s.mat"%sub_dom,{"u_ex%s"%sub_dom: u_Exact})
    io.savemat(args.result+"/gradu_exact_sub%s.mat"%sub_dom,{"gradu_Exact%s"%sub_dom: grad_u_Exact})
    io.savemat(args.result+"/u_NN_test_ite%d_sub%s.mat"%(iter_num, sub_dom), {"u_NN_sub%s"%sub_dom: u_NN_Test})    
    io.savemat(args.result+"/gradu_NN_test_ite%d_sub%s.mat"%(iter_num, sub_dom),{"grad_u_test%s"%sub_dom: gradu_NN_test})
    io.savemat(args.result+"/err_test_ite%d_sub%s.mat"%(iter_num, sub_dom), {"pointerr%s"%sub_dom: err})
    torch.save(model.state_dict(), args.result + "/model_ite-%d-%s.pth"%(iter_num, sub_dom))
    return model, errorL2, errorH1
    ##############################################################################################