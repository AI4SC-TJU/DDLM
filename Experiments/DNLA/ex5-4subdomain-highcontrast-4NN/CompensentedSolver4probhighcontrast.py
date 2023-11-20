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
from Datasets.Square2D4prob import Sample_Points, Exact_Solution_gradcontrast
from Utils import helper

# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')

# load data from two datasets within the same loop
from itertools import cycle




def CompensentSolver(args, model_E_1, model_E_2, iter_num, u_cross, sub_dom='2'):
    print("pytorch version", torch.__version__, "\n")

    dim_prob = 2

    ##############################################################################################
    ## hyperparameter configuration
    ## ------------------------- ##
    batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
    batchsize_bndry_pts_D = 2*args.num_bndry_pts_D // args.num_batches
    #batchsize_bndry_pts_G = args.num_bndry_pts_G // args.num_batches
    ## ------------------------- ##
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> preparing training and testing datasets ...')
    print('*', '-' * 45, '*')

    # training dataset for sample points inside the domain
    class TraindataInterior(Dataset):    
        def __init__(self, num_intrr_pts, sub_dom, alphain, alphaout, dim_prob): 
            
            self.SmpPts_Interior = Sample_Points.SmpPts_Interior_Square2D(num_intrr_pts, sub_dom, dim_prob)
            self.f_Exact_SmpPts = Exact_Solution_gradcontrast.f_Exact(self.SmpPts_Interior, alphain, alphaout, sub_dom)        
        def __len__(self):
            return len(self.SmpPts_Interior)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Interior[idx]
            f_SmpPt = self.f_Exact_SmpPts[idx]

            return [SmpPt, f_SmpPt]
        
    # training dataset for sample points at the Dirichlet boundary
    class TraindataBoundaryDirichlet(Dataset):    
        def __init__(self, num_bndry_pts_D, sub_dom, alphain, alphaout, dim_prob):         
            
            self.SmpPts_Bndry_D = Sample_Points.SmpPts_Boundary_Square2D(num_bndry_pts_D, sub_dom, dim_prob)
            self.g_SmpPts = Exact_Solution_gradcontrast.g_Exact(self.SmpPts_Bndry_D, alphain, alphaout, sub_dom)        
            
        def __len__(self):
            return len(self.SmpPts_Bndry_D)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Bndry_D[idx]
            g_SmpPt = self.g_SmpPts[idx]

            return [SmpPt, g_SmpPt]    
        

    # testing dataset for equidistant sample points over the entire domain
    class Testdata(Dataset):    
        def __init__(self, num_test_pts,sub_dom, alphain, alphaout): 
            
            self.SmpPts_Test = Sample_Points.SmpPts_Test_Square2D(num_test_pts, sub_dom)
            self.u_Exact_SmpPts = Exact_Solution_gradcontrast.u_Exact(self.SmpPts_Test, alphain, alphaout, sub_dom)        
            self.Grad_u_Exact_SmpPts = Exact_Solution_gradcontrast.Grad_u_Exact(self.SmpPts_Test, alphain, alphaout, sub_dom)
                            
        def __len__(self):
            return len(self.SmpPts_Test)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Test[idx]
            u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
            Grad_u_Exact_SmpPt = self.Grad_u_Exact_SmpPts[idx]
    
            return [SmpPt, u_Exact_SmpPt, Grad_u_Exact_SmpPt]
                
    # create training and testing datasets

    traindata_intrr = TraindataInterior(args.num_intrr_pts, sub_dom, args.alpha_R, args.alpha_B, dim_prob)
    traindata_bndry_D = TraindataBoundaryDirichlet(args.num_bndry_pts_D, sub_dom, args.alpha_R, args.alpha_B, dim_prob)
    testdata = Testdata(args.num_test_pts, sub_dom, args.alpha_R, args.alpha_B)
    # define dataloader 
    dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D = DataLoader(traindata_bndry_D, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(testdata, batch_size=args.num_test_pts*args.num_test_pts, shuffle=False, num_workers=0)
    #certify the extension area:
    if sub_dom == 1:
        Extensionarea = ['1E2','1E4']
    if sub_dom == 2:
        Extensionarea = ['2E1','2E3']
    if sub_dom == 3:
        Extensionarea = ['3E2','3E4']    
    if sub_dom == 4:
        Extensionarea = ['4E1','4E3']   

    traindata_intrr_E_1= TraindataInterior(args.num_intrr_pts, Extensionarea[0], args.alpha_R, args.alpha_B, dim_prob)
    traindata_bndry_D_E_1 = TraindataBoundaryDirichlet(args.num_bndry_pts_D, Extensionarea[0], args.alpha_R, args.alpha_B, dim_prob)
    # define dataloader 
    dataloader_intrr_E_1 = DataLoader(traindata_intrr_E_1, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D_E_1 = DataLoader(traindata_bndry_D_E_1, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)

    traindata_intrr_E_2= TraindataInterior(args.num_intrr_pts, Extensionarea[1], args.alpha_R, args.alpha_B, dim_prob)
    traindata_bndry_D_E_2 = TraindataBoundaryDirichlet(args.num_bndry_pts_D, Extensionarea[1], args.alpha_R, args.alpha_B, dim_prob)
    # define dataloader 
    dataloader_intrr_E_2 = DataLoader(traindata_intrr_E_1, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D_E_2 = DataLoader(traindata_bndry_D_E_1, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)

       ##############################################################################################
    plt.figure()

    plt.scatter(traindata_intrr.SmpPts_Interior[:,0], traindata_intrr.SmpPts_Interior[:,1], c = 'red' , label = 'interior points' )
    plt.scatter(traindata_intrr_E_1.SmpPts_Interior[:,0], traindata_intrr_E_1.SmpPts_Interior[:,1], c = 'blue', label = 'interior points Extension area 1' )
    plt.scatter(traindata_intrr_E_2.SmpPts_Interior[:,0], traindata_intrr_E_2.SmpPts_Interior[:,1], c = 'grey', label = 'interior points Extension area 2' )
    plt.scatter(traindata_bndry_D.SmpPts_Bndry_D[:,0],  traindata_bndry_D.SmpPts_Bndry_D[:,1], c = 'green', label = 'boundary of right' )
    plt.scatter(traindata_bndry_D_E_1.SmpPts_Bndry_D[:,0],  traindata_bndry_D_E_1.SmpPts_Bndry_D[:,1], c = 'black', label = 'boundary of Extension area 1' )
    plt.scatter(traindata_bndry_D_E_2.SmpPts_Bndry_D[:,0],  traindata_bndry_D_E_2.SmpPts_Bndry_D[:,1], c = 'yellow', label = 'boundary of Extension area 2' )
    plt.title('Sample Points during Training \n Square with a hole')
    plt.legend(loc = 'lower right')
    str1='SamplePointsfortraining-%s.png'%sub_dom 
    plt.savefig(os.path.join(args.result,str1)) 
    plt.close()
    plt.figure()
    plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'red',   label = 'test points ' )

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

        loss_intrr_epoch, loss_bndry_epoch, loss_intrr_E_epoch, loss_bndry_E_epoch, loss_epoch= 0, 0, 0, 0, 0

        # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
        # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)
        for i, (data_intrr, data_intrr_E_1, data_intrr_E_2, data_bndry_D, data_bndry_D_E_1, data_bndry_D_E_2) in enumerate(zip(dataloader_intrr, cycle(dataloader_intrr_E_1), cycle(dataloader_intrr_E_2), cycle(dataloader_bndry_D), cycle(dataloader_bndry_D_E_1), cycle(dataloader_bndry_D_E_2))):
            # get mini-batch training data
            smppts_intrr, f_smppts = data_intrr
            smppts_intrr_E_1, f_smppts_E_1 = data_intrr_E_1
            smppts_intrr_E_2, f_smppts_E_2 = data_intrr_E_2

            smppts_bndry_D, g_D_smppts = data_bndry_D            
            smppts_bndry_D_E_1, g_D_smppts_E_1= data_bndry_D_E_1
            smppts_bndry_D_E_2, g_D_smppts_E_2= data_bndry_D_E_2
            # send training data to device

            smppts_intrr = smppts_intrr.to(device)
            smppts_intrr_E_1 = smppts_intrr_E_1.to(device)
            smppts_intrr_E_2 = smppts_intrr_E_2.to(device)
            f_smppts = f_smppts.to(device)
            f_smppts_E_1 = f_smppts_E_1.to(device)
            f_smppts_E_2 = f_smppts_E_2.to(device)
            smppts_bndry_D = smppts_bndry_D.to(device)
            smppts_bndry_D_E_1 = smppts_bndry_D_E_1.to(device)
            smppts_bndry_D_E_2 = smppts_bndry_D_E_2.to(device)
            g_D_smppts = g_D_smppts.to(device)
            g_D_smppts_E_1 = g_D_smppts_E_1.to(device)
            g_D_smppts_E_2 = g_D_smppts_E_2.to(device)

            # set requires_grad
            smppts_intrr.requires_grad = True
            smppts_intrr_E_1.requires_grad = True
            smppts_intrr_E_2.requires_grad = True
            # forward pass to obtain NN prediction of u(x) for Black problem
            u_NN_intrr = model(smppts_intrr)
            u_NN_bndry_D = model(smppts_bndry_D)

            # foward pass to obtain NN prediction of u(x) in the domain of Red problem 
            u_NN_intrr_E_1 = model(smppts_intrr_E_1)
            u_NN_intrr_E_2 = model(smppts_intrr_E_2)
            u_NN_bndry_D_E_1 = model(smppts_bndry_D_E_1)
            u_NN_bndry_D_E_2 = model(smppts_bndry_D_E_2)
            # zero parameter gradients and then compute NN prediction of gradient u(x) for Red subproblem
            model.zero_grad()
            grad_u_NN_intrr_E_1 = torch.autograd.grad(outputs=u_NN_intrr_E_1, inputs=smppts_intrr_E_1, grad_outputs = torch.ones_like(u_NN_intrr_E_1), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            grad_u_NN_intrr_E_2 = torch.autograd.grad(outputs=u_NN_intrr_E_2, inputs=smppts_intrr_E_2, grad_outputs = torch.ones_like(u_NN_intrr_E_2), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            grad_u_NN_intrr = torch.autograd.grad(outputs=u_NN_intrr, inputs=smppts_intrr, grad_outputs = torch.ones_like(u_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            laplace_u = torch.zeros([len(grad_u_NN_intrr), 1]).float().to(device)
            for index in range(2):
                #p_temp =dfdx[:, index].detach().clone().reshape([len(grad_u_hat[0]), 1]).float().to(device)
                p_temp = grad_u_NN_intrr[:, index].reshape([len(grad_u_NN_intrr), 1])
                #grad_u_hat = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)
                temp = torch.autograd.grad(p_temp,smppts_intrr,grad_outputs=torch.ones_like(p_temp),create_graph=True)[0]
                laplace_u = temp[:, index].reshape([len(grad_u_NN_intrr), 1]) + laplace_u
            # construct mini-batch loss function and then perform backward pass
            loss_intrr = torch.mean(0.5 * args.alpha_B * torch.sum(torch.pow(grad_u_NN_intrr, 2), dim=1) - f_smppts * torch.squeeze(u_NN_intrr))
            loss_intrr += torch.mean((-args.alpha_B * laplace_u-f_smppts.reshape(-1,1))**2)
            loss_bndry = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D) - g_D_smppts, 2))
            #u_E_1 = model_E_1(smppts_intrr_E_1)
            #u_E_2 = model_E_2(smppts_intrr_E_2)    
            u_E_1 = Exact_Solution_gradcontrast.u_Exact(smppts_intrr_E_1, args.alpha_R, args.alpha_B, 1).reshape(-1,1)
            u_E_2 = Exact_Solution_gradcontrast.u_Exact(smppts_intrr_E_2, args.alpha_R, args.alpha_B, 3).reshape(-1,1)
            grad_u_E_1 = torch.autograd.grad(outputs=u_E_1, inputs=smppts_intrr_E_1, grad_outputs = torch.ones_like(u_E_1), retain_graph=True, create_graph=True, only_inputs=True)[0]                
            grad_u_E_2 = torch.autograd.grad(outputs=u_E_2, inputs=smppts_intrr_E_2, grad_outputs = torch.ones_like(u_E_2), retain_graph=True, create_graph=True, only_inputs=True)[0] 
            cross_point = 0.5*torch.ones(100,2).to(device)
            u_NN_cross = model(cross_point)
            loss_cross   = torch.mean(torch.square(u_cross - u_NN_cross))         
            loss_intrr_E_1 = torch.mean(-f_smppts_E_1 * torch.squeeze(u_NN_intrr_E_1)) + args.alpha_R * torch.mean( grad_u_E_1[:,0] * torch.squeeze(grad_u_NN_intrr_E_1[:,0]) + grad_u_E_1[:,1] * torch.squeeze(grad_u_NN_intrr_E_1[:,1]))
            loss_intrr_E_2 = torch.mean(-f_smppts_E_2 * torch.squeeze(u_NN_intrr_E_2)) + args.alpha_R * torch.mean( grad_u_E_2[:,0] * torch.squeeze(grad_u_NN_intrr_E_2[:,0]) + grad_u_E_2[:,1] * torch.squeeze(grad_u_NN_intrr_E_2[:,1]))
            loss_bndry_E_1 = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D_E_1) - g_D_smppts_E_1, 2))         
            loss_bndry_E_2 = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D_E_2) - g_D_smppts_E_2, 2))

            loss_intrr_E = loss_intrr_E_1 + loss_intrr_E_2
            loss_bndry_E = loss_bndry_E_1 + loss_bndry_E_2
            #loss_minibatch =  loss_intrr + loss_intrr_E + args.beta * (loss_bndry + loss_bndry_E + loss_cross)


            #u_ex = Exact_Solution.u_Exact_Square2D(smppts_intrr[:,0], smppts_intrr[:,1]).reshape(-1,1)
            #grad_u_ex = torch.autograd.grad(outputs=u_ex, inputs=smppts_intrr, grad_outputs = torch.ones_like(u_ex), retain_graph=True, create_graph=True, only_inputs=True)[0]        

            #loss_intrr_ex = torch.mean(0.5 * torch.sum(torch.pow(grad_u_ex, 2), dim=1) - f_smppts * torch.squeeze(u_ex))

            #loss_minibatch = loss_intrr + loss_intrr_E + args.beta * loss_bndry + args.mu * loss_bndry_E + args.beta * loss_cross
            loss_minibatch =  loss_intrr + loss_intrr_E + args.beta * (loss_bndry + loss_bndry_E)
            # zero parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss_minibatch.backward()
            # parameter update
            optimizer.step()     

            # integrate loss over the entire training datset
            loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
            loss_bndry_epoch += loss_bndry.item() * smppts_bndry_D.size(0) / traindata_bndry_D.SmpPts_Bndry_D.shape[0]
            loss_intrr_E_epoch += loss_intrr_E.item() * smppts_intrr_E_1.size(0) / traindata_intrr_E_1.SmpPts_Interior.shape[0]
            loss_bndry_E_epoch += loss_bndry_E.item() * smppts_bndry_D_E_1.size(0) / traindata_bndry_D_E_1.SmpPts_Bndry_D.shape[0]
            loss_epoch += loss_intrr_epoch + loss_intrr_E_epoch + args.beta * (loss_bndry_epoch + loss_bndry_E_epoch)

        return loss_intrr_epoch, loss_bndry_epoch, loss_intrr_E_epoch, loss_bndry_E_epoch, loss_epoch
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
            
            loss_u = torch.mean(torch.pow(u_NN_smppts - u_exact_smppts.reshape(-1, 1), 2))         
            
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
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
        trainloss_intrr_epoch, trainloss_intrr_E_epoch, trainloss_bndry_epoch, trainloss_bndry_E_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
        testloss_u_epoch, testloss_grad_u_epoch = test_epoch(epoch, model, optimizer, device)    
        
        # save current and best models to checkpoint
        is_best = trainloss_epoch < trainloss_best
        if is_best:
            print('==> Saving best model ...')
        trainloss_best = min(trainloss_epoch, trainloss_best)
        helper.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'trainloss_intrr_epoch': trainloss_intrr_epoch,
                                'trainloss_intrr_E_epoch': trainloss_intrr_E_epoch,
                                'trainloss_bndry_epoch': trainloss_bndry_epoch,
                                'trainloss_bndry_E_epoch': trainloss_bndry_E_epoch,
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

        print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
        print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad(u-u_NN) = {:.4e}'.format(testloss_grad_u_epoch), "\n")
        if epoch>2000:
            if abs(test_loss_u[epoch-2000]-test_loss_u[epoch])/test_loss_u[epoch-2000]<10**-3:
                break           
    time_elapsed = time.time() - since

    # # save learning curves
    # helper.save_learncurve({'train_curve': train_loss, 'test_curve': test_loss}, curve=args.image)   

    print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
    print('*', '-' * 45, '*', "\n", "\n")
    ##############################################################################################

    # plot learning curves
    fig = plt.figure()
    plt.plot(torch.log10(torch.abs(torch.tensor(train_loss))), c = 'red', label = 'training loss' )
    plt.title('Learning Curve during Training')
    plt.legend(loc = 'upper right')
    # plt.show()
    str0='TrainCurve(sub_dom_N_%s)%d.png'%(sub_dom,iter_num)    
    fig.savefig(os.path.join(args.result,str0))
    plt.close()
    fig = plt.figure()
    plt.plot(torch.log10(torch.tensor(test_loss_u)), c = 'red', label = 'testing loss (u)' )
    plt.plot(torch.log10(torch.tensor(test_loss_grad_u)), c = 'black', label = 'testing loss (gradient u)' )
    plt.title('Learning Curve during Testing')
    plt.legend(loc = 'upper right')
    # plt.show()
    str1='TestCurve(sub_dom_N_%s)%d.png'%(sub_dom,iter_num)    
    fig.savefig(os.path.join(args.result,str1))
    plt.close()
    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> loading trained model for inference ...')

    # load trained model
    # compute NN predicution of u and gradu
    SmpPts_Test = testdata.SmpPts_Test    
    SmpPts_Test = SmpPts_Test.to(device)
    SmpPts_Test.requires_grad = True

    u_Exact = testdata.u_Exact_SmpPts.reshape(-1,1).to(device)
    grad_u_Exact = testdata.Grad_u_Exact_SmpPts.to(device)

    
    u_NN_Test = model(SmpPts_Test)
    model.zero_grad()
    gradu_NN_test = torch.autograd.grad(outputs=u_NN_Test, inputs=SmpPts_Test, grad_outputs=torch.ones_like(u_NN_Test), retain_graph=True, create_graph=True, only_inputs=True)[0]


    err = u_NN_Test - u_Exact
    errorL2 = torch.norm(err)
    errorH1 = torch.norm(grad_u_Exact - gradu_NN_test) 
    err = err.cpu().detach().numpy()

    u_NN_Test = u_NN_Test.cpu().detach().numpy()
    u_Exact = u_Exact.cpu().detach().numpy()
    grad_u_Exact = grad_u_Exact.detach().cpu().numpy()
    gradu_NN_test = gradu_NN_test.cpu().detach().numpy()

    

    print('*', '-' * 45, '*', "\n", "\n")
    io.savemat(args.result+"/u_exact_sub%s.mat"%sub_dom,{"u_ex%s"%sub_dom: u_Exact})
    io.savemat(args.result+"/gradu_exact_sub%s.mat"%sub_dom,{"gradu_Exact%s"%sub_dom: grad_u_Exact})
    io.savemat(args.result+"/u_NN_test_ite%d_sub%s.mat"%(iter_num, sub_dom), {"u_NN_sub%s"%sub_dom: u_NN_Test})    
    io.savemat(args.result+"/gradu_NN_test_ite%d_sub%s.mat"%(iter_num, sub_dom),{"grad_u_test%s"%sub_dom: gradu_NN_test})
    io.savemat(args.result+"/err_test_ite%d_sub%s.mat"%(iter_num, sub_dom), {"pointerr%s"%sub_dom: err})
    torch.save(model.state_dict(), args.result + "/model_ite-%d-%s.pth"%(iter_num, sub_dom))
    return model, errorL2, errorH1
    ##############################################################################################