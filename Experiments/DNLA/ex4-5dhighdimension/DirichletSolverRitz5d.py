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
from Datasets.Square5Dcos import Sample_Points, Exact_Solutionplus
from Utils import helper
# create neural network surrogate model
from Models.FcNet import FcNet
plt.switch_backend('agg')

# load data from two datasets within the same loop
from itertools import cycle





def DirichletSolverRitz(args, traindata_bndry_G, dataloader_bndry_G, iter_num, sub_dom=1):
    print("pytorch version", torch.__version__, "\n")

    dim_prob = args.dim_prob

    ##############################################################################################
    ## hyperparameter configuration
    ## ------------------------- ##
    batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
    batchsize_bndry_pts_D = (2*args.dim_prob-1)*args.num_bndry_pts_D // args.num_batches
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
            
            self.SmpPts_Interior = Sample_Points.SmpPts_Interior_Square10D(num_intrr_pts, sub_dom, dim_prob)
            self.f_Exact_SmpPts = Exact_Solutionplus.f_Exact_Square10D(self.SmpPts_Interior, dim_prob)        
                
        def __len__(self):
            return len(self.SmpPts_Interior)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Interior[idx]
            f_SmpPt = self.f_Exact_SmpPts[idx]

            return [SmpPt, f_SmpPt]
        
    # training dataset for sample points at the Dirichlet boundary
    class TraindataBoundaryDirichlet(Dataset):    
        def __init__(self, num_bndry_pts_D, sub_dom, dim_prob):         
            
            self.SmpPts_Bndry_D = Sample_Points.SmpPts_Boundary_Square10D(num_bndry_pts_D, sub_dom, dim_prob)
            self.g_SmpPts = Exact_Solutionplus.g_Exact_Square10D(self.SmpPts_Bndry_D, dim_prob)        
            
        def __len__(self):
            return len(self.SmpPts_Bndry_D)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Bndry_D[idx]
            g_SmpPt = self.g_SmpPts[idx]

            return [SmpPt, g_SmpPt]    
        

    # testing dataset for equidistant sample points over the sub-domain to be solved
    class Testdata(Dataset):    
        def __init__(self, num_test_pts, sub_dom, dim_prob): 
            
            self.SmpPts_Test = Sample_Points.SmpPts_Test_Square10D(num_test_pts, sub_dom, dim_prob)
            self.u_Exact_SmpPts = Exact_Solutionplus.u_Exact_Square10D(self.SmpPts_Test, dim_prob)        
            self.Gradu_Exact_SmpPts = Exact_Solutionplus.Gradu_Exact_Square10D(self.SmpPts_Test, dim_prob)   
                    
        def __len__(self):
            return len(self.SmpPts_Test)
        
        def __getitem__(self, idx):
            SmpPt = self.SmpPts_Test[idx]
            u_Exact_SmpPt = self.u_Exact_SmpPts[idx]
            Gradu_Exact_SmpPt = self.Gradu_Exact_SmpPts[idx]

    
            return [SmpPt, u_Exact_SmpPt, Gradu_Exact_SmpPt]             
    # create training and testing datasets
    traindata_intrr = TraindataInterior(args.num_intrr_pts, sub_dom,  dim_prob)
    traindata_bndry_D = TraindataBoundaryDirichlet(args.num_bndry_pts_D, sub_dom,  dim_prob)
    testdata = Testdata(args.num_test_pts, sub_dom,  dim_prob)
    # define dataloader 
    dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
    dataloader_bndry_D = DataLoader(traindata_bndry_D, batch_size=batchsize_bndry_pts_D, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(testdata, batch_size=args.num_test_pts**dim_prob, shuffle=False, num_workers=0)

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
            
            smppts_interior, f_smppts = data_intrr
            smppts_bndry_D, g_D_smppts = data_bndry_D
            smppts_bndry_G = data_bndry_G[0]
            g_1_select = data_bndry_G[1]
            # send training data to device
            smppts_interior = smppts_interior.to(device)
            f_smppts = f_smppts.to(device).reshape(-1,1)
            smppts_bndry_D = smppts_bndry_D.to(device)
            g_D_smppts = g_D_smppts.to(device)
            smppts_bndry_G = smppts_bndry_G.to(device)
            smppts_bndry_G.requires_grad = True

            smppts_interior.requires_grad = True

            # forward pass to obtain NN prediction of u(x)
            u_NN_interior = model(smppts_interior)
            u_NN_bndry_D = model(smppts_bndry_D)
            u_NN_bndry_G = model(smppts_bndry_G)
            # zero parameter gradients and then compute NN prediction of gradient u(x)
            model.zero_grad()
            grad_u_NN_interior = torch.autograd.grad(outputs=u_NN_interior, inputs=smppts_interior, grad_outputs=torch.ones_like(u_NN_interior), retain_graph=True, create_graph=True, only_inputs=True)[0]        
            loss_interior = torch.mean(0.5 * torch.sum(torch.pow(grad_u_NN_interior, 2), dim=1) - f_smppts.reshape(1, -1) * torch.squeeze(u_NN_interior))
            loss_bndry_D = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_D) - g_D_smppts.reshape(1, -1), 2))
            loss_bndry_R = torch.mean(torch.pow(torch.squeeze(u_NN_bndry_G) - g_1_select.reshape(1, -1), 2))


            # construct mini-batch loss function and then perform backward pass
   

            loss_minibatch = loss_interior + args.beta * (loss_bndry_D + loss_bndry_R)
            # zero parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss_minibatch.backward()
            # parameter update
            optimizer.step()     

            # integrate loss over the entire training datset
            loss_intrr_epoch += loss_interior.item() * smppts_interior.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
            loss_bndry_D_epoch += loss_bndry_D.item() * smppts_bndry_D.size(0) / traindata_bndry_D.SmpPts_Bndry_D.shape[0]
            loss_bndry_G_epoch += loss_bndry_R.item() * smppts_bndry_G.size(0) / traindata_bndry_G.SmpPts_Bndry_G.shape[0]
            loss_epoch += loss_intrr_epoch + loss_bndry_G_epoch + args.beta * loss_bndry_D_epoch  

        return loss_intrr_epoch, loss_bndry_D_epoch, loss_bndry_G_epoch, loss_epoch
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> creating testing model ...')
    print('*', '-' * 45, '*', "\n", "\n")

    def test_epoch(epoch, model, optimizer, device):
        
        # set model to testing mode
        model.eval()

        epoch_loss_u, epoch_loss_gradu = 0, 0
        for smppts_test, u_exact_smppts, gradu_exact_smppts in dataloader_test:
            
            # send inputs, outputs to device
            smppts_test = smppts_test.to(device)
            u_exact_smppts = u_exact_smppts.to(device)  
            gradu_exact_smppts = gradu_exact_smppts.to(device)
            
            smppts_test.requires_grad = True
            
            # forward pass and then compute loss function for approximating u by u_NN
            u_NN_smppts = model(smppts_test) 
            
            loss_u = torch.mean(torch.pow(torch.squeeze(u_NN_smppts) - u_exact_smppts, 2))         
            
            # backward pass to obtain gradient and then compute loss function for approximating grad_u by grad_u_NN
            model.zero_grad()
            gradu_NN_smppts = torch.autograd.grad(outputs=u_NN_smppts, inputs=smppts_test, grad_outputs=torch.ones_like(u_NN_smppts), retain_graph=True, create_graph=True, only_inputs=True)[0]
            
            loss_gradu = torch.norm(gradu_NN_smppts - gradu_exact_smppts)
            # integrate loss      
            epoch_loss_u += loss_u.item()         
            epoch_loss_gradu += loss_gradu.item()  
        
        return epoch_loss_u, epoch_loss_gradu
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
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.5)

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: {}'.format(device), "\n")
    model = model.to(device)

    # train and test 
    train_loss, test_loss_u, test_loss_gradu = [], [], []
    trainloss_best = 1e10
    since = time.time()
    for epoch in range(args.num_epochs):        
        print('Epoch {}/{}'.format(epoch, args.num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))              
        # execute training and testing
        trainloss_intrr_epoch, trainloss_bndry_D_epoch, trainloss_bndry_G_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device, sub_dom)
        testloss_u_epoch, testloss_gradu_epoch = test_epoch(epoch, model, optimizer, device)    
        
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
                                'testloss_gradu_x_epoch': testloss_gradu_epoch,
                                'trainloss_best': trainloss_best,
                                'optimizer': optimizer.state_dict(),
                            }, is_best, checkpoint=args.result)   
        # save training process to log file
              
        # adjust learning rate according to predefined schedule
        schedular.step()

        # print results
        train_loss.append(trainloss_epoch)
        test_loss_u.append(testloss_u_epoch)
        test_loss_gradu.append(testloss_gradu_epoch)
        print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
        print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), '  Grad_x(u-u_NN) = {:.4e}'.format(testloss_gradu_epoch), "\n")
        
    time_elapsed = time.time() - since
    ## plot and save training and testing curves 
    fig = plt.figure()
    plt.plot(torch.tensor(train_loss), c = 'red', label = 'training loss' )
    plt.title('Learning Curve during Training')
    plt.legend(loc = 'upper right')
    # plt.show()
    if sub_dom==1:
        str1='TrainingCurve%dsub_dom%d.png'%(iter_num,sub_dom)
    else:
        str1='TrainingCurve%dsub_dom%d.png'%(iter_num,sub_dom)      
    fig.savefig(os.path.join(args.result,str1))
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(torch.log10(torch.tensor(test_loss_u)), c = 'red', label = 'testing loss (u)' )
    plt.plot(torch.log10(torch.tensor(test_loss_gradu)), c = 'blue', label = 'testing loss (gradu)' )
    plt.title('Learning Curve during Testing')
    plt.legend(loc = 'upper right')
    # plt.show()
    if sub_dom==1:
        str1='TestCurve%dsub_dom%d.png'%(iter_num,sub_dom)
    else:
        str1='TestCurve%dsub_dom%d.png'%(iter_num,sub_dom)     
    fig.savefig(os.path.join(args.result,str1))
    plt.close(fig)
    # # save learning curves
    # helper.save_learncurve({'train_curve': train_loss, 'test_curve': test_loss}, curve=args.image)   

    print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
    print('*', '-' * 45, '*', "\n", "\n")
    ##############################################################################################


    ##############################################################################################
    print('*', '-' * 45, '*')
    print('===> loading trained model for inference ...')

    # load trained model
    # compute NN predicution of u and gradu
    SmpPts_Test = testdata.SmpPts_Test    
    SmpPts_Test = SmpPts_Test.to(device)
    SmpPts_Test.requires_grad = True

    u_Exact = testdata.u_Exact_SmpPts.reshape(-1,1).to(device)

    grad_u_Exact = testdata.Gradu_Exact_SmpPts.to(device)

    u_NN_Test = model(SmpPts_Test)
    model.zero_grad()
    grad_u_NN_test = torch.autograd.grad(outputs=u_NN_Test, inputs=SmpPts_Test, grad_outputs=torch.ones_like(u_NN_Test), retain_graph=True, create_graph=True, only_inputs=True)[0]


    err = u_NN_Test - u_Exact
    errorL2 = torch.norm(err)
    errorH1 = torch.norm(grad_u_Exact - grad_u_NN_test)
    err = err.cpu().detach().numpy()

    u_NN_Test = u_NN_Test.cpu().detach().numpy()
    u_Exact = u_Exact.cpu().detach().numpy()

    grad_u_NN_test = grad_u_NN_test.cpu().detach().numpy()
    grad_u_Exact = grad_u_Exact.detach().cpu().numpy()

    print('*', '-' * 45, '*', "\n", "\n")
    io.savemat(args.result+"/u_exact_sub%d.mat"%sub_dom,{"u_ex%d"%sub_dom: u_Exact})
    io.savemat(args.result+"/grad_u_exact_sub%d.mat"%sub_dom,{"grad_u_ex%d"%sub_dom: grad_u_Exact})
    io.savemat(args.result+"/u_NN_test_ite%d_sub%d.mat"%(iter_num, sub_dom), {"u_NN_sub%d"%sub_dom: u_NN_Test})    
    io.savemat(args.result+"/grad_u_NN_test_ite%d_sub%d.mat"%(iter_num, sub_dom),{"grad_u_test%d"%sub_dom: grad_u_NN_test})
    io.savemat(args.result+"/err_test_ite%d_sub%d.mat"%(iter_num, sub_dom), {"pointerr%d"%sub_dom: err})

    return model, errorL2, errorH1
    ##############################################################################################

