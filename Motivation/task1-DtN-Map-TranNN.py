import tensorflow.compat.v1 as tf
import numpy as np
import os
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import seaborn as sns
import argparse
from Poisson2D_model_tf import Sampler, Poisson2D
if __name__ == '__main__':
    ## parser arguments
    parser = argparse.ArgumentParser(description='Physics-Informed Neural Network for Poisson Subproblem with Robin Interface Condition')

    # path for saving results
    parser.add_argument('-r', '--result', default='Results/Poisson/M1/Square2D/Dirichlet/simulation-test-M1', type=str, metavar='PATH', help='path to save checkpoint')
    parser.add_argument('-m', '--method', default='M1', type=str, metavar='PATH', help='the method you choose')
    # optimization options
    parser.add_argument('--num_epochs', default=100001, type=int, metavar='N', help='number of total epochs to run')

    # network architecture
    parser.add_argument('--depth', type=int, default=8, help='network depth')
    parser.add_argument('--width', type=int, default=50, help='network width')

    # datasets options 
    parser.add_argument('--num_intrr_pts', type=int, default=20000, help='total number of interior sampling points')
    parser.add_argument('--num_bndry_pts_D', type=int, default=20000, help='total number of sampling points at each line segment of Dirichlet boundary')
    parser.add_argument('--num_bndry_pts_G', type=int, default=5000, help='total number of sampling points at inteface')
    parser.add_argument('--num_test_pts', type=int, default=100, help='number of sampling points for each dimension during testing')

    # Dirichlet-Neumann algorithm setting
    parser.add_argument('--alpha', type=float, default=100, help='alpha of the inner subproblem')    
    parser.add_argument('--alpha_R', type=float, default=1, help='alpha of the inner subproblem')
    parser.add_argument('--alpha_B', type=float, default=1, help='alpha of the outer subproblem')
    parser.add_argument('--r0', type=float, default=0.5, help='radius of the sphere in the square')
    parser.add_argument('--max_ite_num', type=int, default=16, help='maximum number of outer iterations')
    parser.add_argument('--dim_prob', type=int, default=2, help='dimension of the sub-problem to be solved')

    args = parser.parse_args()   
    File_Path = args.result
    if not os.path.exists(File_Path):
        os.makedirs(File_Path) 
    a_1 = 2
    a_2 = 2
    
    def u(x, a_1, a_2):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * (np.cos(a_2 * np.pi * x[:, 1:2])- 1)

    def u_x(x, a_1, a_2):
        return a_1 * np.pi * np.cos(a_1 * np.pi * x[:, 0:1]) * (np.cos(a_2 * np.pi * x[:, 1:2])- 1)

    def u_y(x, a_1, a_2):
        return -a_2 * np.pi * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_xx(x, a_1, a_2):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * (np.cos(a_2 * np.pi * x[:, 1:2])- 1)
    def u_yy(x, a_1, a_2):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.cos(a_2 * np.pi * x[:, 1:2])
    
    def grad_u(x, a_1, a_2):
        return np.concatenate([u_x(x, a_1, a_2), u_y(x, a_1, a_2)], axis = 1)
    
    # Forcing
    def f(x, a_1, a_2, k):
        return -u_xx(x, a_1, a_2) - u_yy(x, a_1, a_2)

    def operator(u, x1, x2, k, sigma_x1=1.0, sigma_x2=1.0):
        u_x1 = tf.gradients(u, x1)[0] / sigma_x1
        u_x2 = tf.gradients(u, x2)[0] / sigma_x2
        u_xx1 = tf.gradients(u_x1, x1)[0] / sigma_x1
        u_xx2 = tf.gradients(u_x2, x2)[0] / sigma_x2
        residual = -u_xx1 - u_xx2
        return residual

    # Parameter
    k = 1.0

    # Domain boundaries
    bc1_coords = np.array([[0.0, 0.0],
                           [0.5, 0.0]])
    bc2_coords = np.array([[0.5, 0.0],
                           [0.5, 1.0]])
    bc3_coords = np.array([[0.5, 1.0],
                           [0.0, 1.0]])
    bc4_coords = np.array([[0.0, 1.0],
                           [0.0, 0.0]])

    dom_coords = np.array([[0.0, 0.0],
                           [0.5, 1.0]])

    # Create initial conditions samplers
    ics_sampler = None

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a_1, a_2, k), name='Forcing')

    # Define model
    mode = args.method            # Method: 'M1', 'M2', 'M3', 'M4'
    stiff_ratio = False    # Log the eigenvalues of Hessian of losses
    #layers = [2, 50, 50, 50, 1]
    layers = []
    for i in range(args.depth):
        if i==0:
            layers.append(args.dim_prob)
        else:
            if i==args.depth-1:
                layers.append(1)
            else:
                layers.append(args.width)

    model = Poisson2D(args, layers, operator, ics_sampler, bcs_sampler, res_sampler, k, mode, stiff_ratio)

    # Train model
    model.train(nIter=args.num_epochs, batch_size=128)

    # Test data
    nn = 100
    x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    x1, x2 = np.meshgrid(x1, x2)
    X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

    # Exact solution
    u_star = u(X_star, a_1, a_2)
    f_star = f(X_star, a_1, a_2, k)
    grad_u_star_x = u_x(X_star, a_1, a_2)
    grad_u_star_y = u_y(X_star, a_1, a_2)

    # Predictions
    u_pred = model.predict_u(X_star)
    f_pred = model.predict_r(X_star)
    grad_u_pred = model.predict_grad_u(X_star)


    # Relative error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_f = np.linalg.norm(f_star - f_pred, 2) / np.linalg.norm(f_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_u: {:.2e}'.format(error_f))

    ### Plot ###

    # Exact solution & Predicted solution
    # Exact soluton
    U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')
    F_star = griddata(X_star, f_star.flatten(), (x1, x2), method='cubic')
    grad_U_star_x = griddata(X_star, grad_u_star_x.flatten(), (x1, x2), method='cubic')
    grad_U_star_y = griddata(X_star, grad_u_star_y.flatten(), (x1, x2), method='cubic')
    grad_U_star = np.concatenate((grad_U_star_x.reshape(-1,1), grad_U_star_y.reshape(-1,1)), axis=1)

    # Predicted solution
    U_pred = griddata(X_star, u_pred.flatten(), (x1, x2), method='cubic')
    F_pred = griddata(X_star, f_pred.flatten(), (x1, x2), method='cubic')
    grad_U_pred_x = griddata(X_star, grad_u_pred[:,0].reshape(-1,1).flatten(), (x1, x2), method='cubic')
    grad_U_pred_y = griddata(X_star, grad_u_pred[:,1].reshape(-1,1).flatten(), (x1, x2), method='cubic')
    grad_U_pred = np.concatenate((grad_U_pred_x.reshape(-1,1), grad_U_pred_y.reshape(-1,1)), axis = 1)


    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Exact $u(x)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Predicted $u(x)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(args.result+"/solution.png")
    plt.show()

    fig_0 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, np.linalg.norm(grad_U_star, axis=1).reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Exact $||\nabla u(x)||_{l^2}$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, np.linalg.norm(grad_U_pred, axis=1).reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Predicted $||\nabla u(x)||_{l^2}$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(np.linalg.norm(grad_U_star, axis=1) - np.linalg.norm(grad_U_pred, axis=1)).reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(args.result+"/gradient_solution.png")
    plt.show()

    fig_0 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, grad_U_star_x, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Exact $\frac{\partial u}{\partial x}$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, grad_U_pred_x.reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Predicted $\frac{\partial u}{\partial x}$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(grad_U_star_x-grad_U_pred_x).reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(args.result+"/gradient_partial_x.png")
    plt.show()

    fig_10 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, grad_U_star_y, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Exact $\frac{\partial u}{\partial y}$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, grad_U_pred_y.reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Predicted $\frac{\partial u}{\partial y}$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(grad_U_star_y-grad_U_pred_y).reshape(nn, nn), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(args.result+"/gradient_partial_y.png")
    plt.show()

    # Residual loss & Boundary loss
    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log

    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_res, label='$\mathcal{L}_{r}$')
    ax.plot(loss_bcs, label='$\mathcal{L}_{u_b}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.result+"/loss.png")

    # Adaptive Constant
    adaptive_constant = model.adpative_constant_log

    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant, label='$\lambda_{u_b}$')
    ax.set_xlabel('iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.result+"/Adaptive_Constant.png")

    # Gradients at the end of training
    data_gradients_res = model.dict_gradients_res_layers
    data_gradients_bcs = model.dict_gradients_bcs_layers

    gradients_res_list = []
    gradients_bcs_list = []

    num_hidden_layers = len(layers) - 1
    for j in range(num_hidden_layers):
        gradient_res = data_gradients_res['layer_' + str(j + 1)][-1]
        gradient_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]

        gradients_res_list.append(gradient_res)
        gradients_bcs_list.append(gradient_bcs)

    cnt = 1
    fig_4 = plt.figure(4, figsize=(13, 4))
    for j in range(num_hidden_layers):
        ax = plt.subplot(1, args.depth - 1, cnt)
        ax.set_title('Layer {}'.format(j + 1))
        ax.set_yscale('symlog')
        gradients_res = data_gradients_res['layer_' + str(j + 1)][-1]
        gradients_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]
        data = {r'$\nabla_\theta \lambda_{u_b} \mathcal{L}_{u_b}$':gradients_res, r'$\nabla_\theta \mathcal{L}_r$':gradients_bcs}
        
        data = pd.DataFrame(data)
        
        sns.displot(data=data, kind="kde")
        #sns.displot(gradients_res, kind="kde",label=r'$\nabla_\theta \mathcal{L}_r$')
      
        #ax.get_legend().remove()
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([0,100])
        cnt += 1
    handles, labels = ax.get_legend_handles_labels()
    fig_4.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
               borderaxespad=0, bbox_transform=fig_4.transFigure, ncol=2)
    plt.tight_layout()
    plt.savefig(args.result+"/gradient.png")

    # Eigenvalues if applicable
    if stiff_ratio:
        eigenvalues_list = model.eigenvalue_log
        eigenvalues_bcs_list = model.eigenvalue_bcs_log
        eigenvalues_res_list = model.eigenvalue_res_log
        eigenvalues_res = eigenvalues_res_list[-1]
        eigenvalues_bcs = eigenvalues_bcs_list[-1]

        fig_5 = plt.figure(5)
        ax = fig_5.add_subplot(1, 1, 1)
        ax.plot(eigenvalues_res, label='$\mathcal{L}_r$')
        ax.plot(eigenvalues_bcs, label='$\mathcal{L}_{u_b}$')
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        ax.set_yscale('symlog')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.result+"/Eigenvalues.png")
    ite_index = 1
    
    err = U_pred - U_star
    io.savemat(args.result+"/u_exact.mat",{"u_exact": U_star.reshape(-1,1)})
    io.savemat(args.result+"/u_NN_ite%d_test.mat"%ite_index,{"u_NN_test": U_pred.reshape(-1,1)})
    io.savemat(args.result+"/gradu_exact.mat",{"grad_u_exact": grad_U_star.reshape(-1,2)})    
    io.savemat(args.result+"/gradu_NN_ite%d_test.mat"%ite_index,{"grad_u_NN_test": grad_U_pred.reshape(-1,2)})
    io.savemat(args.result+"/err_test_ite%d.mat"%ite_index, {"pointerr": err.reshape(-1,1)})
    io.savemat(args.result+"/trainloss%d.mat"%ite_index, {"trainloss": model.train_loss_log})
    io.savemat(args.result+"/testloss%d.mat"%ite_index, {"testloss": model.test_loss_log})
    io.savemat(args.result+"/testloss_x%d.mat"%ite_index, {"testloss_x": model.test_loss_x_log})
    io.savemat(args.result+"/testloss_y%d.mat"%ite_index, {"testloss_y": model.test_loss_y_log})