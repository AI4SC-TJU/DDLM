# Motivation \#1: DtN Map for Dirichlet-Neumann Algorithm

With boundary conditions being included as soft constraints in the training loss function, the trained network solution of Dirichlet problem is often observed to furnish more precision inside the domain, rather than at the boundary.

| ![WechatIMG1159](https://github.com/AI4SC-TJU/DDLM/assets/131741694/6b3fb203-dd8d-4f54-88df-d36425e1973e)              |
|:--------------------------------------------------------------:|
| *Network solutions of Dirichlet subproblem using different structures and optimization tricks, together with their error profiles.* |

This pattern of error distribution, i.e., higher precision is attained inside the domain rather than at the boundary, also aligns with various other studies. A question naturally arises: Is it feasible to utilize the interior solution for data exchange between neighbouring subproblems?








# Motivation \#2: Weight Imbalance for Robin-Robin Algorithm

With boundary conditions being included as soft constraints in the training loss function, the trained network solution of Dirichlet subproblem is often observed to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion between neighbouring subdomains.


# Introduction
This is the code for the figures shown in Remark 2.1 and Remark 2.2
## Table 1 - FCNN results
To obtain the results shown in Table 1 for the FCNN, execute the script `Overfit-Dirichlet.py`
## Table 1 - Transformer Network Results
For the results shown in Table 1 related to the transformer network, execute the script `Poisson2D.py`.
## Table 2 - results
To generate the results presented in Table 2, run the script `Overfit-Robin.py`. Additionally, set the parameter "alpha_left" to 1 for the first simulation and 1000 for the second simulation.
## Figure Generation:
Utilize MATLAB to execute the script `plot_overfit_baseline.m` in order to generate the figure associated with the data obtained from the previous steps.

