# Motivation \#1: DtN Map for Dirichlet-Neumann Algorithm

With boundary conditions being included as soft constraints in the training loss function, the trained network solution of Dirichlet problem is often observed to furnish more precision inside the domain, rather than at the boundary.

| ![WechatIMG1159](https://github.com/AI4SC-TJU/DDLM/assets/131741694/6b3fb203-dd8d-4f54-88df-d36425e1973e)              |
|:--------------------------------------------------------------:|
| *Network solutions of Dirichlet subproblem using different structures and optimization tricks, together with their error profiles.* |

This pattern of error distribution, i.e., higher precision is attained inside the domain rather than at the boundary, also aligns with various other studies. A question naturally arises: Is it feasible to utilize the interior solution for data exchange between neighbouring subproblems?








# Motivation \#2: Weight Imbalance for Robin-Robin Algorithm

With boundary conditions being included as soft constraints in the training loss function, the trained network solution of Dirichlet subproblem is often observed to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion between neighbouring subdomains.

| ![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/c3895adc-a2da-4a16-b67a-668dcc4851e1)             |
|:--------------------------------------------------------------:|
| *Network solutions of Robin subproblem with different values of $`\kappa_1`$, together with their error profiles.* |

Based on the observation that the network solution of Dirichlet and Robin subproblem often exhibit higher errors at the boundary compared to its interior domain, we can draw a conclusion that d


To answer this question, We consider employing variational principles, rather than a direct flux transmission along subdomain interfaces, when developing the domain decomposition learning method.

Let's see more details about 
On the one hand, the Neumann subproblem within the DN-PINNs strategy is also solved using PINNs, i.e.,
```math
\begin{align*}
    \hat{u}_2^{[k]} = \mathop{\text{arg\,min}}_{u_2} \int_{\Omega_2} | \Delta u_2 + f |^2\,dx + \beta\left(\int_{\textcolor{red}{\Gamma}} \Big|  \frac{\partial u_2}{\partial \mathbf{n}_2} - \textcolor{red}{\frac{\partial \hat{u}^{[k]}_1}{\partial \mathbf{n}_2}} \Big|^2\,ds + \int_{\partial\Omega\cap\partial\Omega_2} |u_2|^2\,ds \right),
\end{align*}
```
where the boundary data (marked in red color) relies on the Neumann trace of Dirichlet subproblem $`\textcolor{red}{\nabla \hat{u}^{[k]}_1 |_\Gamma}`$. On the other hand, the loss function of our proposed method is defined as
```math
\begin{align*}
    	\hat{u}_2^{[k]} = \mathop{\text{arg\,min}}_{\hat{u}_2} \int_{\Omega_2} \Big( \frac12 | \nabla \hat{u}_2 |^2  - f \hat{u}_2 \Big) dx + \int_{\textcolor{blue}{\Omega_1}} \Big( \textcolor{blue}{\nabla \hat{u}_1^{[k]}}\cdot \nabla \hat{u}_2  - f \hat{u}_2 \Big) dx + \beta \int_{\partial\Omega} |\hat{u}_2|^2\,ds,
\end{align*}
```
where the exchanged data (marked in blue color) is represented using the interior solution $`\textcolor{blue}{\nabla \hat{u}^{[k]}_1 |_{\Omega_1}}`$.








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

