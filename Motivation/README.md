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

