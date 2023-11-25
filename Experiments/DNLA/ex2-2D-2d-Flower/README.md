In this subsection, we tackle the Poisson problem with a more intricate interface geometry to demonstrate the mesh-free character of deep learning solvers. The problem is described as follows:
```math
\begin{equation}
\begin{array}{cl}
-\Delta u(x,y)  = \displaystyle-35\sin(12 \arctan(\frac{y}{x})) + 2 - 9\sqrt{x^2 + y^2}  \ & \text{in}\ \Omega=(0.01,1.01)\times (0,1),\\
u(x,y) = g(x,y)\ \ & \text{on}\ \partial \Omega,
\end{array}
\end{equation}
```
where the exact solution $u(x,y) = (x^2 + y^2)(\sqrt{x^2 + y^2} - r_f(x,y))$, the interface is a curved flower line, and the flower function $r_f(x,y)$ reads
```math
\begin{equation*}
	r_f(x,y) = 0.5 + 0.25 \sin(12 \arctan(\frac{y}{x})).
\end{equation*}
```
Our learning algorithm can easily handle such irregular shapes, while finite difference or finite element methods necessitates meticulous treatment of edges and corners.

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/00192b1b-7616-4d3f-9ed0-273208b6be0f)|
|:--------------------------------------------------------------:|
|From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$|


The DN-PINNs approach fails to converge to the true solution as the network solution of Dirichlet subproblem is prone to return inaccurate Neumann traces at interface.
|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/dd7d6036-70fe-484f-a174-cf05a32c62ac)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DN-PINNs on the test dataset.* |


In contrast, by solving the Neumann subproblem through our compensated deep Ritz method, the numerical results demonstrate that our DNLA (PINNs) can obtain a satisfactory approximation to the exact solution, which also avoids the meshing procedure that is often challenging for problems with complex interfaces. 

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/d20c8753-aa7a-4b39-9701-e782827fa3a3)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/9313cf33-c7d1-4973-a481-46fb4c000d2c)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (Ritz) on the test dataset.* |





