First, we consider the benchmark Poisson problem in two dimension, that is,
```math
\begin{equation}
\begin{array}{cl}
-\Delta u(x,y)  = 4 \pi^2 \sin(2 \pi x)  (2 \cos(2 \pi y) - 1)  \ & \text{in}\ \Omega=(0,1)^2,\\
u(x,y) = 0\ \ & \text{on}\ \partial \Omega,
\end{array}
\end{equation}
```
where the true solution is given by $u(x,y) = \sin(2\pi x)(\cos(2\pi y)-1)$ and the interface $\Gamma=\partial\Omega_1\cap\partial\Omega_2$ is a straight line segment from $(0.5,0)$ to $(0.5,1)$. 

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/45867af9-580d-4015-ab87-6935ec7aa7a3)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$ |



It is noteworthy that this solution reaches local extrema at $(0.5,0.5)$, thereby deviations in estimating the Neumann trace at and near the extreme point can create a cascading effect in the convergence of outer iterations, which differs from other examples that have simple gradients on the interface.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/3f050a5b-10eb-4fc4-9cf3-de32d7a6f2d5)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DN-PINNs on the test dataset.* |


But our proposed method, DNLA (PINNs) and DNLA (Ritz), can still work since the variational principle is used.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/0bef6490-812e-43ac-bc23-01b1d0a1f004)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/76b05d6d-d4bf-4b4d-852c-d1d6eaad60ca)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (Ritz) on the test dataset.* |


