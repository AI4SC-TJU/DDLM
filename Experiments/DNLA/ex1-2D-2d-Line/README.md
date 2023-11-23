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

| ![exact-ex1](https://github.com/AI4SC-TJU/DDLM/assets/93070782/ab5798e2-7179-4f23-9021-df8243e31bcb)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $u(x,y)$, and its partial derivatives $\partial_x u(x,y)$, $\partial_y u(x,y)$* |



It is noteworthy that this solution reaches local extrema at $(0.5,0.5)$, thereby deviations in estimating the Neumann trace at and near the extreme point can create a cascading effect in the convergence of outer iterations, which differs from other examples that have simple gradients on the interface.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/30e96eff-5e05-4261-8798-b7adc02995e3)|
|:--------------------------------------------------------------:|
| *Iterative solutions $\hat{u}^{[k]}(x,y)$ using DN-PINNs on the test dataset.* |


But our proposed method, DNLA (PINNs) and DNLA (Ritz), can still work since the variational principle is used.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/a0295d1a-0c66-495a-9965-e6944aefe41e)|
|:--------------------------------------------------------------:|
| *Iterative solutions $\hat{u}^{[k]}(x,y)$ using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/e1767d6e-121d-4b6a-b9c9-2e0640bb7f75)|
|:--------------------------------------------------------------:|
| *Iterative solutions $\hat{u}^{[k]}(x,y)$ using DNLA (Ritz) on the test dataset.* |


