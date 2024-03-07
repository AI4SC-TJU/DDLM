Here, we consider an elliptic interface problem in two dimension with high-contrast coefficients, 
```math
\begin{equation}
\begin{array}{cl}
-\nabla \cdot \left( c(x,y) \nabla u(x,y)  \right) = 32 \pi^2 \sin(4\pi x)\cos(4\pi y)\ \ & \text{in}\ \Omega=(0,1)^2,\\
u(x,y) = 0\ \ & \text{on}\ \partial \Omega, 
\end{array}
\end{equation}
```
where the computational domain is decomposed into four isolated subdomains is shown as flollows, 

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/5c881b55-2782-46f0-860a-eab6cf2103b2)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$ * |


the exact solution is given by $`u(x,y) = \sin(4\pi x) \sin(4\pi y) / c(x,y)`$, and the coefficient $`c(x,y)`$ is piecewise constant with respect to the partition of domain
```math
\begin{equation*}
c(x,y) = \left\{
\begin{array}{cl}
1 \ & \text{in}\ \Omega_1\cup\Omega_3,\\
100\ \ & \text{in}\ \Omega_2\cup\Omega_4.
\end{array}\right.
\end{equation*}
```
Here, we choose $`h^{[0]}=100\cos(100\pi x)\cos(100\pi y)+100xy`$ as the initial guess, and the numerical results using DNLA are depicted as follows. Clearly, our method can facilitate the convergence of outer iterations in the presence of inaccurate flux estimations.

In this case, DN-PINNs fails to predict exact solution.

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/c6c9745a-48cd-4a65-9cea-3dd6758e85d8)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DN-PINNs on the test dataset.* |


But our proposed method, DNLA (PINNs) and DNLA (Ritz), can still work since the variational principle is used.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/0e171466-8756-4bbe-86cc-a7291e438ea8)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/8341dc5d-c3af-4301-aaac-8174c26dbec9)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (Ritz) on the test dataset.* |




