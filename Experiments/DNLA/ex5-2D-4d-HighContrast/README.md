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

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/6b7c5688-5534-4258-b6fb-2dc1a60bc690)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$ |


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


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/02bbfdc4-9bb0-47d4-a3ad-2e4a9c0bd861)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DN-PINNs on the test dataset.* |


But our proposed method, DNLA (PINNs) and DNLA (Ritz), can still work since the variational principle is used.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/19d627b8-34ae-4f7d-95ce-799e802f907a)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/2d6d9091-9057-4c00-9841-718e15e40892)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DNLA (Ritz) on the test dataset.* |






# Instructions for Generating Table SM1 in Our Supplement Materials
## Table SM1 - DN-PINNs Result
To acquire the data represented in the first row of Table SM1, execute the script `DN-PINNs-4prob-2D-highcontrast-4NN.py`.

## Table SM1 - DNLM(PINN) Result
For the results displayed in the Table SM1, execute the script `DNLM-4prob-2D-Compensent-highcontrast-4NN.py` to generate the corresponding data.

## Table SM1 - DNLM(Ritz) Result
For the data displayed in the Table SM1, execute the script `DNLM-4prob-2D-Compensent-Ritz-highcontrast-4NN.py` to generate the corresponding results.

## Figure Generation:
Utilize MATLAB and execute the script `plot_DNLM_4d_NN_1by1.m` to create the graphical representation associated with the data obtained from the previous steps.

