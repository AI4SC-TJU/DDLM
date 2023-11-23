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

Note that in this case, the extension operator is different from ex3, which is shown as follows

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/54fcfa6b-2fb9-46ed-86c6-47d725b10313)|
|:--------------------------------------------------------------:|
| *Illustration of neural network extension operators for 4 subdomains.* |

In this case, DN-PINNs fails to predict exact solution.

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
To acquire the data represented in the first row of Table SM1, execute the script `task1-DN-PINNs-2D-4prob-highcontrast.py`.

## Table SM1 - DNLM(PINN) Result
For the results displayed in the Table SM1, execute the script `task2-DN-DNLA_PINNs-2D-4prob-highcontrast.py` to generate the corresponding data.

## Table SM1 - DNLM(Ritz) Result
For the data displayed in the Table SM1, execute the script `task3-DN-DNLA_Ritz-2D-4prob-highcontrast.py` to generate the corresponding results.

## Figure Generation:
Utilize MATLAB and execute the script `task4_show_results.m` to create the graphical representation associated with the data obtained from the previous steps.

