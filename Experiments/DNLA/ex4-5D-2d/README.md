As is well known, another key and desirable advantage of using deep learning solvers is that they can tackle difficulties induced by the curse of dimensionality. To this end, we consider a Poisson problem in five dimension, i.e.,
```math
\begin{equation}
\begin{array}{cl}
\displaystyle -\Delta u(x_1,\cdots,x_5)  =  4\pi^2\sum\limits_{i=1}^5 \sin (x_i)\ & \text{in}\ \Omega = (0,1)^5, \\
u(x_1,\cdots,x_5) = 0\ \ & \text{on}\ \partial \Omega,
\end{array}
\end{equation}
```
where the exact solution $`u(x_1,\cdots,x_5) = \sum\limits_{i=1}^5 \sin (x_i)`$, and the domain is decomposed into two subdomains $`\Omega_1= \big\{(x_1,\cdots,x_5)\in\Omega \,\big|\, x_1<0.5 \big\}`$ and $`\Omega_2= \big\{(x_1,\cdots,x_5)\in\Omega \,\big|\, x_1>0.5 \big\}`$. Here, the initial guess of the Dirichlet data at interface is chosen as $` h^{[0]}(\mathbf{x})=u(\mathbf{x})-5000\left(x_1\prod\limits_{i=2}^5 x_i(x_i-1)\right)`$, and the neural network employed here has 8 hidden layers of 50 neurons each. The computational results using DN-PINNs, DNLA (PINNs), and DNLA (deep Ritz) approaches are shown as follows, which implies that our proposed learning algorithms can achieve better performance to the existing DN-PINNs approach.

![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/cefd2a7d-d95e-4de0-acb4-a17ee4c4fcca)

# Instructions for Generating Table 8 in Our Revised Manuscript
## Table 8 - DN-PINNs Result
To acquire the data represented in the first row of Table 8, execute the script `task1-DN-PINNs-5D-2prob.py`.

## Table 8 - DNLM(PINN) Result
For the results displayed in the Table 8, execute the script `task2-DN-DNLA_PINNs-5D-2prob.py` to generate the corresponding data.

## Table 8 - DNLM(Ritz) Result
For the data displayed in the Table 8, execute the script `task3-DN-DNLA_deepRitz-5D-2prob.py` to generate the corresponding results.

Owing to the challenges in visualizing high-dimensional problems, our presentation of results is exclusively through tabular formats in this situation.

