
To demonstrate the effectiveness of our compensated deep Ritz method for realizing the non-overlapping Robin-Robin algorithm, we consider the following Poisson equation in two-dimension
```math
\begin{equation}
\begin{array}{cl}
-\Delta u(x,y)  = 4 \pi^2 \sin(2 \pi x)  (2 \cos(2 \pi y) - 1)  \ & \text{in}\ \Omega=(0,1)^2,\\
u(x,y) = 0\ \ & \text{on}\ \partial \Omega,
\end{array}
\end{equation}
```
where the exact solution $u(x,y) = \sin(2\pi x)(\cos(2\pi y)-1)$, and the interface $\Gamma=\partial\Omega_1\cap\partial\Omega_2$ is a straight line segment from $(0.5,0)$ to $(0.5,1)$. 

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/ab9eb048-55bd-47a2-a912-f633b9fb912f)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using RRLA on the test dataset.*  |




# Code introduction
## RR-PINNs Result
To acquire the data represented in Figure 14 and Figure 15, execute the script `RR-PINNs-2Prob-2D.py`.

## RRLM(PINN) Result
For the results displayed in Figure 16, execute the script `RRLM-2prob-2D-Compensent.py` to generate the corresponding data.

## Figure Generation:
Utilize MATLAB and execute the script `plot_solution_1by1.m` to create the graphical representation associated with the data obtained from the previous steps.

Please ensure that the parameter "alpha_left" and "alpha_right" represent $\kappa_1$ and $\kappa_2$ respectively in our manuscript.
