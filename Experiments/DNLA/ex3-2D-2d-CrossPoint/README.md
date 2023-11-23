Here, we consider the Poisson problem that is divided into four subproblems in two-dimension, i.e.,
```math
\begin{equation}
\begin{array}{cl}
-\Delta u(x,y)  = f(x,y)\ & \text{in}\ \Omega=(0,1)^2, \\
u(x,y) = 0\ \ & \text{on}\ \partial \Omega,
\end{array}
\end{equation}
```
where $`u(x,y) = \sin(2\pi x)\sin(2\pi y)`$ and $f(x,y)=$ $8 \pi^2 \sin(2\pi x)\sin(2\pi y)$. Here, the domain is decomposed using the red-black partition, while the multidomains are categorized into two sets. Then, the deep learning-based algorithms are deployed, with the initial guess at interface chosen as $h^{[0]}(x,y)=u(x,y)-1000x(x-1)y(y-1)$ in what follows.

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/20aee609-98d8-4505-ac33-1b27b5da7150)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$ |

For the problem with non-trivial flux functions along the interface, it is not guaranteed that iterative solutions using DN-PINNs will converge to the true solution due to issue of inaccurate Dirichlet-to-Neumann map.

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/8bcfecdb-44b2-4d86-b038-59c533eddc07)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DN-PINNs on the test dataset.* |


However, even though the inaccurate flux predicition on subdomain interfaces remains unresolved when using our methods (see \autoref{Experiments-DNLA-ex3-Overfit-Dirichlet-Subproblem}), the compensated deep Ritz method has enabled the Neumann subproblem to be solved with acceptable accuracy.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/f7f1f48e-9872-4c0d-9f58-7ff5abe31efc)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/149239dd-d892-486a-9e7a-8f11c3318e74)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ and the pointwise absolute error using DNLA (Ritz) on the test dataset.* |
