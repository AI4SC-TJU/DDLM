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

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/3bffc9a1-c5c9-4a28-b1fc-651123268f0e)|
|:--------------------------------------------------------------:|
| *From left to right: decomposition into two subdomains, true solution $`u(x,y)`$, and its partial derivatives $`\partial_x u(x,y)`$, $`\partial_y u(x,y)`$ |



|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/2d6b9d5c-ee25-4ed9-b63f-7b63bb13c165)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DN-PINNs on the test dataset.* |


But our proposed method, DNLA (PINNs) and DNLA (Ritz), can still work since the variational principle is used.


|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/a4ff311e-4fa6-45b0-9dbb-f6395cf9df12)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (PINNs) on the test dataset.* |

|![image](https://github.com/AI4SC-TJU/DDLM/assets/93070782/f9a78c6b-516b-46b7-bb34-4ac27d506cce)|
|:--------------------------------------------------------------:|
| *Iterative solutions $`\hat{u}^{[k]}(x,y)`$ using DNLA (Ritz) on the test dataset.* |
