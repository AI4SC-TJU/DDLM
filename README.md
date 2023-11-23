# Domain Decomposition Learning Methods for Elliptic Problems
Based on a direct transmission of Dirichlet and Neumann traces along subdomain interfaces, neural networks have already been employed as subproblem solvers in certain overlapping and non-overlapping methods. However, the boundary penalty treatment often leads to a tendency for the network solution and its derivatives to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion with increased accuracy. In this study, a novel learning approach, i.e., the compensated deep Ritz method using neural network extension operators, is proposed to construct effective learning algorithms for realizing non-overlapping domain decomposition methods even in the presence of inaccurate interface conditions. 



We consider the Dirichlet-Neumann algorithm for solving Poisson's equations, where the Dirichlet subproblem is solved through the aforementioned PINNs approach.









On the one hand, the Neumann subproblem within the DN-PINNs strategy is also solved using PINNs, i.e.,
```math
\begin{align*}
    \hat{u}_2^{[k]} = \mathop{\text{arg\,min}}_{u_2} \int_{\Omega_2} | \Delta u_2 + f |^2\,dx + \beta\left(\int_{\textcolor{red}{\Gamma}} \Big|  \frac{\partial u_2}{\partial \mathbf{n}_2} - \textcolor{red}{\frac{\partial \hat{u}^{[k]}_1}{\partial \mathbf{n}_2}} \Big|^2\,ds + \int_{\partial\Omega\cap\partial\Omega_2} |u_2|^2\,ds \right),
\end{align*}
```
where the boundary data (marked in red color) relies on the Neumann trace of Dirichlet subproblem \textcolor{red}{$`\nabla \hat{u}^{[k]}_1 |_\Gamma`$}. On the other hand, the loss function of our proposed method is defined as
```math
\begin{align*}
    	\hat{u}_2^{[k]} = \mathop{\text{arg\,min}}_{\hat{u}_2} \int_{\Omega_2} \Big( \frac12 | \nabla \hat{u}_2 |^2  - f \hat{u}_2 \Big) dx + \int_{\textcolor{blue}{\Omega_1}} \Big( \textcolor{blue}{\nabla \hat{u}_1^{[k]}}\cdot \nabla \hat{u}_2  - f \hat{u}_2 \Big) dx + \beta \int_{\partial\Omega} |\hat{u}_2|^2\,ds,
\end{align*}
```
where the exchanged data (marked in blue color) is represented using the interior solution \textcolor{blue}{$`\nabla \hat{u}^{[k]}_1 |_{\Omega_1}`$}.


|![fine-tuning](https://github.com/AI4SC-TJU/DDLM/assets/93070782/1db6cefd-b7bf-460e-87c6-5ff0bd523bca)|
|:--------------------------------------------------------------:|
| *Network solutions and error profiles for Poisson problem using DN-PINNs and DNLA (PINNs), with fine-tuned hyperparameters.* |




## Citation

    @article{sun2022domain,
      title={Domain Decomposition Learning Methods for Solving Elliptic Problems},
      author={Sun, Qi and Xu, Xuejun and Yi, Haotian},
      journal={arXiv preprint arXiv:2207.10358},
      year={2022}
            }
