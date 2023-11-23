# Domain Decomposition Learning Methods for Elliptic Problems
Based on a direct transmission of Dirichlet and Neumann traces along subdomain interfaces, neural networks have already been employed as subproblem solvers in certain overlapping and non-overlapping methods. However, the boundary penalty treatment often leads to a tendency for the network solution and its derivatives to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion with increased accuracy. In this study, a novel learning approach, i.e., the compensated deep Ritz method using neural network extension operators, is proposed to construct effective learning algorithms for realizing non-overlapping domain decomposition methods even in the presence of inaccurate interface conditions. 



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
