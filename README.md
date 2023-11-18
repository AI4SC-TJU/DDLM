# Domain Decomposition Learning Methods for Solving Elliptic Problems
With recent advancements in computer hardware and software platforms, there has been a surge of interest in solving boundary value problems with deep learning-based methods, and the integration with domain decomposition strategies has attracted considerable attention owing to its enhanced representation and parallelization capacities. Based on a direct transmission of Dirichlet and Neumann traces along subdomain interfaces, neural networks have already been employed as subproblem solvers in certain overlapping and non-overlapping methods. However, the boundary penalty treatment often leads to a tendency for the network solution and its derivatives to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion with increased accuracy. In this study, a novel learning approach, i.e., the compensated deep Ritz method using neural network extension operators, is proposed to construct effective learning algorithms for realizing non-overlapping domain decomposition methods (DDMs) even in the presence of inaccurate interface conditions. Numerical experiments on a variety of elliptic problems, including regular and irregular interfaces, low and high dimensions, two and four subdomains, and smooth and high-contrast coefficients are carried out to validate the effectiveness of our proposed algorithms.

- Qi Sun, XueJun Xu, HaoTian Yi.


## Citation

    @article{sun2022domain,
      title={Domain Decomposition Learning Methods for Solving Elliptic Problems},
      author={Sun, Qi and Xu, Xuejun and Yi, Haotian},
      journal={arXiv preprint arXiv:2207.10358},
      year={2022}
            }
