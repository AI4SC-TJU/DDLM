# DOMAIN DECOMPOSITION LEARNING METHODS FOR SOLVING ELLIPTIC PROBLEMS
With recent advancements in computer hardware and software platforms, there has been a surge of interest in solving partial differential equations with deep learning-based methods, and the integration with domain decomposition strategies has attracted considerable attention owing to its enhanced representation and parallelization capacities of the network solution. While there are already several works that substitute the subproblem solver with neural networks for overlapping Schwarz methods, the non-overlapping counterpart has not been extensively explored because of the inaccurate flux estimation at interface that would propagate errors to neighbouring subdomains and eventually hinder the convergence of outer iterations. In this study, a novel learning approach for solving elliptic boundary value problems, i.e., the compensated deep Ritz method using neural network extension operators, is proposed to enable the reliable flux transmission across subdomain interfaces, thereby allowing us to construct effective learning algorithms for realizing non-overlapping domain decomposition methods (DDMs) in the presence of erroneous interface conditions. Numerical experiments on a variety of elliptic problems, including regular and irregular interfaces, low and high dimensions, two and four subdomains, and smooth and high-contrast coefficients are carried out to validate the effectiveness of our proposed algorithms.

# citation

@article{sun2022domain,
  title={Domain Decomposition Learning Methods for Solving Elliptic Problems},
  author={Sun, Qi and Xu, Xuejun and Yi, Haotian},
  journal={arXiv preprint arXiv:2207.10358},
  year={2022}
}

