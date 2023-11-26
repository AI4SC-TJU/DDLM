# Domain Decomposition Learning Methods for Elliptic Problems
### Qi Sun<sup>1</sup>, Xuejun Xu<sup>1,2</sup> and Haotian Yi<sup>1</sup>

Paper: [https://arxiv.org/abs/2207.10358](https://arxiv.org/abs/2207.10358)

Abstract: _Based on a direct transmission of Dirichlet and Neumann traces along subdomain interfaces, neural networks have already been employed as subproblem solvers in certain overlapping and non-overlapping methods. However, the boundary penalty treatment often leads to a tendency for the network solution and its derivatives to furnish more precision inside the domain, rather than at the boundary, thereby motivating the exploration of a variational approach for enforcing flux transmisssion with increased accuracy. In this study, a novel learning approach, i.e., the compensated deep Ritz method using neural network extension operators, is proposed to construct effective learning algorithms for realizing non-overlapping domain decomposition methods even in the presence of inaccurate interface conditions._ 

<sub><sub><sup>1</sup>School of Mathematical Sciences, Tongji University, Shanghai 200092, China, TX</sub></sub><br>
<sub><sub><sup>2</sup>Institute of Computational Mathematics, AMSS, Chinese Academy of Sciences, Beijing 100190, China, MD</sub></sub><br>


## Preview of results
|![fine-tuning](https://github.com/AI4SC-TJU/DDLM/assets/93070782/1db6cefd-b7bf-460e-87c6-5ff0bd523bca)|
|:--------------------------------------------------------------:|
| *Network solutions and error profiles for Poisson problem using DN-PINNs and DNLA (PINNs), with fine-tuned hyperparameters.* |


## Requirements

Code was implemented in `python 3.7` with the following package versions:

```
pytorch version = 1.8.1 + cu111
tensorflow version = 2.8.0
```

and `Matlab 2023b` was used for visualization.


## Citation

    @article{sun2022domain,
      title={Domain Decomposition Learning Methods for Solving Elliptic Problems},
      author={Sun, Qi and Xu, Xuejun and Yi, Haotian},
      journal={arXiv preprint arXiv:2207.10358},
      year={2022}
            }
