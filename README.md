# HomPINNs: homotopy physics-informed neural networks for solving the inverse problems of nonlinear differential equations with multiple solutions

Experiment code for "[HomPINNs](https://www.sciencedirect.com/science/article/pii/S0021999123008471)".

```
@article{hompinns,
  title={HomPINNs: homotopy physics-informed neural networks for solving the inverse problems of nonlinear differential equations with multiple solutions},
  author={Zheng, Haoyang and Huang, Yao and Huang, Ziyang and Hao, Wenrui and Lin, Guang},
  journal={Journal of Computational Physics},
  pages={112751},
  year={2024},
  publisher={Elsevier}
}
```

## Prerequisites
Please refer to "requirement.txt" 

## The first example

```math
  \left\{\begin{aligned}
    &\frac{\partial ^2 u(x)}{\partial x^2}=-\lambda\left(1+ u^4\right),\ \ x\in (0,1)\\
    &{\left.\frac{\partial u(x)}{\partial x}\right |_{x=0}=\left.u(x)\right |_{x=1}=0}.
  \end{aligned}\right.
```
with $\lambda=1.20$

Please run:
```
python3 main_HomPINN.py
```

Results:
[pred1.pdf](https://github.com/haoyangzheng1996/HomPINN/files/14172901/pred1.pdf)

## The second example

```math
  \left\{
  \begin{aligned}
     &\frac{\partial ^2 u(x)}{\partial x^2} =u^4-\lambda u^2,\ \ x\in (0\ ,1),\\
     &\left.\frac{\partial u(x)}{\partial x}\right |_{x=0}=\left.u(x)\right |_{x=1}=0.
  \end{aligned}\right.
```
with $\lambda=18.00$

Please run:
```
python3 main_HomPINN.py --data_dir ./data/obs_ex2.mat --n_epoch 20000 --max_epoch 40000 --num_sol 3 --num_obs 100 --lr_low 1e-4 --lr_gap 0.98
```

Results:
[pred2.pdf](https://github.com/haoyangzheng1996/HomPINN/files/14172911/pred2.pdf)


## Contact
Haoyang Zheng, School of Mechanical Engineering, Purdue University

Email: zheng+528 at purdue dot edu

More Aboue Me: [link](https://haoyangzheng.github.io/)
