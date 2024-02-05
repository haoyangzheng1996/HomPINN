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
$$
  \left\{\begin{aligned}
    &\frac{\partial ^2 u(x)}{\partial x^2}=-\lambda\left(1+ u^4\right),\ \ x\in (0,1)\\
    &{\left.\frac{\partial u(x)}{\partial x}\right |_{x=0}=\left.u(x)\right |_{x=1}=0}.
  \end{aligned}\right.
$$
Please run:
```
python3 main_HomPINN.py
```



More will be updated soon.

## Contact
Haoyang Zheng, School of Mechanical Engineering, Purdue University

Email: zheng+528 at purdue dot edu

More Aboue Me: [link](https://haoyangzheng.github.io/)
