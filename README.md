# MKSR

Multivariable Killer of Symbolic Regression 
(by applying control variable)

### Install

```sh
conda create --name sr python=3.10 -y
conda activate sr
conda install numpy pandas sympy scipy
```

### Steps

1. Get a set of $[x1, x2, x3, ... xn, y]$
2. Use arbitrary NN to learn y = f(x1, x2, x3, ... xn) 
   (like knowledge distillation)
   or skip this steps if already have enough data.
3. Generate the data using trained NN
4. Learn $f_{x2, x3, ... xn}(x1), f_{x1, x3, ... xn}(x2), \dots$ using SVSR.
5. Combine the result.

### TODO

- [x] finish first demo (2023.03.12)
- [x] try Monte Carlo method (canceled)
- [x] implement gradient descent (2023.03.14)
- [x] implement Dominant Search (2023.03.15)
- [x] reading dso (2023.03.19)
- [x] learning Monte Carlo method for SR (2023.04.02)
- [x] finish version 1 and testing (2023.04.04)
- [x] solve problems of y = x_1 * x_2 + x_1 + 2 * x_2 (2023.04.06)
- [x] integrate step 3, 4, 5 (2023.04.06)
- [ ] reading dso other dissertation
- [ ] optimizing speed
- [ ] implement step 1, 2 and combine

### Other's work
- https://github.com/brendenpetersen/deep-symbolic-optimization
- https://github.com/isds-neu/SymbolicPhysicsLearner

### Current Testing Result
command
```sh
conda activate sr
python3 src/main.py
```
2023/04/06

ground truth equation: $\sin(x_0) (2.5 x_1^2 + \cos(x_1)) + x_1 + 3$

discovered equation: x1 + (2.50004998*x1**2 + cos(x1))*sin(x0) + 3.0

---

ground truth equation: $x_0 x_1 + x_0 + 2 x_1 + e^{x_1}$

discovered equation: x0*(x1 + 1.0) + 2.0*x1 + exp(x1)

---

ground truth equation: $x_0 x_1 + x_0 + 2 \frac{x1}{x2} + x_2 e^{x_1}$

discovered equation: (2.0*x1 + x2*(x0*(x1 + 1.0) + x2*exp(x1)))/x2