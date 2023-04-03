# MKSR

Multivariable Killer of Symbolic Regression 
(by applying control variable)

### Install

```sh
conda create --name sr python=3.10 -y
conda activate sr
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

### Steps

1. Get a set of $[x1, x2, x3, ... xn, y]$
2. Use arbitrary NN to learn y = f(x1, x2, x3, ... xn) \\
   (like knowledge distillation) \\      
   or skip this steps if already have enough data.
3. Generate the data using trained NN
4. Learn $f_{x2, x3, ... xn}(x1), f_{x1, x3, ... xn}(x2), \dots$ using SVSR.
5. Combine the result.

### TODO

- [x] finish first demo (2023.03.12)
- [x] try Monte Carlo method (canceled)
- [x] implement gradient descent (2023.03.14)
- [x] implement Dominant Search (2023.03.15)
- [ ] reading dso
- [ ] solve problems of y = x_1 * x_2 + x_1 + 2 * x_2 
- [ ] integrate step 3, 4, 5
- [ ] reading gnn distill

### 
- https://github.com/brendenpetersen/deep-symbolic-optimization