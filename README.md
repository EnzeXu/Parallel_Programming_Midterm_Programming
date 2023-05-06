# MKSR

Multivariable Killer of Symbolic Regression 
(by applying control variable)

### Install

(to be update)

### Steps

(to be update)

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
- [x] implement step 1, 2 and combine (2023.04.13)

### Other's work
- https://github.com/brendenpetersen/deep-symbolic-optimization
- https://github.com/isds-neu/SymbolicPhysicsLearner

### Tips
* Config file for test `test/config/alltests.py`
* If update the data range or data num, run `cd test; python3 gen.py` first. data will locate at `test/data/*`
* Running our method: `cd test; python3 run_mksr_spl.py --task=Korns-2`
* NN related code: `src/srnn/mlnn/trainer.py`, `src/srnn/mlnn/model/*`