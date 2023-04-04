import numpy as np
import torch
import math
import bisect
import time
from copy import deepcopy
# from gplearn import genetic
# import re
from .svsr_item import *

class SVtree:
    def __init__(self, 
                 prog,
                 tree,
                 train_x,
                 train_y,
                 learning_rate = 0.05):
        self.prog = prog
        self.tree = tree
        self.train_x = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        
        self.const_num = 0
        self.CountConstNum(self.tree)
        self.regression_times = 0
        self.loss = 10**9  # inf
        self.is_bad_expression = False

        C_mean = torch.mean(train_x)
        C_std = torch.std(train_x) * 0.5
        sample_id = np.random.choice(len(self.train_x), 2)
        for _ in range(5): # try different C 5 times. if nan, return.
            self.C = torch.normal(mean=torch.tensor([C_mean] * self.const_num), std=C_std)
            value = torch.sum(self.tree.EvalForConstant(self.C, train_x[sample_id]))
            if not math.isnan(value.item()):
                break
        value = torch.sum(self.tree.EvalForConstant(self.C, train_x[sample_id]))
        if math.isnan(value.item()):
            self.is_bad_expression = True
        

    def CountConstNum(self, tree):
        if isinstance(tree, TerminalOp):
            if isinstance(tree, Constant):
                tree.id = self.const_num
                self.const_num += 1
        elif isinstance(tree, UnaryOp):
            self.CountConstNum(tree.son)
        elif isinstance(tree, BinaryOp):
            self.CountConstNum(tree.lson)
            self.CountConstNum(tree.rson)

    def RegressionStep(self, steps):
        # print("Regress For:", self.tree)
        self.regression_times += steps
        if self.const_num:
            sid = np.random.choice(len(self.train_x), 5)
            train_x = self.train_x[sid]
            train_y = self.train_y[sid]
            for _ in range(steps):
                self.C.requires_grad_(True)
                loss = torch.sum(self.tree.EvalForConstant(self.C, train_x) - train_y)**2 / len(train_x)
                loss.backward()
                self.C = (self.C - self.learning_rate * self.C.grad).detach()
        # time.sleep(1)
        self.loss = torch.sum((self.tree.EvalForConstant(self.C, self.train_x) - self.train_y)**2) / len(self.train_x)
        # print("finish. loss =", self.loss)

class SVSR:
    def __init__(self, 
                 var_name, 
                 train_x, 
                 train_y, 
                 binary_op = ['add', 'sub', 'mul', 'div'],
                 unary_op = ['sqrt', 'sqr', 'cube', 'sin', 'cos', 'log', 'exp'],
                 terminal_op = ['x', 'const'],
                 require_error = 0.1,
                 regression_round = 5,
                 keep_module_number = 100,
                 ):
        """
        regression_round: after model is designed, regression for how many rounds.
        """
        self.var_name = var_name
        self.train_x = train_x
        self.train_y = train_y
        self.binary_op = binary_op
        self.unary_op = unary_op
        self.terminal_op = terminal_op
        self.require_error = require_error
        self.regression_round = regression_round
        self.keep_module_number = keep_module_number
        self.debug = True

        self.update_counter = 0
        self.best_trees = []

        for len in range(1, 6):
            self.SearchSymbolic([], 1, len)
            if self.best_trees[0].loss < require_error:
                break
        # for x in self.best_trees:
        #     print(x.tree, x.loss.item(), x.regression_times)
        sp = self.best_trees[0]
        print(sp.tree, sp.loss.item())
        for incremental_round in range(3):
            if self.best_trees[0].loss < require_error:
                break
            best_trees = deepcopy(self.best_trees)
            for tree in best_trees:
                for len in range(1, 4):
                    for op in self.binary_op:
                        self.SearchSymbolic([op] + tree.prog, 1, len)
            sp = self.best_trees[0]
            print(sp.tree, sp.loss.item())


        # print(sp.tree.EvalForConstant(sp.C, sp.train_x))

    def BuildTree(self, prog_stack):
        """
        be careful with the order!!
        prog[-1] is the first one.
        """
        if not prog_stack:
            raise Exception("Parse Failed")
        token = prog_stack[-1]
        prog_stack.pop()
        if token in self.binary_op:
            lson = self.BuildTree(prog_stack)
            rson = self.BuildTree(prog_stack)
            if token == 'add':
                return Add(lson, rson)
            elif token == 'sub':
                return Sub(lson, rson)
            elif token == 'mul':
                return Mul(lson, rson)
            elif token == 'div':
                return Div(lson, rson)
            else:
                raise Exception("No such Binary Op")
        elif token in self.unary_op:
            son = self.BuildTree(prog_stack)
            if token == 'sqrt':
                return Sqrt(son)
            elif token == 'sqr':
                return Sqr(son)
            elif token == 'cube':
                return Cube(son)
            elif token == 'sin':
                return Sin(son)
            elif token == 'cos':
                return Cos(son)
            elif token == 'log':
                return Log(son)
            elif token == 'exp':
                return Exp(son)
            else:
                raise Exception("No such Unary Op")
        elif token == 'x':
            return Variable(self.var_name)
        elif token == 'const':
            return Constant(token)
        else:
            raise Exception(f"could not reconize '[{token}]' as a token")

    def SearchSymbolic(self, prog, stack_size, remain_len):
        if remain_len == 0:
            tree = SVtree(prog.copy(), self.BuildTree(prog[::-1].copy()), self.train_x, self.train_y)
            if tree.is_bad_expression:
                return
            tree.RegressionStep(self.regression_round)
            bisect.insort(self.best_trees, tree, key = lambda x: x.loss)
            if len(self.best_trees) > self.keep_module_number:
                self.best_trees.pop()
            self.update_counter += 1
            if self.update_counter == self.keep_module_number * 4:
                self.update_counter = 0
                for tree in self.best_trees:
                    tree.RegressionStep(self.regression_round)
                self.best_trees.sort(key = lambda x: x.loss)
            return
        if remain_len >= stack_size + 2:
            for op in self.binary_op:
                prog.append(op)
                self.SearchSymbolic(prog, stack_size + 1, remain_len - 1)
                prog.pop()
        if remain_len >= stack_size + 1:
            for op in self.unary_op:
                prog.append(op)
                self.SearchSymbolic(prog, stack_size, remain_len - 1)
                prog.pop()
        if not (remain_len > 1 and stack_size == 1):
            for op in self.terminal_op:
                prog.append(op)
                self.SearchSymbolic(prog, stack_size - 1, remain_len - 1)
                prog.pop()

def Merge(ast1, ast2, tolerant = 0.1):
    print(f"Match ~ {ast1.hash}, {ast2.hash}")
    if np.abs(ast1.hash - ast2.hash) > tolerant:
        raise Exception(f"ast match failed")
    if isinstance(ast1, RealNumber):
        return ast2
    elif isinstance(ast2, RealNumber):
        return ast1
    elif (isinstance(ast1, Add) and isinstance(ast2, Add)) or (isinstance(ast1, Mul) and isinstance(ast2, Mul)):
        ast1.sons.sort(key = lambda x: 1e100 if isinstance(x, RealNumber) else x.hash)
        ast2.sons.sort(key = lambda x: 1e100 if isinstance(x, RealNumber) else x.hash)
        p1 = 0
        p2 = 0
        s1 = []
        s2 = []
        sons = []
        while p1 < len(ast1.sons) and not isinstance(ast1.sons[p1], RealNumber) and p2 < len(ast2.sons) and not isinstance(ast2.sons[p2], RealNumber):
            if ast1.sons[p1].hash == ast2.sons[p2].hash:
                sons.append(Merge(ast1.sons[p1], ast2.sons[p2]))
                p1 += 1
                p2 += 1
            elif ast1.sons[p1].hash < ast2.sons[p2].hash:
                s1.append(ast1.sons[p1])
                p1 += 1
            else:
                s2.append(ast2.sons[p2])
                p2 += 1

        if (p1 == len(ast1.sons) and s2) or (p2 == len(ast2.sons) and s1):
            raise Exception(f"ast match failed for add/mul (no coe)")
            ast1.sons = sons
            return ast1
        if p1 == len(ast1.sons) or p2 == len(ast2.sons):
            raise Exception(f"ast match failed for add/mul")
        if len(ast1.sons) < len(ast2.sons):
            ast1.sons.pop()
            ast1.sons.extend(ast2[p:])
        return ast1
    elif isinstance(ast1, Neg) and isinstance(ast2, Neg):
        ast1.son = Merge(ast1.son, ast2.son)
        return ast1
    elif isinstance(ast1, Inv) and isinstance(ast2, Inv):
        ast1.son = Merge(ast1.son, ast2.son)
        return ast1
    else:
        raise Exception(f"ast could not merge")