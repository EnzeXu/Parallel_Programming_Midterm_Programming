import numpy as np
import torch

class TerminalOp:
    pass

class UnaryOp:
    def __str__(self):
        return f"({self.__class__.__name__} {str(self.son)})"
    def Eval(self, pivot):
        self.son.Eval(pivot)
        self.value = self.func(self.son.value)
    def EvalForConstant(self, C, X):
        return self.func(self.son.EvalForConstant(C, X))

class BinaryOp:
    def __str__(self):
        return f"({self.__class__.__name__} {str(self.lson)}, {str(self.rson)})"
    def Eval(self, pivot):
        self.lson.Eval(pivot) 
        self.rson.Eval(pivot)
        self.value = self.func(self.lson.value, self.rson.value)
    def EvalForConstant(self, C, X):
        return self.func(self.lson.EvalForConstant(C, X), self.rson.EvalForConstant(C, X))


class Constant(TerminalOp):
    def __str__(self):
        return f"({self.__class__.__name__})"
    def __init__(self, value):
        self.value = value
    def Eval(self, pivot):
        self.value = 1.0
    def EvalForConstant(self, C, X):
        cur = torch.zeros(len(X))
        cur[:] = C[self.id]
        return cur

class Variable(TerminalOp):
    def __str__(self):
        return f"({self.__class__.__name__} {self.name})"
    def __init__(self, name):
        self.name = name
    def Eval(self, pivot):
        self.value = pivot
    def EvalForConstant(self, C, X):
        return X

class Add(BinaryOp):
    def __init__(self, lson, rson):
        self.lson = lson
        self.rson = rson
        self.func = lambda x, y: x + y

class Sub(BinaryOp):
    def __init__(self, lson, rson):
        self.lson = lson
        self.rson = rson
        self.func = lambda x, y: x - y

class Mul(BinaryOp):
    def __init__(self, lson, rson):
        self.lson = lson
        self.rson = rson
        self.func = lambda x, y: x * y

class Div(BinaryOp):
    def __init__(self, lson, rson):
        self.lson = lson
        self.rson = rson
        self.func = lambda x, y: x / y
        
class Sqrt(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = lambda x: torch.sqrt(torch.abs(x))

class Sqr(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = lambda x: x**2

class Cube(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = lambda x: x**3

class Sin(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = torch.sin

class Cos(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = torch.cos

class Log(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = lambda x: torch.log(torch.abs(x))

class Exp(UnaryOp):
    def __init__(self, son):
        self.son = son
        self.func = lambda x: torch.exp(x)