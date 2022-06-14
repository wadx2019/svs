import numpy as np


class SampleFunc:
    pass

class Linear(SampleFunc):

    def __init__(self, c, b):
        super().__init__()
        self.c = c
        self.b = b
        self.name = "linear"

    def __call__(self, x):
        tmp = self.c*x+self.b
        return np.clip(tmp, None, 1)

class Quaratic(SampleFunc):

    def __init__(self, a, c, b):
        super().__init__()
        self.a = a
        self.c = c
        self.b = b
        self.name = "quad"

    def __call__(self, x):
        tmp = self.a*x*x+self.c*x+self.b
        return np.clip(tmp, None, 1)

class Square(SampleFunc):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.name = "square"

    def __call__(self, x):
        tmp = self.a*np.sqrt(x)+self.b
        return np.clip(tmp, None, 1)

class Log(SampleFunc):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.name = "log"

    def __call__(self, x):
        tmp = self.a*np.log(x)+self.b
        return np.clip(tmp, None, 1)

lin = Linear(0.0016, 0)
quad = Quaratic(0.000004, 0, 0)
square = Square(0.1, 0)
log = Log(0.25, 0)
