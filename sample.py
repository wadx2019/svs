import numpy as np


class SampleFunc:
    pass

class Linear(SampleFunc):

    def __init__(self, c, b):
        super().__init__()
        self.c = c
        self.b = b

    def __call__(self, x):
        tmp = self.c*x+self.b
        return np.clip(tmp, None, 1)

class Quaratic(SampleFunc):

    def __init__(self, a, c, b):
        super().__init__()
        self.a = a
        self.c = c
        self.b = b

    def __call__(self, x):
        tmp = self.a*x*x+self.c*x+self.b
        return np.clip(tmp, None, 1)

iden = Linear(1.0, 0)
quad = Quaratic(1.0,0,0)
