import numpy as np
import GPy

class SplitGP(Object):
    '''
    A number of seperate models inferred by splitting up training data.
    '''
    def __init__(self, X, Y, kern1, kern2, name):
        pass

    def optimize(self, reassignment='naive'):
        pass

    def plot(self):
        pass
