import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy import stats

class KernelPair(object):
    '''
    '''
    def __init__(self, kern1, kern2):
        self.kern1 = kern1
        self.kern2 = kern2

class SplitGP(object):
    '''
    A number of seperate models inferred by splitting up training data.
    '''
    def __init__(self, X, Y, kern1, kern2, name='SplitGP'):
        self.X = X
        self.Y = Y
        self.kern = KernelPair(kern1, kern2)
        self.membership = np.random.randint(0, 2, size=Y.shape)

        i_1 = np.nonzero(self.membership)[0]
        i_2 = np.nonzero(1 - self.membership)[0]

        self.m1 = GPy.models.GPRegression(self.X[i_1], self.Y[i_1], kernel=self.kern.kern1)
        self.m2 = GPy.models.GPRegression(self.X[i_2], self.Y[i_2], kernel=self.kern.kern2)

    def log_likelihood(self):
        return self.m1.log_likelihood() + self.m2.log_likelihood()

    def optimize(self, reassignment='naive', n_iter=4):
        '''
        '''
        self.m1.optimize()
        self.m2.optimize()

        for i in range(n_iter):
            if reassignment == 'naive':
                new_assignment = np.hstack((np.abs(self.m1.predict(self.X)[0] - self.Y),
                                            np.abs(self.m2.predict(self.X)[0] - self.Y))).argmax(1)

            if reassignment == 'gmm':
                # WARNING: This mode occasionally segfaults Python.
                new_assignment = np.hstack((stats.norm.logpdf(self.Y, *self.m1.predict(self.X)),
                                            stats.norm.logpdf(self.Y, *self.m2.predict(self.X)))).argmax(1)

            i_1 = np.nonzero(new_assignment)[0]
            i_2 = np.nonzero(1 - new_assignment)[0]

            self.m1.set_XY(self.X[i_1], self.Y[i_1])
            self.m1.optimize()
            self.m2.set_XY(self.X[i_2], self.Y[i_2])
            self.m2.optimize()

    def plot(self):
        ax = plt.subplot(111)
        self.m1.plot(ax=ax, which_data_rows=[], linecol='r');
        ax.scatter(self.m1.X, self.m1.Y, c='r')
        self.m2.plot(ax=ax, which_data_rows=[], linecol='k');
        ax.scatter(self.m2.X, self.m2.Y, c='k');

        plt.ylim(self.Y.min(), self.Y.max());
        plt.xlim(self.X.min(), self.X.max());
