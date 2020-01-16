from numpy import random as r
import numpy as np
from sklearn.datasets import make_classification

class DataGenerator():
    def __init__(self, types, params, size=1000):
        self.funcs = types
        self.args = params
        self.size = size
    
    def count(self):
        #Цикл по 10000 строчек
        #для каждого распределения создается массив
        #превращаю массив в np.array
        #конкатинирую
        list_of_ML = ["classification", "regression"]
        li = []
        for i in range(len(self.funcs)):
            if self.funcs[i] not in list_of_ML:
                li.append(eval( "self." + self.funcs[i] + "(*" + str(self.args[i]) + ")" ))
            else:
                x, y = eval( "self." + self.funcs[i] + "(*" + str(self.args[i]) + ")" )
                for values in x:
                    li.append(values)
                li.append(y)
        return np.array(li).T.tolist()

    ### DISTRIBUTIONS ###
    def beta(self, a, b):
        '''
        Parameters:\n
        a: float. b: float.
        '''
        return r.beta(a, b, self.size)

    def binomial(self, n, p):
        '''
        Parameters:\n
        n: integer, >=0 . p: float, >=0.
        '''
        return r.binomial(n, p, self.size)

    def exponential(self, scale):
        '''
        Parameters:\n
        scale: float >=0.
        '''
        return r.exponential(scale, self.size)

    def gamma(self, k, theta):
        '''
        Parameters:\n
        k: float, >=0 . theta: float, >=0.
        '''
        return r.gamma(k, theta, self.size)

    def geometric(self, p):
        '''
        Parameters:\n
        p: float, >=0.
        '''
        return r.geometric(p, self.size)

    def hypergeometric(self, ngood, nbad, nall):
        '''Parameters:\n
        ngood: integer, >=0.\n
        nbad: integer, >=0.\n
        nall: integer, >=1 and <=ngood+nbad.
        '''
        return r.hypergeometric(ngood, nbad, nall, self.size)

    def laplace(self, mean, scale):
        '''Parameters:\n
        mean: float. scale: float, >=0.
        '''
        return r.laplace(mean, scale, self.size)

    def logistic(self, mean, scale):
        '''
        Parameters:\n
        mean: float. scale: float, >=0.
        '''
        return r.logistic(mean, scale, self.size)

    def lognormal(self, mean, std):
        '''
        Parameters:\n
        mean: float. std: float, >=0.
        '''
        return r.lognormal(mean, std, self.size)

    def logarithmic(self, p):
        '''
        Parameters:\n
        p: float, must be in range (0, 1).
        '''
        return r.logseries(p, self.size)

    def multinomial(self, n, pr_of_vals):
        '''
        Parameters:\n
        n: int, >=0.\n
        pr_of_vals: list of float probabilities, sum must be = 1.
        '''
        return r.multinomial(n, pr_of_vals, self.size)

    def negative_binomial(self, n, p):
        '''
        Parameters:\n
        n: int, >0. p: float in range [0, 1].
        '''
        return r.negative_binomial(n, p, self.size)

    def normal(self, mean, std):
        '''
        Parameters:\n
        mean: float. std: float >=0.
        '''
        return r.normal(mean, std, self.size)

    def poisson(self, lam):
        '''
        Parameters:\n
        lam: float, >0.
        '''
        return r.poisson(lam, self.size)

    def triangular(self, left, top, right):
        '''
        Parameters:\n
        left: float. \n
        top: float, must be >= left.\n
        right: float, must be >= top.
        '''
        return r.triangular(left, top, right, self.size)

    def uniform(self, left, right):
        '''
        Parameters:\n
        left: float.\n
        right: float, must be >left.
        '''
        return r.uniform(left, right, self.size)

    def weibull(self, a):
        '''
        Parameters:\n
        a: float, >=0.
        '''
        return r.weibull(a, self.size)

    ### ML-like DATA
    def classification(self, n_features, n_informative, n_redundant, n_classes, labels=0, weights=None, noise=0.01, complexity=1.0, intervals=None):
        #n_classes * n_clusters_per_class must be smaller or equal 2 ** n_informative

        #shift and scale from intervals
        shift = []
        scale = []
        for interval in intervals:
            shift.append((interval[1] + interval[0]) / (interval[1] - interval[0]))
            scale.append((interval[1] - interval[0]) / 2)
        
        X, y = make_classification(n_samples=self.size, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant,
         n_classes=n_classes, n_clusters_per_class=1, weights=weights, flip_y=noise, class_sep=complexity, shift=shift, scale=scale, shuffle=False)
        #naming labels if needed
        if labels != 0:
            for i in range(n_classes):
                y[y == i] = label[i]

        return X, y
    #shuffle=False