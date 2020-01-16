from numpy import random as r
from numpy import array as array

class DataGenerator():
    def __init__(self, types, params, size=1):
        self.funcs = types
        self.args = params
        self.size = size
    
    def count(self):
        li = [] 
        for i in range(len(self.funcs)):
            li.append(eval( "self." + self.funcs[i] + "(*" + str(self.args[i]) + ")" )[0]) ###might rethink 0 index
        return li

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

    def hypergeometric(self, ngood, nbad, nall)
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

    

    ###WIP###