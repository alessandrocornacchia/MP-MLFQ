from functools import partial
import numpy as np
from pyswarm import pso
from scipy.optimize import basinhopping
import copy

from includes.utils import precision_round

analytical_cdfs = {'ws_bp': {'name': 'BP', 'L': 3, 'H': 29200, 'alpha': 0.125, 'desc': 'WS_BP', 'dx' : 1, 'class': 'BoundedPareto'},
                   'dm_bp': {'name': 'BP', 'L': 0.1, 'H': 100000, 'alpha': 0.26, 'desc': 'DM_BP', 'dx': 1, 'class': 'BoundedPareto'},
                   'exp': {'name': 'EXP', 'mean': 10, 'H': 10000, 'dx': .1}}

def create_BP(alpha, L, H, dx):
    x = np.arange(L, H, dx)
    x = np.append(x,H)
    y = [(1-(L/_x)**alpha)/(1-(L/H)**alpha) for _x in x]
    return x, y

def create_EXP(mean, H, dx):
    x = np.arange(0, H, dx)
    x = np.append(x, H)
    _lambda = 1/mean
    y = [1 - np.exp(- _lambda * _x) for _x in x]
    return x, y

class Func(object):
    """Container for x and y"""

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.min_x = np.min(x)
        self.min_y = np.min(y)
        self.max_x = np.max(x)
        self.max_y = np.max(y)


class Cdf(object):
    """Complete this class to support analytical CDFs"""
    
    def __init__(self, cdf_desc):
        self.cdf_desc = cdf_desc

        if self.cdf_desc['name'] == 'BP':
            if 'alpha' not in self.cdf_desc or 'L' not in self.cdf_desc or 'H' not in self.cdf_desc:
                raise Exception('Missing parameters for the BP distribution')

            self._F = Func(
                *create_BP(self.cdf_desc['alpha'], self.cdf_desc['L'], self.cdf_desc['H'], self.cdf_desc['dx']))
        elif self.cdf_desc['name'] == 'EXP':
            self._F = Func(*create_EXP(self.cdf_desc['mean'], self.cdf_desc['H']))
        else:
            raise NotImplementedError('cdf not supported')
        
        self.dx = self.cdf_desc['dx']
        self._Fi = Func(self._F.y, self._F.x)
        self._nF = Func(self._F.x, [1 - it for it in self._F.y])
        self.avg = self.average(self._F.min_x, self._F.max_x)
    def F(self, x):
        """Returns F(x)"""
        return np.interp(x, self._F.x, self._F.y)

    def Fi(self, x):
        """Returns F^(-1)(x)"""
        return np.interp(x, self._F.y, self._F.x)

    def integrate(self, a, b):
        """ Calculates \int_a^b f(x)dx """
        f = self._F
        _x = np.arange(a, b, self.dx)
        np.append(_x, b)
        _y = np.interp(_x, f.x, f.y)
        return np.trapz(_y, _x, self.dx)

    def average(self, a=None, b=None, norm=1):
        """ Calculates \int_a^b xf(x)dx """
        a = np.nanmax([np.float64(a), self._F.min_x])
        b = np.nanmin([np.float64(b), self._F.max_x])
        return (b*self.F(b) - a*self.F(a) - self.integrate(a, b))/norm

    def equal_splits(self, n, a=0, b=1):
        """ Split the CDF in eequal parts on the y axis """
        return [self.Fi(y) for y in np.linspace(a, b, n+1)]

    def proportional_split(self, N, target_load=None, a=None, b=None, precision=5):
        """ Splits the CDF proportionally to the amount of traffic that each split will observe """
        
        if a is None:
            a = 0 # here has to be zero ---> WHY ?!!
        if b is None:
            b = self.Fi(1)
        if target_load is None:
            target_load = self.avg/N
        
        th = [a]
        for i in range(N):
            res = self._midpoint_search(
                th[i], 
                th[i], 
                b, 
                target_load,
                precision=precision)
            th.append(res)
        return th

    
    def bounded_proportional_split(self, N, a, **kwargs):
        """ Splits the CDF below a, such that the total load --- excluded the load on splits
        above a --- handled on each split is the same """
        
        # for each queue (in this case corresponding to a server) we want to have the same load,
        # which is (total load - load of the last queue / N:
        # = \int_a^{\inf} (x-a)f(x)dx 
        target_load = self._per_queue_load(0,a) / N
        th = self.proportional_split(N-1, target_load, b=a, **kwargs)
        th.append(a)
        return th
    
    def _per_queue_load(self, a_l, a_h):
        """ returns the load on the interval [a_l, a_h] 
        i.e. \int_a_l^a_h (x-a_l)f(x)dx + (a_h - a_l) \int_a_h ^ \inf f(x)dx"""
        # safety check : if a_l less than minimum of the support, we substract zero 
        # equivalent to impose a_l = 0
        return (self.average(a_l, a_h) + a_h*(1 - self.F(a_h)) - (a_l > self.Fi(0))*a_l*(1 - self.F(a_l)))

    def _midpoint_search(self, a, l, r, target_load, precision=5):
        """
        Returns threshold th that gives target_load on the interval [a,th] 
        using binary search in the interval [a,r]
        """
        if l == r:
            """ Maximum precision reached, return"""
            return l
        mid = l + (r - l)/2.0
        load = self._per_queue_load(a, mid)
        
        if precision_round(load, precision) == precision_round(target_load, precision):
            return mid
        elif load > target_load:
            return self._midpoint_search(a, l, mid, target_load, precision)
        else:
            return self._midpoint_search(a, mid, r, target_load, precision)
    
    # redundant with get queue load
    #def get_load_below_cut(self, cut):
    #    """ Returns load of flows """
    #    elephant_load = (self.average(cut, self.Fi(1)) 
    #                        - cut * (1-self.F(cut)))
    #    return self.avg - elephant_load

    def var(self, a, b):
        k = (self.cdf_desc['alpha'] * self.cdf_desc['L']**self.cdf_desc['alpha']) / \
            (1 - self.cdf_desc['L']/self.cdf_desc['H'])**self.cdf_desc['alpha']
        return k/(-self.cdf_desc['alpha'])*(b**(-self.cdf_desc['alpha']) - a**(-self.cdf_desc['alpha']))

class BoundedPareto(Cdf):

    def __init__(self, cdf_desc):
        self.L = cdf_desc['L']
        self.H = cdf_desc['H']
        self.alpha = cdf_desc['alpha']
        if self.alpha < 0 or self.alpha > 2:
            raise Exception('alpha must be 0 <= alpha <= 2')
        self.avg = self.expectation()
        """ create bounded pareto object with only extremes of the support
            to be compliant with Cdf class, but all computations are done 
            with analytical formulas so we don't need all the points. This is NOT
            a clean solution but it does its job by now"""
        self._F = Func(*create_BP(self.alpha, self.L, self.H, dx= self.H - self.L))
        
    def clip_to_support(self, x):
        # convert to numpy array
        if not isinstance(x,np.ndarray):
            x = np.array(x,dtype='float')   
        x[x < self._F.min_x] = self._F.min_x
        x[x > self._F.max_x] = self._F.max_x
        return x

    # returns pdf on x values
    def pdf(self, x):
        x = self.clip_to_support(x)
        num = self.alpha * self.L ** self.alpha * x ** (-self.alpha-1)
        den = 1-(self.L/self.H)**self.alpha
        return num / den
    
    # cdf
    def F(self, x):
        x = self.clip_to_support(x)
        return (1 - (self.L / x) ** self.alpha) / (1 - (self.L / self.H) ** self.alpha)

    # inverse cdf
    def Fi(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y,dtype='float')
        if np.any(y < 0) or np.any(y > 1):
            raise Exception('y out of range [0,1]')
        num = - (y * self.H ** self.alpha - y * self.L **
                 self.alpha - self.H ** self.alpha)
        den = (self.H * self.L) ** self.alpha
        return (num/den) ** (-1/self.alpha)

    # survivor
    def nF(self, x):
        x = self.clip_to_support(x)
        return 1 - self.F(x)

    # value of \int_0^x xf(x) dx
    def G(self, x):
        x = self.clip_to_support(x)
        num = self.alpha * (x * self.L ** self.alpha - x ** self.alpha * self.L)
        den = (1-self.alpha) * (self.H ** self.alpha - self.L ** self.alpha)
        return (self.H / x) ** self.alpha * num / den

    # Average between a and b.
    def average(self, a=None, b=None, norm=1):
        if a is None:
            a = self.L
        if b is None:
            b = self.H
        return (self.G(b) - self.G(a))/norm
    
    # Expected value
    def expectation(self):
        return self.moment(1)

    # Variance
    def variance(self):
        return self.moment(2)

    #back compatible
    def var(self, a, b):
        k = (self.alpha * self.L**self.alpha) / \
            (1 - self.L/self.H)**self.alpha
        return k/(-self.alpha)*(b**(-self.alpha) - a**(-self.alpha))

    # n-th moment
    def moment(self, n=1):
        return self.alpha / ((n-self.alpha)*(self.H**self.alpha-self.L**self.alpha)) * \
            (self.H**n * self.L**self.alpha - self.L**n * self.H**self.alpha)

class Ecdf(Cdf):
    """Create ECDF from file"""

    def __init__(self, cdf_desc):
        # super().__init__()
        self.cdf_desc = cdf_desc
        if type(cdf_desc) is dict:
            assert 'x' in cdf_desc
            assert 'y' in cdf_desc
            self._F = Func(cdf_desc['x'], cdf_desc['y'])
        else:
            self._F = Func(*self._load_cdf_from_file(self.cdf_desc))
        self._Fi = Func(self._F.y, self._F.x)
        self._nF = Func(self._F.x, [1 - it for it in self._F.y])
        self.avg = self.average(self._F.min_x, self._F.max_x)

    def _load_cdf_from_file(self, file_name):
        pkts = []
        prob = []

        fp = open(file_name, 'r')
        data = fp.readlines()

        for d in data:
            v, p = d.split(' ')
            pkts.append(float(v))
            prob.append(float(p))

        fp.close()
        return pkts, prob

# wrapper to normalize cdf
class CdfNormalizer():
    # min and max of new support, cdf object
    def __init__(self, cdf, mins = -np.Inf, maxs = +np.Inf):
        self.min_x = max(mins, cdf._F.min_x)
        self.max_x = min(maxs, cdf._F.max_x)
        self.cdf = copy.deepcopy(cdf)
        self.cdf._F.min_x = self.min_x
        self.cdf._F.max_x = self.max_x
        # for compatibility
        self._F = self.cdf._F
        self.avg = self.average()
        
    # cdf
    def F(self, x):
        x = self.cdf.clip_to_support(x)
        return (self.cdf.F(x) - self.cdf.F(self.min_x)) * 1/(self.cdf.F(self.max_x) - self.cdf.F(self.min_x)) 
    
    # inverse cdf
    def Fi(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y,dtype='float')
        if np.any(y < 0) or np.any(y > 1):
            raise Exception('y out of range [0,1]')
        y1 = y / 1/(self.cdf.F(self.max_x) - self.cdf.F(self.min_x)) + self.cdf.F(self.min_x)
        return self.cdf.Fi(y1)

    # survivor
    def nF(self, x):
        x = self.cdf.clip_to_support(x)
        return 1 - self.F(x)
    
    # Average between a and b.
    def average(self, a=None, b=None):
        return self.cdf.average(a,b) * 1/(self.cdf.F(self.max_x) - self.cdf.F(self.min_x))
