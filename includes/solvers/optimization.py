import itertools

import pandas as pd
from includes.functions import *
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from scipy.optimize import basinhopping, minimize, curve_fit


class SolverPIAS(object):
    def __init__(self, pars, a0=None, an=None):
        self._cdf = pars['cdf']
        self._mu = self._cdf.avg
        self._lambda = pars['lambda']
        self.N = pars['num_queues']-1  #number of thresholds to resolve
        self.step = pars['step']
        
        self.a0 = a0
        self.an = an
        if self.a0 is None:
            # thisis conviently set to 0, not Fi(0). So you ca iterate over ais and use ais[0] as a lower limit
            # of the integral for the expected number of packets in the queue
            self.a0 = 0
        if self.an is None:
            self.an = self._cdf.Fi(1)

    def waiting_times(self, ais):
        """ Calculates the waiting times $T_i$ for different queues given as input $a_i$ 
        note that $F(a_i) - F(a_{i-1}) = \theta_i $ 
        ais : array of thresholds, including the min and max of the CDF a0 = F(0), aN = F(1)
        """
        _mu = []  # \mu_i
        _lambda = []  # \lambda_i
        t = []  # waiting times T_i

        for i in range(len(ais)-1):  # iterate over the number of queues - 1
            # \lambda_i is given by \lamnda * E[L_i]
            _lambda.append(
                self._lambda*self.expected_packets(ais[i], ais[i+1]))
            
            _mu.append(self._mu)
            for j in range(i):
                _mu[i] = _mu[i]*(1 - _lambda[j]/_mu[j])
            # Waiting time is $1/(\mu_i - \lambda_i)$
            t.append(1/(_mu[i] - _lambda[i]))

        return t
    

    def queue_loads(self, ais):

        _mu = []  # \mu_i
        _lambda = []  # \lambda_i
        rhos = []  # loads queue_i

        for i in range(len(ais)-1):  # iterate over the number of queues - 1
            # \lambda_i is given by \lamnda * E[L_i]
            _lambda.append(
                self._lambda*self.expected_packets(ais[i], ais[i+1]))
            _mu.append(self._mu)
            for j in range(i):
                _mu[i] = _mu[i]*(1 - _lambda[j]/_mu[j])
            # Waiting time is $1/(\mu_i - \lambda_i)$
            rhos.append(_lambda[i]/_mu[i])

        return rhos, _mu

    def expected_packets(self, a, b):
        """Calculates expected number of packets thate are brought in a queue which serves flows with size in [a, b]. 
        Equilvalent to $E[L_i]$ in PIAS"""
        return self._cdf._per_queue_load(a, b)

    def total_cost(self, ais):
        """ Compute the total cost function given $a_i$ """
        _ais = np.concatenate((
            [self.a0], 
            sorted(list(ais)),
             [self.an]), )  
        # For simplicity we add to the array of a_i the max and the min of the CDF's x values. 
        # E.g., for 1 threshold we will have a = [F^(-1)(0), a_1, F^(-1)(1)]
        # Clip ais to maximum and minimum values to avoid memory exceptions
        
        # Clip ais values before computing the total cost to avoid memory issues
        if np.any(_ais > self._cdf._F.max_x) or np.any(_ais < self._cdf._F.min_x):
            for i in range(len(_ais)):
                if _ais[i] < self._cdf._F.min_x:
                    _ais[i] = self._cdf._F.min_x
                if _ais[i] > self._cdf._F.max_x:
                    _ais[i] = self._cdf._F.max_x
        # Get the waiting time in each queue given the a_i. 
        # Following the previous example the first queue will serve pkts in [F^(-1)(0), a_1] 
        # and the second in [a_1, F^(-1)(1)]
        t = np.array(self.waiting_times(_ais))
        rho, _mu = self.queue_loads(_ais)
        rho = np.array(rho)
        _mu = np.array(_mu)
        
        _Fais = self._cdf.F(_ais)  # Get the corresponding F(a_i)
        # Compute $\theta_i = F(a_i) - F(a_i-1)$. Since we start from i=0 we compute instead $\theta_i = F({a_i+1}) - F(a_i)$
        thetas = [_Fais[i+1] - _Fais[i] for i in range(len(_Fais)-1)]
        W = 0  # Total waiting time that we will try to minimize
        # print(_ais)
        # print(thetas)
        # print(t)
        
        """ cost function using M/G/1-PS model T(x) = X/C(1-rho), where C is capacity """
        for i in range(len(thetas)):
            norm_cdf = CdfNormalizer(self._cdf, mins=_ais[i], maxs=_ais[i+1])
            # average job size in the interval [ais[i], ais[i+1]]
            avg_queue_job_size = norm_cdf.average()
            # each \theta_i grup of flows pays time for serving _a_i in queues before i, then 
            # for serving remaining size in queue i. We normalize the weighting time of each class
            # of flows by the FCT in empty system
            
            # can be nan if ais[i] == ais[i+1] then truncated cdf is a delta
            if not np.isnan(avg_queue_job_size):    
                #print(i)
                #print(_ais[1:i+2])
                #print(_mu[:i+1])
                assert len(_ais[1:i+1]) == len(_mu[:i])
                W += (thetas[i] * (sum( _ais[1:i+1] / (_mu[:i]*(1-rho[:i])) ) 
                                   + (avg_queue_job_size - _ais[i]) / (_mu[i]*(1-rho[i])) )) / (avg_queue_job_size/self._mu)
            else:
                return 1e8
        """ 
        # Cost function from PIAS paper
        for i in range(len(t)):
            if sum(thetas[i:]) > 0: # avoids 0 * inf = nan
               W += t[i]*sum(thetas[i:])  
        """
        
        # The objective function evolves slowly. Let's try to increase the variation
        # tmax = bool(np.all(ais <= self._cdf._F.max_x))
        # tmin = bool(np.all(ais >= self._cdf._F.min_x))
        
        # if not tmax:
        #     cost_overrun = (max(ais) - self._cdf._F.max_x)**2
        # elif not tmin:
        #     cost_overrun = (np.abs(min(ais)) - self._cdf._F.min_x)**2
        # else:
        #     cost_overrun = 0.0

        return W #+ cost_overrun

    def solve(self):
        if self.N == 3:
            self.solve_3_th_bruteforce()
        elif self.N == 1:
            self.solve_1_th_bruteforce()
        elif self.N == 2:
            return self.solve_2_th_bruteforce()
        else:
            raise Exception("Bruteforce supports only 1,2 or 3 thresholds")

    def solve_3_th_bruteforce(self):
        res = -1  # Store the total best cost

        # Iterate for the first threshold a_0 between [F^{-1}(0). F^{-1}(1)]
        for i in np.arange(self._cdf._F.min_x, self._cdf._F.max_x, self.step):
            print(i)  # Just to check how fast it is
            # Iterate for the second threshold a_1 between [a_0,  F^{-1}(1)]
            for j in np.arange(i, self._cdf._F.max_x, self.step):
                # Iterate for the third threshold a_2 between [a_1,  F^{-1}(1)]
                for k in np.arange(j, self._cdf._F.max_x, self.step):
                    # Check solution and update is it is better than the previous one
                    if self.total_cost([i, j, k]) < res or res < 0:
                        res = self.total_cost([i, j, k])
                        best_solution = [i, j, k]
        print(best_solution)
    
    def plot_cost_function(self, costs, ais):
        fig = plt.figure()
        if self.N == 1:
            for _lambda in costs:
                # Renormalize the waiting time to get the results in function of the average sized packet
                plt.loglog(ais, np.array(costs[_lambda]), lw='2')
            legend = ["$\lambda={:.2f}$".format(it) for it in costs]
            plt.xlabel("Split threshold")
            plt.ylabel("Waiting time")
            plt.legend(legend)
            plt.show()
        elif self.N == 2:
            # create a surface plot with the jet color scheme
            axis = plt.axes(projection='3d')
            axis.plot_surface(ais[0], ais[1], costs, cmap='jet')
            # show the plot
            plt.xlabel('$a_0$')
            plt.ylabel('$a_1$')
            plt.show()
        
    def solve_1_th_bruteforce(self, verbose=True):
        lb_threshold = {}  # Holds the optimal LB thresholds
        lambdas = np.arange(0.1, 1, 0.1)
        lambdas = np.append(lambdas, [0.95, 0.99])
        #lambdas = [0.8]
        costs = {}  # total cost for all lambdas to be plotted later
        ais = np.arange(self._cdf._F.min_x, self._cdf._F.max_x, self.step)
        ais = np.append(ais, self._cdf._F.max_x)
        # Cut the cdf at the proportional split threshold for simplicity
        # ais = np.arange(self._cdf._F.min_x,
        #                self._cdf.proportional_split(2)[1]+2*self.step, self.step)

        for it in lambdas:
            self._lambda = round(it, 2)
            if verbose:
                print("Running lambda={}".format(self._lambda))
            # brute-force: try all possible a_i, get the cost and save it in dictionary for all lambdas
            costs[self._lambda] = [self.total_cost([ai]) for ai in ais]
            # best a_i is the one that minimizes the cost
            lb_threshold[self._lambda] = ais[np.argmin(costs[self._lambda])]

        if verbose:
            print(lb_threshold)
            #print(costs)
            self.plot_cost_function(costs, ais)
            
        return lb_threshold
    
    def solve_2_th_bruteforce(self, verbose=True):
        # define range for input
        r_min = self._cdf._F.min_x
        r_max = self._cdf._F.max_x
        # sample input range uniformly
        ais = np.arange(r_min, r_max, self.step)
        ais = np.append(ais, self._cdf._F.max_x)
        x, y = np.meshgrid(ais, ais)

        print(f'Running brute force for N={self.N}..')
        costs = []
        min_cost = np.inf
        best_solution = None
        for i in range(len(ais)):
            costs.append([])
            for j in range(len(ais)):
                a0, a1 = x[i][j], y[i][j]
                if a0 < a1:
                    c = self.total_cost([a0, a1])
                    if c < min_cost:
                        min_cost = c
                        best_solution = [a0, a1]
                        print('New best {} at {}'.format(min_cost, best_solution))
                else:
                    c = np.nan
                costs[i].append(c)
        #print('Plotting cost function')
        #
        print(min_cost, best_solution)
        self.plot_cost_function(np.array(costs), [x,y])
        return best_solution

class SolverLB(SolverPIAS):
    def waiting_times(self, ais):
        """ Calculates the waiting times $T_i$ for different queues given as input $a_i$ note that $F(a_i) - F(a_{i-1}) = \theta_i $"""
        _mu = [self._mu/(len(ais)-1) for i in ais]  # \mu_i
        _lambda = []  # \lambda_i
        t = []  # waiting times T_i

        for i in range(len(ais)-1):  # iterate over the number of queues - 1
            # \lambda_i is given by \lamnda * E[L_i]
            _lambda.append(
                self._lambda*self.expected_packets(ais[i], ais[i+1]))
            # Waiting time is $1/(\mu_i - \lambda_i)$
            t.append(1/(_mu[i] - _lambda[i]))

        return t if np.all(np.array(t) > 0) else [np.inf for i in t]

    def plot_cost_function(self, costs, ais):       
        self.plot_cost(costs, ais)
        self.plot_load(costs, ais)
        self.plot_cost_gain(costs)
        plt.show()

    def plot_cost(self, costs, ais):
        plt.figure()
        for _lambda in costs:
            # Renormalize the waiting time to get the results in function of the average sized packet
            plt.loglog(ais, np.array(costs[_lambda])*self._mu, lw='2')
        plt.axvline(x=self._cdf.proportional_split(
            2)[1], color='k', linestyle='--', alpha=0.5)
        legend = ["$\lambda={:.2f}$".format(it) for it in costs]
        legend.append("Equal workload split")
        plt.xlabel("Split threshold for spines")
        plt.ylabel("Waiting time")
        plt.legend(legend)

    def plot_cost_gain(self, costs):
        lambdas = [it for it in costs]

        t_es = [self.total_cost([self._cdf.proportional_split(2)[1]])
                for self._lambda in lambdas]
        t_opt = [np.min(costs[l]) for l in lambdas]

        plt.figure()
        plt.plot(lambdas, np.array(t_es)/np.array(t_opt))
        plt.xlabel("$\lambda$")
        plt.ylabel("FCT gain ($FCT_{PRS}/FCT_{OPS}$)")

    def plot_load(self, costs, ais):
        load1 = []
        lambdas = [it for it in costs]

        for it in lambdas:
            # take optimal solution for this lambda and compute the load 
            load1.append(self._cdf._per_queue_load(
                self._cdf._F.min_x, 
                ais[np.argmin(costs[it])]) / self._cdf.avg)

        plt.figure()
        plt.plot(lambdas, load1)
        plt.plot(lambdas, [1 - l for l in load1])
        plt.xlabel("$\lambda$")
        plt.ylabel("Per spine load")
        plt.legend(["Spine 1", "Spine 2"])

class PSOSolver(SolverPIAS):
    
    def solve(self):
        lb = [self._cdf._F.min_x+1] * self.N  # Lower bound for a_i
        ub = [self._cdf._F.max_x-1] * self.N  # Upper bound for a_i
        res, _ = pso(self.total_cost, 
                     lb, 
                     ub, 
                     #f_ieqcons=self.sorted_constraints, 
                     maxiter=1000, 
                     omega=.5, 
                     phip=.5,
                     phig=.5, 
                     swarmsize=100,
                     debug=True)
        #res, _ = pso(self.total_cost, lb=lb, ub=ub, maxiter=100, debug=True)
        return sorted(res)

    def sorted_constraints(self, ais):
        """Sets the sorted constraints, a_{i-1} < a_i"""
        N = len(ais)
        return [-ais[i] + ais[i+1] for i in range(N-1)]

class SolverLBHybrid(SolverPIAS):

    def __init__(self, 
                 pars,
                 a0 = None,
                 an = None):
        super().__init__(pars, a0, an)
        
        self.K = pars['num_servers']
        self.Npq = pars['num_queues']
        self.cut_percentile = pars['cdf_cut']
        self.cdf_cut = self._cdf.Fi(self.cut_percentile)

        print("** Brute force solver for hybrid load balancing with spatial diversity **")
        print(f'K = {self.K}')
        print(f'N (thresholds to optimize on each server) = {self.N}')
        print(f'num_queues = {self.Npq}')
        print(f'cut_percentile = {self.cut_percentile}')
        print(f'cutoff = {self.cdf_cut}')

        if self.Npq != 2: # this is set by super class
            raise NotImplementedError('Hybrid solver supports only 2 thresholds')

    def total_cost(self, ais):
        
        """ Compute the total cost function given $a_i$ """
        # Clip ais values before computing the total cost to avoid memory issues
        if np.any(ais > self._cdf._F.max_x) or np.any(ais < self._cdf._F.min_x):
            for i in range(len(ais)):
                if ais[i] < self._cdf._F.min_x:
                    ais[i] = self._cdf._F.min_x
                if ais[i] > self._cdf._F.max_x:
                    ais[i] = self._cdf._F.max_x

        _ais = np.concatenate((
            [self.a0], 
            sorted(list(ais)),
             [self.an]), )  
        
        # we insert cut percentile threshold as the last threshold
        _ais = np.insert(_ais, -1, self.cdf_cut)

        # get the \gamma_i, probability that a flow after cutoff threshold is sent to a given switch
        e = self._cdf.avg - self._cdf._per_queue_load(0, _ais[-2])
        lis = [self._cdf._per_queue_load(_ais[i], _ais[i+1]) for i in range(len(_ais)-2)]
        gammais = [(self._cdf.avg / self.K - lis[i]) / e for i in range(len(lis))]
        
        # Get the waiting time in each queue given the a_i.
        t = self.waiting_times(_ais, gammais)
        # Get the corresponding F(a_i)
        _Fais = self._cdf.F(_ais)
        # Get $\theta_i = F(a_i) - F(a_i-1)$
        thetas = [_Fais[i+1] - _Fais[i] for i in range(len(_Fais)-1)]
        #print(f'waiting times: {t}\n ais:{_ais}\n thetas: {thetas}')
        
        W = 0  # Total waiting time that we will try to minimize
        
        # for all flows with size (0, cutoff)
        for i in range(len(thetas)-1):
            norm_cdf = CdfNormalizer(self._cdf, mins=_ais[i], maxs=_ais[i+1])
            avg_queue_job_size = norm_cdf.average()
            if not np.isnan(avg_queue_job_size):
                #assert len(_ais[1:i+1]) == len(_mu[:i])
                #W += (thetas[i] * (sum( _ais[1:i+1] / (_mu[:i]*(1-rho[:i])) ) + (avg_queue_job_size - _ais[i]) / (_mu[i]*(1-rho[i])) )) 
                total_waiting_time_flow = np.sum(_ais[1:i+1] * t[:i]) + (avg_queue_job_size - _ais[i])*t[i]
                if total_waiting_time_flow < np.inf:
                    W += thetas[i] * total_waiting_time_flow / (avg_queue_job_size/(self._mu/self.K))
                    ##print(f'theta_{i}: {thetas[i]}, sum(t[:i]): {sum(t[:i+1])}')
                else:
                    W += np.inf

        # now add waiting time for flows in (cutoff, inf)
        norm_cdf = CdfNormalizer(self._cdf, mins=_ais[-2], maxs=_ais[-1])
        avg_queue_job_size = norm_cdf.average()
        for i in range(self.K):
            total_waiting_time_flow = np.sum(_ais[1:self.K+1] * t[:self.K]) + (avg_queue_job_size - self.cdf_cut)*t[self.K + i]
            if total_waiting_time_flow < np.inf:
                # TODO support N > 2 we cannot just sum the first K waiting times
                W += gammais[i] * thetas[-1] * total_waiting_time_flow / (avg_queue_job_size/(self._mu/self.K))
            else:
                W += np.inf

        return W
    
    def waiting_times(self, ais, gammais):
        """ 
        Calculates the waiting times $T_i$ for different queues given as input $a_i$ 
        note that $F(a_i) - F(a_{i-1}) = \theta_i $
        
        gammais : array of probabilities that a flow after cutoff threshold is sent to a given switch

        returns: array of waiting times for each queue. Waiting times and queues are ordered by priority, 
        last K queues are low priority queues on all switches i.e., those serving flows larger than cutoff threshold
       """

        _mu = []  
        _lambda = []  
        t = []
        a_cut = ais[-2]

        # iterate over all (ai, ai+1) except the last one, which is after a_cut
        for i in range(len(ais)-2):  
            _lambda.append(self._lambda*self.expected_packets(ais[i], ais[i+1]))
            # all of these queues go high priority TODO support for N>2
            _mu.append(self._mu / self.K)    

        # compute traffic on all switches' low priority queues (a_cut, +\infty). Iterate over all switches
        # (equivalent to the number of gammais)
        for i in range(len(gammais)):
            _lambda.append(self._lambda * gammais[i] * self.expected_packets(a_cut, ais[-1]))    
            # all of these queue are low priority and serve only when the HP queue on the same
            # switch is not active. We model this by multiplying with 1 - \rho_0 
            # TODO support for N>2
            _mu.append(_mu[i] * (1 - _lambda[i]/_mu[i]))

        # we enter in this loop and we have vectors lambda and mus where first K elements are arrivals
        # and service rates for high priority queues, and last K elements are arrivals and service rates
        # for low priority queues. We compute the waiting time for each queue, and we return the vector
        for i in range(len(_lambda)):
            #T = 1/(_mu[i] - _lambda[i]) # M/M/1 queue waiting time
            T = 1/(_mu[i]*(1 - _lambda[i]/_mu[i])) # M/G/1 queue waiting time
            t.append(T)
            
        # if solution is not feasbile, return inf. This happens when 
        # some _lambda_i > E[X]/K, because in this case means a high priority queue
        # somewhere takes more than the fair share of the traffic.
        return t if np.all(np.array(gammais) > 0) else [np.inf for i in t]

    def solve_1_th_bruteforce(self, verbose=True):
        lb_threshold = {}  # Holds the optimal LB thresholds
        lambdas = np.arange(0.1, 1, 0.1)
        lambdas = np.append(lambdas, [0.95, 0.99])
        
        a_cut = self._cdf.Fi(self.cut_percentile)
        print(f'solving with cut threshold = {a_cut}')

        costs = {}  # total cost for all lambdas to be plotted later
        gammais = {}

        ais = np.arange(self.a0, a_cut, self.step)
        ais = np.append(ais, a_cut)
        
        for it in lambdas:
            
            self._lambda = round(it, 2)
            
            if verbose:
                print("Running lambda={}".format(self._lambda))
            # brute-force: try all possible a_i, get the cost and save it in dictionary for all lambdas
            costs[self._lambda] = []
            gis = []
            for ai in ais:
                cost = self.total_cost([ai])
                costs[self._lambda].append(cost)
                #gis.append(gi)

            # best a_i is the one that minimizes the cost
            lb_threshold[self._lambda] = ais[np.argmin(costs[self._lambda])]
            #gammais[self._lambda] = gis[np.argmin(costs[self._lambda])]

        if verbose:
            print('optimal thresholds:')
            print(lb_threshold)
            print('gammais for optimal solution:')
            print(gammais)
            self.plot_cost_function(costs, ais, a_cut)
            
        return lb_threshold
    
    def solve(self):
        if self.K == 2:
            self.solve_1_th_bruteforce()
        else:
            raise Exception("Bruteforce for hybrid load balancing supports only 1 threshold, make sure you have"
                            "set num_servers = 2")
        
    def plot_cost_function(self, costs, ais, a_cut):
        if self.N == 1:
            for _lambda in costs:
                # Renormalize the waiting time to get the results in function of the average sized packet
                plt.loglog(ais, np.array(costs[_lambda]), lw='2')
            legend = ["$\lambda={:.2f}$".format(it) for it in costs]
            plt.title(f"Cut threshold: {a_cut}")
            plt.xlabel("$\\alpha_1$")
            plt.ylabel("Waiting time")
            plt.legend(legend)
            plt.show()

class PSOSolverLBHybrid(SolverLBHybrid):
    # TODO nice print of welcome message, PSO solver instead of brute force. 
    # print of parameter list should go in super class

    def solve(self):
        lb = [self._cdf._F.min_x] * (self.K-1)  # Lower bound for a_i
        ub = [self._cdf.Fi(self.cut_percentile)-1] * (self.K-1)  # Upper bound for a_i
        res, _ = pso(self.total_cost, lb, ub, f_ieqcons=self.sorted_constraints, maxiter=1000, debug=False)
        #res, _ = pso(self.total_cost, lb=lb, ub=ub, maxiter=100, debug=True)
        return res
    
    def sorted_constraints(self, ais):
        """Sets the sorted constraints, a_{i-1} < a_i"""
        return [-ais[i] + ais[i+1] for i in range(len(ais)-1)]
    
    def plot_load(self):
        from collections import defaultdict

        lambdas = np.arange(0.2, 1, 0.2)
        lambdas = np.append(lambdas, [0.95, 0.99])

        loads = defaultdict(list)
        for it in lambdas:
            print('Running lambda = {}'.format(it))
            self._lambda = round(it, 2)
            ais = self.solve()
            ais = [self._cdf._F.min_x] + sorted(ais) + [self._cdf.Fi(self.cut_percentile)]
            for i in range(self.K):
                l = self._cdf._per_queue_load(ais[i], ais[i+1]) / self._cdf._per_queue_load(0, ais[-2])
                loads[i].append(l)

        df = pd.DataFrame(loads)
        y = df.columns
        df['loads'] = lambdas
        df.plot(x='loads', y=y, kind='line')
        plt.ylim(0,1)
        plt.show()

class BruteForceSolverLB_MG1(SolverPIAS):
    def waiting_times(self, ais):
        """ Calculates the waiting times $T_i$ for different queues given as input $a_i$ note that $F(a_i) - F(a_{i-1}) = \theta_i $"""
        _mu = [self._mu/(len(ais)-1) for i in ais]
        _lambda = []
        t = []

        for i in range(len(ais)-1):
            _lambda.append(
                self._lambda*self.expected_packets(ais[i], ais[i+1]))
            rho = _lambda[i]/_mu[i]
            t.append(
                (1 + (1 + self._cdf.var(ais[i], ais[i+1])/_mu[i]**2)/2 * rho/(1-rho)) * _mu[i]
                )
        return t if np.all(np.array(t) >= 0) else [np.inf for i in t]


class ScipySolver(SolverPIAS):
    
    def solve(self):
        N = self.N
        #minimizer_kwargs = {"method": "Nelder-Mead"}
        x0 = self._cdf.equal_splits(N+1)[1:-1] # strip from a0 and ak
        niter = 10
        now = datetime.datetime.now()
        maxstepsize = (self._cdf._F.max_x - self._cdf._F.min_x)/10
        ret = basinhopping(
            func=self.total_cost,
            x0=x0,  
            # accept_test=self.bounds,
            #minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            disp=True,
            #T=0.0001,
            stepsize=maxstepsize
        )
        
        print(sorted(list(ret.x)))

        r = {'lambda': self._lambda, 'cdf': self._cdf.cdf_desc, 'x0': list(x0), 'fun': ret.fun, 'x': sorted(list(
            ret.x)), 'msg': ret.message, 'end_t': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 'elapsed_t': str(datetime.datetime.now() - now)}
        return json.dumps(r)

    def bounds(self, **kwargs):
        """Constraints function for scipy minimize function. Sets the upper bound, the lower bound and the order constrain for the thresholds"""
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self._cdf._F.max_x))
        tmin = bool(np.all(x >= self._cdf._F.min_x))
        is_sorted = bool(np.all(x[:-1] < x[1:]))
        return tmax and tmin  # and is_sorted


class ScipySolverLB(ScipySolver):
    def solve(self, N):
        x0 = self._cdf.proportional_split(N+1)[1:-1]
        niter = 200
        now = datetime.datetime.now()

        ret = basinhopping(
            func=self.total_cost,
            x0=x0,  # strip from a0 and ak
            # accept_test=self.bounds,
            niter=niter,
            # disp=True,
            stepsize=100
        )
        r = {'lambda': self._lambda, 'cdf': self._cdf.cdf_desc, 'x0': x0, 'fun': ret.fun, 'x': list(
            ret.x), 'msg': ret.message, 'end_t': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 'elapsed_t': str(datetime.datetime.now() - now)}
        return json.dumps(r)


class ParetoFitter(object):
    def bounded_pareto(self, x, L, H, alpha):
        num = (1-L**alpha*x**(-alpha))
        den = (1-(L/H)**alpha)
        return num/den

    def fit_ecdf(self, cdf: Cdf):
        popt, pcov = curve_fit(self.bounded_pareto, cdf._F.x, cdf._F.y, [cdf._F.min_x, cdf._F.max_x, 0.5],
                               bounds=((-np.inf, cdf._F.max_x, 0), (np.inf, np.inf, np.inf)))
        print(popt)
        plt.plot(cdf._F.x, cdf._F.y)
        x = np.arange(popt[0], popt[1], 1)

        y = [self.bounded_pareto(_x, popt[0], popt[1], popt[2]) for _x in x]
        return x, y

    def plot_original_and_fit_cdf(self):
        cdf = Ecdf('ecdf/dataMining.cdf')
        x, y = self.fit_ecdf(cdf)
        plt.semilogx(cdf._F.x, cdf._F.y, 'r', label='DM', linewidth=2)
        plt.semilogx(x, y, 'r--', label='BP DM', linewidth=2)

        cdf = Ecdf('ecdf/webSearch.cdf')
        x, y = self.fit_ecdf(cdf)
        plt.semilogx(cdf._F.x, cdf._F.y, 'b', label='WS', linewidth=2)
        plt.semilogx(x, y, 'b--', label='BP WS', linewidth=2)
        plt.legend()
        plt.show()
