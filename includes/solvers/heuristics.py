import numpy as np
from pandas import unique
from includes.functions import *

class BaseSolver():
    def __init__(self, pars):
        self.cdf = pars['cdf']
        self.n = pars['num_queues']

    def solve(self):
        pass


class ESN_Solver(BaseSolver):

    def solve(self):
        return self.cdf.equal_splits(self.n)


class SD_ESN_Linear_Solver(BaseSolver):
    """ Proportional split for LB thresholds + ES-N for inner thresholds """ 

    def __init__(self, pars):
        super().__init__(pars)
        self.k = pars['num_servers']
        self.cdf_cut = pars['cdf_cut']
        print(f'Solving for {self.k} servers and {self.n} queues, with cdf_cut at {self.cdf_cut}')

    def solve(self):
        cdf_cut = self.cdf.Fi(self.cdf_cut)
        lb_demotion_thresholds = self.cdf.bounded_proportional_split(
            N=self.k, 
            a=cdf_cut, 
            precision=8)
        print('Rerouting thresholds:', lb_demotion_thresholds)

        # for each switch in the topology, calculates inner-thresholds with ES-N heuristic
        demotion_thresholds = []
        for i in range(self.k):
            l = self.cdf.F(lb_demotion_thresholds[i])
            r = self.cdf.F(lb_demotion_thresholds[i+1])    
            sub_thresholds = self.cdf.equal_splits(self.n-1, l, r)
            # !!! below code to avoid numeric issues TODO better management of last re-routing 
            # which doesn't suffer from this issue
            sub_thresholds[0] = lb_demotion_thresholds[i]
            sub_thresholds[-1] = lb_demotion_thresholds[i+1]

            demotion_thresholds.extend(sub_thresholds)
        return unique(demotion_thresholds)



class SD_Prop_Linear_Solver(BaseSolver):
    """ Proportional split for all thresholds """ 

    def __init__(self, pars):
        super().__init__(pars)
        self.k = pars['num_servers']
        self.cdf_cut = pars['cdf_cut']
        print(f'Solving for {self.k} servers and {self.n} queues, with cdf_cut at {self.cdf_cut}')

    def solve(self):
        cdf_cut = self.cdf.Fi(self.cdf_cut)
        lb_demotion_thresholds = self.cdf.bounded_proportional_split(
            N=self.k * (self.n-1), 
            a=cdf_cut, 
            precision=8)

        return lb_demotion_thresholds




class SD_Prop_Circular_Solver(BaseSolver):
    """ Takes lp_threshold as input, computes PROPORTIONAL split for deriving 
    thresholds below lp_threshold, 
    then computes other thresholds (i.e., above lp_threshold) such that the load is balanced 
    across all switches. """

    def __init__(self, pars):
        super().__init__(pars)
        self.k = pars['num_servers']
        # cut threshold, after it load balancing
        self.cdf_cut = self.cdf.Fi(pars['cdf_cut'])
        # threshold after which flows go in low priority queue
        self.lp_threshold = pars['lp_threshold']    

    def solve(self):
        hp_thresh = self.compute_pq0_thresholds()
        return self.match_proportional_split(hp_thresh)
    
    def compute_pq0_thresholds(self):
        # --------- proportional split -------------
        hp_thresh = self.cdf.bounded_proportional_split(self.k, a=self.lp_threshold, precision=8)
        print(f'\nProportional splits below {self.lp_threshold} among available servers results in:')
        print(hp_thresh)
        return hp_thresh


    def match_proportional_split(self, hp_thresh):
        # target load assumes we have queue reserved to elephant flows on all switches
        elephant_load = (self.cdf.average(self.cdf_cut, self.cdf.Fi(1)) 
                        - self.cdf_cut * (1-self.cdf.F(self.cdf_cut)))
        target_load = (self.cdf.avg - elephant_load) / self.k
        
        traffic = [self.cdf._per_queue_load(hp_thresh[i-1], hp_thresh[i]) for i in range(1,len(hp_thresh))]
        print('Normalized traffic on each server with only PQ0:')
        print(np.array(traffic)/self.cdf.avg)
        print('Load to match on each server: ', target_load/self.cdf.avg)
        print(f'Elephant will load the remaining: {elephant_load/self.k/self.cdf.avg}')
        missing_load = target_load - np.array(traffic)
        print('\nMissing load on each server: ', missing_load/self.cdf.avg)

        # at this point we have the first K thresholds out of the K x (N-1) that we need
        # we can start computing the remaining ones as follows. For each server, we compute
        # the threshold that would match the target load.
        thresholds = copy.copy(hp_thresh)
        for i in range(len(thresholds)-1, len(thresholds)-1 + (self.k-1)):
            print('Search for threshold between', thresholds[i], 'and', self.cdf_cut, 'that matches', missing_load[i - (len(hp_thresh)-1)])
            res = self.cdf._midpoint_search(
                thresholds[i], 
                thresholds[i],
                self.cdf_cut,
                target_load =missing_load[i - (len(thresholds)-1)])
            thresholds.append(res)

        thresholds.extend([self.cdf_cut, self.cdf.Fi(1)])
        
        return thresholds
