from includes.base import TopologyManager
from includes.server import SPServer, Queue, SPServer_PS, find_queue_for_item_last_server_in_hybrid_spatial
from includes.applications import Sink, Application
from includes.functions import *
from includes.data_analyzer import DataManager

from datetime import datetime
import simpy
import numpy as np
import argparse
import json
import copy
from random import seed, random
import os

from includes.solvers.heuristics import SD_ESN_Linear_Solver, SD_Prop_Circular_Solver, SD_Prop_Linear_Solver

""" BIG TODOs:
    1) Implement a logger module or find an external library
    2) Rethink and augmentthe data processing module
    3) Move the topology management entirely inside the TopologyManager class
    4) Implement better threshold management
    7) Add type hints
    10) In server.py there is an ugly O(n) search for queue index need to write the binary search for it
"""

class Simulation(object):

    """ TODO Manages the overall simulation """

    def __init__(self):
        pass


def progress_bar(env, sim_time, S):
    """ Prints the progress of the simulation """
    interval = sim_time/10
    while(True):
        yield env.timeout(interval)
        print("Simulation at {}%".format(int(env.now/sim_time*100)))
        print('-- queue sizes --')
        for idx, s in enumerate(S):
            print("{}: {}".format(idx, s.get_queue_size()))
        if env.now >= sim_time:
            break


def parse_arguments():
    available_scenarios = [it for it in globals() if 'task_' in it]
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='switchone',
                        help='Specify the experiment scenario. Available scenearios are: {}'.format(available_scenarios))
    parser.add_argument('--data_files', type=str, default='',
                        help='Used only when task=analyze. Comma-separated list of files to load during analysis.')
    parser.add_argument('--num_servers', type=int, default=1,
                        help='Number of servers to use.')
    parser.add_argument('--num_queues', type=int, default=1,
                        help='Number of queues per server.')
    parser.add_argument('--allocation', type=str, default="linear",
                        help='Priority allocation strategy')
    parser.add_argument(
        '--lambda', type=float, help='Arrival rate of flows')
    parser.add_argument(
        '--cdf', type=str, help='The name of the cdf. For supported analytical cdf and format check functions.py')
    parser.add_argument('--sim_time', type=int, default='100000',
                        help='Maximum simulation time limit')
    parser.add_argument('--results_folder', type=str, default="./",
                        help="Folder to store simulation results in.")
    parser.add_argument('--seed', type=int, default=11,
                        help="Seed to use for simulations.")
    parser.add_argument('--postfix', type=str, default='',
                        help='Postfix to append to the simulation task')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        help='List of demotion thresholds to use')
    parser.add_argument('--cut-threshold', type=float,
                        help='Threshold to use as cut-tail for the cdf. Must be expressed as percentile')
    parser.add_argument('--disc', type=str, default='FIFO',
                        help='Serving discipline to use. Default is FIFO')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode', default=False)

    args = parser.parse_args()

    args._lambda = vars(args)['lambda']
    if args.cdf is not None:
        args.cdf_name = args.cdf
        if args.cdf in analytical_cdfs:
            args.cdf = globals()[analytical_cdfs[args.cdf]['class']](analytical_cdfs[args.cdf])
        elif ".cdf" in args.cdf:
            args.cdf = Ecdf('ecdf/{}'.format(args.cdf))
        else:
            raise Exception("Invalid CDF specified!")

    # Append final '/' to the results_folder path if missing
    if args.results_folder[-1] != '/':
        args.results_folder += '/'

    seed(args.seed)
    np.random.seed(args.seed)

    if args.task not in globals():
        raise Exception("Trying to run an unrecognized task!")
    args.task_f = globals()[args.task]

    return args


def task_mlfq(env, par):
    """ parallel MLFQ, thresholds are set with ES-N policy"""
    cdf = par['cdf']
    mu = par['mu']
    _lambda = par['_lambda']
    num_queues = par['num_queues']
    num_servers = par['num_servers']

    if par['disc'] == 'FIFO':
        server_type = SPServer
    elif par['disc'] == 'PS':
        server_type = SPServer_PS
    else:
        raise Exception("Unrecognized serving discipline.")

    T = TopologyManager()
    sink = Sink(env, mu)
    if par['thresholds'] is not None:
        demotion_thresholds = par['thresholds']
        demotion_thresholds = [0, *demotion_thresholds, cdf.Fi(1)]
    else:
        demotion_thresholds = cdf.equal_splits(num_queues)
        demotion_thresholds[0] = 0
    s = []
    for i in range(num_servers):
        Q = [Queue(env, i) for i in range(num_queues)]
        A = Application(cdf, _lambda/num_servers, env)
        S = server_type(
            mu/num_servers, 
            env,
            i,
            debug=par['debug'])
        
        s.append(S)
        
        S.attach_queues(Q, [(demotion_thresholds[i], demotion_thresholds[i+1]) for i in range(num_queues)])
        for q in Q:
            q.attach_server(S)

        # Topology
        T.attach_link(A, Q[0])
        T.attach_link(S, sink)
        # attach queues one to the other for demotion
        for prio in range(num_queues-1):
            print(f'Connecting PQ {prio} to PQ {prio+1}..')
            T.attach_link(Q[prio], Q[prio+1], direction='uni')

    return sink, s, demotion_thresholds


def task_sd_mlfq_pias(env, par):
    """ SD-MLFQ : threshold set to provide optimal load split among servers"""
    cdf = par['cdf']
    mu = par['mu']
    _lambda = par['_lambda']
    num_queues = par['num_queues']
    num_servers = par['num_servers']

    if par['disc'] == 'FIFO':
        server_type = SPServer
    elif par['disc'] == 'PS':
        server_type = SPServer_PS
    else:
        raise Exception("Unrecognized serving discipline.")

    T = TopologyManager()
    sink = Sink(env, mu)
    
    with open("opt_results", "r") as fp:
        line = fp.readline()
        while line:
            l = json.loads(line)
            if l['K'] == num_servers and l['N'] == num_queues and l['lambda'] == _lambda:
                demotion_thresholds = l['x']
                break
            line = fp.readline()

    #print(demotion_thresholds)

    demotion_thresholds.insert(0, cdf.Fi(0)); demotion_thresholds.append(cdf.Fi(1))
    
    assert num_queues*num_servers == len(demotion_thresholds) - 1

    print(demotion_thresholds)
    demotion_thresholds = [demotion_thresholds[i*num_queues:(i+1)*num_queues+1] for i in range(num_servers)]
    #split_th = np.split(np.array(demotion_thresholds), num_servers)
    #print([list(split_th[i]) for i in range(num_servers)])
    #exit()

    S = [server_type(mu/num_servers, env, demotion_thresholds[i], i, debug=par['debug'])
         for i in range(num_servers)]
    A = Application(cdf, _lambda, env)

    Q = []
    for i in range(num_servers):
        q = [Queue(env, i) for i in range(num_queues)]
        Q.append(q)

        S[i].attach_queue(q)
        for _q in q:
            _q.attach_server(S[i])

    # Topology
    # Attach Application to the first queue of the first server
    T.attach_link(A, Q[0][0])

    for i in range(len(S)-1):  # Attach servers one after another
        T.attach_link(S[i], Q[i+1][0])
        T.attach_link(S[i], sink)
    T.attach_link(S[-1], sink)

    
    return sink, S

def task_sd_mlfq(env, par):
    # TODO re-do connection between queues according to new mechanism i.e., directly connect
    # queues and not servers !!
    """ SD-MLFQ, thresholds set to provide proportional load split among servers"""
    cdf = par['cdf']
    mu = par['mu']
    _lambda = par['_lambda']
    num_queues = par['num_queues']
    num_servers = par['num_servers']

    if par['disc'] == 'FIFO':
        server_type = SPServer
    elif par['disc'] == 'PS':
        server_type = SPServer_PS
    else:
        raise Exception("Unrecognized serving discipline.")

    T = TopologyManager()
    sink = Sink(env, mu)
    #demotion_thresholds = cdf.proportional_split(num_queues*num_servers); demotion_thresholds[0] = 0; demotion_thresholds[-1] = cdf.Fi(1)
    lb_demotion_thresholds = cdf.proportional_split(num_servers)
    lb_demotion_thresholds[0] = 0
    lb_demotion_thresholds[-1] = cdf.Fi(1)
    print('LB thresholds:', lb_demotion_thresholds)
    #print(cdf.equal_splits(num_queues, cdf.F(lb_demotion_thresholds[0]), cdf.F(lb_demotion_thresholds[1])))
    demotion_thresholds = []
    for i in range(len(lb_demotion_thresholds)-1):
        subth = cdf.equal_splits(num_queues, 
                             cdf.F(lb_demotion_thresholds[i]), 
                             cdf.F(lb_demotion_thresholds[i+1]))
        if i==0:
            subth[0] = 0
        demotion_thresholds.extend(subth)
    # print(demotion_thresholds)
    #print([demotion_thresholds[i*num_queues:(i+1)*num_queues+1] for i in range(num_servers)])
    split_th = np.split(np.array(demotion_thresholds), num_servers)
    #print([list(split_th[i]) for i in range(num_servers)])
    
    S = [server_type(mu/num_servers, env, i, debug=par['debug'])
         for i in range(num_servers)]
    A = Application(cdf, _lambda, env)

    Q = []
    for i in range(num_servers):
        q = [Queue(env, i) for i in range(num_queues)]
        Q.append(q)

        queues_job_size_ranges = [(split_th[i][j], split_th[i][j+1]) for j in range(num_queues)]
        S[i].attach_queues(q, queues_job_size_ranges)
        for _q in q:
            _q.attach_server(S[i])

    # Topology
    # Attach Application to the first queue of the first server
    T.attach_link(A, Q[0][0])

    for i in range(len(S)-1):  # Attach servers one after another
        T.attach_link(S[i], Q[i+1][0])
        T.attach_link(S[i], sink)
    T.attach_link(S[-1], sink) # last server to sink only
    
    
    return sink, S

def task_sd_mlfq_hybrid(env, par):
    """ SD_MLFQ with hybrid thresholds : high percentiles of flow size distribution are load balanced across servers, 
    while spatial diversity is applied to other percentiles """
    
    cdf = par['cdf']
    mu = par['mu']
    _lambda = par['_lambda']
    num_queues = par['num_queues']
    num_servers = par['num_servers']
    
    if par['disc'] == 'FIFO':
        server_type = SPServer
    elif par['disc'] == 'PS':
        server_type = SPServer_PS
    else:
        raise Exception("Unrecognized serving discipline.")

    
    T = TopologyManager()
    sink = Sink(env, mu)
    
    cdf_cut = par['cut_threshold']

    # get load balance thresholds according to provided strategy
    pars = {
        'num_servers': num_servers,
        'num_queues': num_queues,
        'cdf': cdf,
        'cdf_cut': cdf_cut
    }

    if par['thresholds'] is not None:
        res = par['thresholds']
        res = [0, *res, cdf.Fi(1)]
    else:
        if par['allocation'] == 'linear-esn':
            solver= SD_ESN_Linear_Solver(pars)
        elif par['allocation'] == 'linear-prop':
            solver= SD_Prop_Linear_Solver(pars)
        elif par['allocation'] == 'circular':
            raise Exception("TODO exception, not implemented yet")
            pars['lp_threshold'] = par['thresholds'][0] # TODO manage differently
            solver= SD_Prop_Circular_Solver(pars)
        else:
            raise Exception("Unrecognized allocation strategy.")
        res = solver.solve()
        res = np.append(res, cdf.Fi(1))
    
    print('Thresholds:', res)

    # map: prio -> (a_l, a_h)
    demotion_thresholds = {}
    for j in range(len(res)-2):
        demotion_thresholds[j] = (res[j],res[j+1])
        demotion_thresholds[(num_queues-1)*num_servers] = (res[-2], res[-1]) #(cdf.Fi(cdf_cut), cdf.Fi(1))
    
    print('Priority ranges:', demotion_thresholds)
    
    S = [server_type(
            mu/num_servers, 
            env, 
            i,
            debug=par['debug']
        ) for i in range(num_servers-1)]
    
    # last server chooses a random destination when overshoot happens
    S.append(server_type(
            mu/num_servers,
            env,
            num_servers-1,
            find_queue_for_item=find_queue_for_item_last_server_in_hybrid_spatial,
            debug=par['debug']
        ))
    
    A = Application(cdf, _lambda, env)
    
    Q = []  # all queues
    priority_to_queue = {} # map : priority -> queue object
    lb_probabilities = []
    for i in range(num_servers):
    
        server_queues = [Queue(env, j) for j in range(num_queues)]
        Q.append(server_queues)

        queues_job_size_ranges = []
        if 'linear' in par['allocation']:
            for j in range((num_queues-1)*i,(num_queues-1)*(i+1)):
                queues_job_size_ranges.append(demotion_thresholds[j])
                q = server_queues[j%(num_queues-1)]
                priority_to_queue[j] = q
                print(f'PQ{j} mapped to Q{q.queue_idx} on server {i}')
        elif par['allocation'] == 'circular':
            for j in range(i, num_servers*(num_queues-1), num_servers):
                queues_job_size_ranges.append(demotion_thresholds[j])
                q = server_queues[j//num_servers]
                priority_to_queue[j] = q
                print(f'PQ{j} mapped to Q{q.queue_idx} on server {i}')
        else:
            raise Exception("Unrecognized allocation strategy.")
        

        # before appending range of lowest priority queue, 
        # calculate load balance probabilities that compensate if the load is not uniform
        a_l = queues_job_size_ranges[0][0]
        a_h = queues_job_size_ranges[-1][-1]
        print(f'Calculating load balance probabilities for range [{a_l}, {a_h}]')
        
        # load on server i with current threshold allocation
        load_i = cdf._per_queue_load(a_l, a_h)
        print(f'Load in server {i} (no elephants):', load_i)
        if load_i > cdf.avg / num_servers:
            raise Exception(f"Traffic allocation is not feasible ! Load on server {i} is already greater than "
                            "load / num_servers. There is nothing we can do to compensate with elephants."
                            "\nTry with different thresholds")
        # overall elephant load 
        elephant_load = cdf.avg - cdf._per_queue_load(0, cdf.Fi(cdf_cut))
        print('Elephant load:', elephant_load)
        print('Average load:', cdf.avg)
        # solve : load_i + gamma * elephant_load = average_load/K
        gamma = (cdf.avg / num_servers - load_i) / elephant_load
        lb_probabilities.append(gamma)
        
        # hybrid scheme: last queue is devoted to load balance flows (low priority)
        queues_job_size_ranges.append(demotion_thresholds[(num_queues-1)*num_servers])

        S[i].attach_queues(
            server_queues, 
            queues_job_size_ranges
        )
        for _q in server_queues:
            _q.attach_server(S[i])

    print('Load balance probabilities:', lb_probabilities)
    for i in range(num_servers):
        S[i].lb_weights = lb_probabilities
    
    # Topology
    # Attach Application to the first queue of the first server
    T.attach_link(A, Q[0][0], direction='uni')

    # Attach all servers to sink
    for i in range(num_servers):
        T.attach_link(S[i], sink)
    
    # attach queues one after the other, in order of priority
    for prio in range(num_servers*(num_queues-1)-1):
        print(f'Connecting PQ {prio} to PQ {prio+1}..')
        T.attach_link(priority_to_queue[prio], priority_to_queue[prio+1], direction='uni')

    # attach one-to-last queue of last server to all lowest priority queues on other servers
    q = priority_to_queue[num_servers*(num_queues-1)-1]
    for j in range(num_servers):
        T.attach_link(q, Q[j][-1])
        print(f'Connecting PQ{num_servers*(num_queues-1)-1}'
              f' to Q{Q[j][-1].queue_idx} on server {Q[j][-1].server.server_idx}..')
    
    return sink, S, res


def bulk_run(par, analyzer) -> DataManager:
    
    sim_par = copy.deepcopy(par)
    sim_par['mu'] = sim_par['cdf'].average(
        0, sim_par['cdf']._F.max_x)  # Serving rate of the server

    
    _lambda = par['lambda']
    print("Running simulation with parameters: {} (lambda={})".format(par, _lambda))
    env = simpy.Environment()  # New environment every time to reset everything
    sim_par['_lambda'] = _lambda

    # Build and run rimulation
    sink, servers, thresholds = par['task_f'](env, sim_par)
    env.process(progress_bar(env, par['sim_time'], servers))

    env.run(par['sim_time'])  # Run simulation

    # Get statistics
    data = sink.finalize()
    
    data['load'] = _lambda 
    # add thresholds if computed at runtime
    analyzer.exp_pars['demotion_thresholds'] = thresholds   
    analyzer.add_data(data)
    print('time idle:', [s.idle_time for s in servers])
    print('measured loads:', [1 - s.idle_time/par['sim_time'] for s in servers])
    print('average load:', sum([1 - s.idle_time/par['sim_time'] for s in servers])/len(servers))


    return analyzer


def task_analyze(args):
    data_files = args.data_files.split(',')

    if not data_files:
        raise Exception("Please specify data files for analysis")

    A = DataManager()
    for f in data_files:
        d = A.load_file(f)
        A.merge_experiments(d)

    #A.plot_fct_vs_flow_size(_lambda=args._lambda, normalized=True)
    A.plot_fct_vs_load(normalized=True, legend_col='dataset')
    #A.plot_experiments(log_y=False, normalized=True)


def main():
    args = parse_arguments()
    if args.task == 'task_analyze':
        task_analyze(args)
    else:
        experiment_parameters = vars(args)

        dm = DataManager()  # Gathers and processes simulation data
        dm.exp_pars = dict((k, vars(args)[k]) for k in [
                            'task', 
                            'num_servers', 
                            'num_queues', 
                            'sim_time', 
                            '_lambda', 
                            'cdf_name', 
                            'thresholds',
                            'cut_threshold',
                            'disc'])
        bulk_run(experiment_parameters, dm)  # Run experiments

        if args.postfix != '':
            dm.exp_pars['task'] = "{}-{}".format(args.task, args.postfix)
        
        if not os.path.exists(args.results_folder):
            os.makedirs(args.results_folder)
        
        dm.save_data("{}{}".format(args.results_folder, datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")))  # Save to file


if __name__ == '__main__':
    main()
