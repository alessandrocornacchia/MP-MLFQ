
import os
from includes.functions import *
import itertools

""" Generate command strings for different experiments here. Convention is
that they should be named experiment_<name> and return a list of commands """

def experiment_two_queues():

    """
    Two queues, SD vs ESN, single parameter alpha: cut threshold. Other thresholds are all
    LB thresholds, so they are calculated by the solver to ensure proportional split.
    """
    # generate commands
    commands = []
    S = [3,6,9]
    Q = [3]
    #lambdas = [0.2, 0.5, 0.8]
    lambdas = [0.2, 0.4, 0.6, 0.8]
    #lambdas = [0.6]
    cuts = np.round(np.append(np.arange(0.1, 1.0, 0.1), [0.95, 0.99]),2)
    cdfs = ['ws_bp']
    disc = ['PS']

    #task = ['task_sd_mlfq', 'task_sd_mlfq_hybrid'] # task_esn
    task = ['task_esn', 'task_sd_mlfq_hybrid']
    Nflows = 100000

    # get function name
    basedir = 'two-queues-all-lambdas'

    if not os.path.exists(f'./outputs/{basedir}'):
        os.makedirs(f'./outputs/{basedir}')
    
    for s,q,cdf,d,t,l,cut in itertools.product(S,Q,cdfs,disc,task,lambdas,cuts):
        
        pareto = BoundedPareto(analytical_cdfs[cdf])
        
        cmd = f"py .\markov_simulator.py --task {t} "
        cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
        cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
        cmd += f"--results_folder ./sim_results/{basedir}/{t}_K{s}_N{q}_l{l}_{cdf}_{d}_{cut}/ "
        
        if t == 'task_esn':
            cmd += f"--threshold {pareto.Fi(cut)} "
        elif t == 'task_sd_mlfq_hybrid':
            cmd += f"--cut-threshold {cut} "
        
        cmd += f"> ./outputs/{basedir}/{t}_K{s}_N{q}_l{l}_{cdf}_{d}_{cut}.txt 2>&1"
        
        commands.append(cmd)
    
    return commands



def experiment_vary_elephant_th():
    """ 
    Change cut threshold, 3 queues, linear-prop allocation for sub-thresholds
    """
    # generate commands
    commands = []
    S = [3,6,9]
    Q = [3]
    #lambdas = [0.2, 0.5, 0.8]
    lambdas = [0.2, 0.4, 0.6, 0.8]
    #lambdas = [0.6]
    cuts = np.round(np.append(np.arange(0.5, 1.0, 0.1), [0.95, 0.99]),2)
    cdfs = ['ws_bp']
    disc = ['PS']

    #task = ['task_sd_mlfq', 'task_sd_mlfq_hybrid'] # task_esn
    task = ['task_sd_mlfq_hybrid']
    Nflows = 100000

    # get function name
    basedir = 'sd-mlfq-vs-cut'

    if not os.path.exists(f'./outputs/{basedir}'):
        os.makedirs(f'./outputs/{basedir}')
    
    for s,q,cdf,d,t,l,cut in itertools.product(S,Q,cdfs,disc,task,lambdas,cuts):
        
        pareto = BoundedPareto(analytical_cdfs[cdf])
        
        cmd = f"py .\markov_simulator.py --task {t} "
        cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
        cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
        cmd += f"--results_folder ./sim_results/{basedir}/{t}_K{s}_N{q}_l{l}_{cdf}_{d}_{cut}/ "
        cmd += f"--allocation linear-prop "
        cmd += f"--cut-threshold {cut} "
        
        cmd += f"> ./outputs/{basedir}/{t}_K{s}_N{q}_l{l}_{cdf}_{d}_{cut}.txt 2>&1"
        
        commands.append(cmd)
    
    return commands


# 1666.91
def experiment_two_queues_unequal_split():

    """
    Two queues, SD-MLFQ vs MLFQ, single parameter alpha: cut threshold. Other thresholds are all
    LB thresholds, so they are calculated by the solver to ensure proportional split.

    This experiment is for the case where we have 2 queues and 2 servers, and we want to search for
    optimal split with beta

    """
    # generate commands
    commands = []
    S = [2]
    Q = [2]
    
    lambdas = [0.2, 0.4, 0.6, 0.8]

    cuts = [0.85]
    cdfs = ['ws_bp']
    disc = ['PS']
    betas = [0.1, 0.25, 0.4, 0.8] # load proportion that we want on high-priority queue
    Nflows = 100000
    tasks = ['task_sd_mlfq_hybrid']

    # get function name
    basedir = 'N2-unequal-split'

    if not os.path.exists(f'./outputs/{basedir}'):
        os.makedirs(f'./outputs/{basedir}')
    
    # task sd mlfq hybrid
    if 'task_sd_mlfq_hybrid' in tasks:
        for s,q,cdf,d,l,cut_percentile,beta in itertools.product(S,Q,cdfs,disc,lambdas,cuts,betas):
            
            assert s==2 and q==2

            pareto = BoundedPareto(analytical_cdfs[cdf])
            cut = pareto.Fi(cut_percentile)

            # compute threshold that gives beta portion of load on high-priority queue
            th = pareto._midpoint_search(
                a = 0,
                l = 0,
                r = cut,
                target_load= beta * pareto._per_queue_load(0,cut),
                precision = 8
            )
            
            label = f'task_sd_mlfq_hybrid_K{s}_N{q}_l{l}_{cdf}_{d}_cut_{cut_percentile}_beta_{beta}'

            cmd = f"py .\markov_simulator.py --task task_sd_mlfq_hybrid "
            cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
            cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
            cmd += f"--results_folder ./sim_results/{basedir}/{label}/ "
            cmd += f"--cut-threshold {cut_percentile} "
            cmd += f"--threshold {th} {cut} "   # manually enforce these thresholds
            cmd += f"> ./outputs/{basedir}/{label}.txt 2>&1"
            
            commands.append(cmd)

    if 'task_mlfq' in tasks:
        # task esn
        for s,q,cdf,d,l,cut_percentile in itertools.product(S,Q,cdfs,disc,lambdas,cuts):
            
            assert s==2 and q==2

            pareto = BoundedPareto(analytical_cdfs[cdf])
            cut = pareto.Fi(cut_percentile)
            
            label = f'task_esn_K{s}_N{q}_l{l}_{cdf}_{d}_cut_{cut_percentile}'

            cmd = f"py .\markov_simulator.py --task task_esn "
            cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
            cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
            cmd += f"--results_folder ./sim_results/{basedir}/{label}/ "
            cmd += f"--threshold {cut} "
            cmd += f"--cut-threshold {cut_percentile} "
            cmd += f"> ./outputs/{basedir}/{label}.txt 2>&1"
            
            commands.append(cmd)
            
    return commands




def experiment_two_queues_unequal_split_many_servers():

    """
    Two queues, SD-MLFQ vs MLFQ, single parameter alpha: cut threshold.  In this experiment we compute
    optimal thresholds.
    """
    # generate commands
    commands = []
    
    # spatial diversity thresholds
    opt_thresholds_sd = {
        0.8: {
            2: [406],
            4: [34.96015732,  355.91042985, 1308.24670113],
            6: [17.69662024,  133.36466651,  478.32494279, 1098.90043269, 2004.25019049],
            7: [21.00279514, 148.91260898, 496.09957273, 1088.04940064, 1868.95596688,
                2411.62173439]
        },
        0.4: {
            2: [214.02367314],
            4: [15.68502221,  197.07478094, 1015.38241083],
        },
        0.5: {
            2: [275.4762952],
            4: [17.72343357,  226.69631179, 1083.81642369]
        },
        0.6: {
            2: [323.70640492],
            4: [20.78589757,  259.40838577, 1153.23148183]
        },
        0.7: {
            2: [366.26931075],
            4: [26.02399243,  302.38072543, 1229.42762127]
        }

    }

    # pias thresholds
    opt_thresholds = {
        0.2: {
            1: [29200],
            #2: [103] 
            2: [29200]
        },
        0.4: {
            #2: [1003]
            1: [29200],
            2: [2903],
            #5: [15.446427721317525, 547.0784243139417, 4970.1241097662805, 29199.0]
            5: [14.04227382170775, 269.1894109486644, 2240.157264677822, 10618.198474449578]
        },
        0.5: {
            2: [1287.640219758345],
            1: [29200],
            #5: [16.190060588265244, 568.1310883998677, 4847.940410631738, 23805.958847991933]
            5: [14.20228784343039, 251.17098797076048, 2010.816154509429, 9185.646531062433]
        },
        0.6: {
            #2: [1603]
            1: [29200],
            2: [2403],
            5: [14.48671677944967, 276.43751578742695, 2109.7984745327212, 8997.777707970541]
        },
        0.7: {
            1: [29200],
            2: [2006.4633462127254],
            #5: [16.870769459616, 500.09238267244007, 3519.445459346961, 12724.599184496303],
            5: [15.51006401633511, 346.81182732018493, 2529.222566017736, 9814.235638299515]
        },
        0.8: {
            1: [29200],
            2: [3003],
            #5: [17.35286148,   486.28536303,  3453.40683668, 12106.46670828],
            5: [16.613184432962292, 418.01436812628424, 2976.0156555372528, 10740.039185357855]
        },
    }

    #lambdas = [0.2, 0.4, 0.6, 0.8]
    lambdas = [0.4,0.5,0.6,0.7]
    
    S = [2]
    Q = [1]
    cuts = [0.85]
    cdfs = ['ws_bp']
    disc = ['PS']
    Nflows = 100000
    tasks = ['task_mlfq'] #, 'task_sd_mlfq_hybrid']
    #tasks = ['task_sd_mlfq_hybrid']

    # get function name
    basedir = 'N2-unequal-split-many-servers'

    if not os.path.exists(f'./outputs/{basedir}'):
        os.makedirs(f'./outputs/{basedir}')
    
    # task sd mlfq hybrid
    if 'task_sd_mlfq_hybrid' in tasks:
        for s,q,cdf,d,l,cut_percentile in itertools.product(S,Q,cdfs,disc,lambdas,cuts):
            
            #assert s==2 and q==2

            pareto = BoundedPareto(analytical_cdfs[cdf])
            cut = pareto.Fi(cut_percentile)

            th = opt_thresholds_sd[l][s]

            label = f'task_sd_mlfq_hybrid_K{s}_N{q}_l{l}_{cdf}_{d}_cut_{cut_percentile}'

            cmd = f"py .\markov_simulator.py --task task_sd_mlfq_hybrid "
            cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
            cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
            cmd += f"--results_folder ./sim_results/{basedir}/{label}/ "
            cmd += f"--cut-threshold {cut_percentile} "
            cmd += f"--threshold " + ' '.join([str(thh) for thh in th]) + f" {cut} "   # manually enforce these thresholds
            cmd += f"> ./outputs/{basedir}/{label}.txt 2>&1"
            
            commands.append(cmd)

    if 'task_mlfq' in tasks:
        
        for s,q,cdf,d,l,cut_percentile in itertools.product(S,Q,cdfs,disc,lambdas,cuts):
            
            pareto = BoundedPareto(analytical_cdfs[cdf])
            cut = pareto.Fi(cut_percentile)
            
            th = opt_thresholds[l][q]

            label = f'task_mlfq_K{s}_N{q}_l{l}_{cdf}_{d}_cut_{cut_percentile}'

            cmd = f"py .\markov_simulator.py --task task_mlfq "
            cmd += f"--lambda {l} --num_servers {s} --num_queues {q} "
            cmd += f"--seed 0 --disc {d} --cdf {cdf} --sim_time {int(Nflows/l)} "
            cmd += f"--results_folder ./sim_results/{basedir}/{label}/ "
            cmd += f"--threshold " + ' '.join([str(thh) for thh in th]) + " " 
            cmd += f"--cut-threshold {cut_percentile} "
            cmd += f"> ./outputs/{basedir}/{label}.txt 2>&1"
            
            commands.append(cmd)
            
    return commands

