import copy
import numpy as np
import pandas as pd


def precision_round(n, decimals=8):
    """Avoid bugs with the Python's float precision"""
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier



def load_sanity_check(dirname, target_load, tolerance=.02):
    """ Useful before plotting as a sanity check """
    
    dirname = dirname.replace('sim_results', 'outputs')

    try:
        with open(dirname + '.txt', 'r') as f:
            
            lines = f.readlines()
            
            # this line should contain average load
            if not lines[-1].startswith('average'):
                raise Exception('Simulation failed')
            
            avg_load = float(lines[-1].split(': ')[-1])
    except FileNotFoundError:
        print(f'* Warning: could not find {dirname}.txt')
        return
        
    if abs(avg_load - target_load) > tolerance:
        print(f'* Warning: avg load: {avg_load}, target load: {target_load}')




def from_pars_to_df(A, parameter, newcolumn):
    """ Adds new columns to dataframe from dictionary of parameters. 
        Useful for plotting having everything in one place."""

    pars = copy.copy(A.exp_pars)
    
    # A.exp_pars is a dictionary EXPERIMENT_IDS -> exp_pars, where exp_pars is the per-experiment
    # dictionary. Here we do something like v['K'] = v['num_servers']. Ugly but flexible
    for k,v in pars.items():
        key = parameter.split('["')[1].split('"]')[0]
        if key not in v:
            return None
        exec(f'v[newcolumn] = v{parameter}')
    
    # create from the dictionary experiment a dataframe using only the column we are 
    # interested in, and then merge with the df containing the numerical results
    df = pd.DataFrame.from_dict(pars, orient='index', columns=[newcolumn])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'experiment_id'}, inplace=True)
    if newcolumn in A.df.columns:
        A.df.drop(columns=[newcolumn], inplace=True)
    A.df = A.df.merge(df, on='experiment_id', suffixes=(None,None))
    return df



def thresholds_to_strings(analyzer, ids=None):
    """ join vector of thresholds into a string """
    res = {}
    if ids is None:
        ids = analyzer.exp_pars.keys()
    for id in ids:
        ths = analyzer.exp_pars[id]["demotion_thresholds"]
        res[id] = ', '.join(f'{int(x)}' for x in ths)
    return res
