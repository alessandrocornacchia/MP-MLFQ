#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from includes.data_analyzer import DataManager
import os
import datetime as dt
import itertools
import json
import glob
from includes.utils import load_sanity_check
from includes.utils import thresholds_to_strings, from_pars_to_df


def create_legend(x):
    if x == 'task_mlfq' or x == 'task_esn':
        return 'MLFQ'
    return 'MP-MLFQ'


#%% IMPORT SIMULATION RESULTS AND FOR EACH PARAMETER CONFIGURATION READ DIRECTORY AND 
# PICK MOST RECENT SIMULATION

S = [2, 4]
Q = [2,5]
lambdas = [0.4, 0.5, 0.6, 0.7, 0.8]
cdfs = ['ws_bp']
disc = ['PS']
task = ['task_sd_mlfq_hybrid', 'task_mlfq']

resdir = './sim_results/N2-unequal-split-many-servers'
#resdir = './sim_results/K2-N2-unequal-split' # compare for several betas

paper_dir = './'

# --- if already aggregated, skip loading and aggregation
category = 'mice' #empty if all
aggregated_out = os.path.join(resdir, f'fct{category}.csv')

force = False
plot = False # plot when aggregate


if not os.path.exists(aggregated_out) or force:

    A = DataManager()

    for s, q, l, cdf, d, t in itertools.product(S, Q, lambdas, cdfs, disc, task):
        # pick last available simulation
        dirnames = glob.glob(f"{resdir}/{t}_K{s}_N{q}_l{l}_{cdf}_{d}*")
        for dirname in dirnames:
            if os.path.exists(dirname):
                print(f"Loading {dirname}")
                load_sanity_check(dirname, l)
                files = os.listdir(dirname)
                dates = [dt.datetime.strptime(f.split('.')[0], '%Y-%m-%d--%H-%M-%S') for f in files]
                last = sorted(dates)[-1] # use most recent simulation
                
                # load pickle, will contain dataframe and experiment parameters
                experiment = A.load_file(f'{dirname}/{last.strftime("%Y-%m-%d--%H-%M-%S")}.pickle')
                
                # assign to each experiment an id and update exp_pars dictonary
                # map: id -> experiment parameters 
                A.merge_experiments(experiment)


    #%% add parameters to A.df from A.exp_pars
    # TODO this should go in the merge experiment function...
    # new column name to populate -> cutoff threshold numeric value
    th_col = '$\\alpha$'
    if resdir == './sim_results/pias-threshold-2-queues':
        col = from_pars_to_df(A, '["thresholds"][0]', th_col)
    else:
        col = from_pars_to_df(A, '["demotion_thresholds"][1 if v["task"]=="task_esn" else -2]', th_col)
    task_col = from_pars_to_df(A, '["task"]', 'task')
    kcol = from_pars_to_df(A, '["num_servers"]', 'K')
    kcol = from_pars_to_df(A, '["num_queues"]', 'N')
    alphas_percentile = from_pars_to_df(A, '["cut_threshold"]', 'alpha_perc')

    #%% COMPUTE FCT VS LOAD AND WRITE TO CSV

    A.df['scheduler'] = A.df['task'].apply(create_legend)
    A.df['legend'] = A.df['scheduler'] + ' @ ' + A.df['N'].astype(str) + 'PQ, ' + A.df['K'].astype(str) + ' Spine'
    #A.df['legend'] = A.df[['N', 'legend']].apply(lambda x: 'FIFO' if x['N']==1 else x['legend'])


    print(f'Saving to {aggregated_out}...')
    df = A.df[A.df['cat'] == category]
    df.groupby(['load', 'legend', 'N', 'K', 'scheduler'])['fct_n'].agg(
        ['mean', 'std', ('95th', lambda x: np.percentile(x, 95))]
    ).reset_index().to_csv(aggregated_out)

    # Plot if flag enabled
    if plot:
        A.plot_fct_vs_load(log_y=False, 
                        normalized=True,
                        breakdown=False,
                        ylim=[1,2],
                        legend_col='legend')
        
        A.plot_fct_vs_flow_size(_lambda=0.8, 
                            normalized=True)
    




print(f'Loading from {aggregated_out}...')
# can load from file what we just saved
df = pd.read_csv(aggregated_out)

df['cv'] = df['std'] / df['mean']

sns.reset_defaults()
sns.set_style("whitegrid", {'axes.grid' : True})     
sns.set_context("paper", 
                font_scale=1.2, 
                rc={
                "lines.linewidth": 3.,
                "lines.markersize": 6
                })

ys = ['mean', 'cv', '95th', 'std']
ylims = [[1,None], None, [1,None], None]
ylabels = ['Normalized FCT', 'Coefficient of Variation', '95th Percentile FCT', 'Standard Deviation']
fignames = ['afct', 'cv', '95th', 'std']
if category != '':
    fignames = [s + f'-{category}' for s in fignames]


for i in range(len(ys)):

    figsize = (3,3) if category == '' else (2,2.5)
    plt.figure(figsize=figsize, dpi=300)

    ax = sns.lineplot(
        x='load', 
        y=ys[i], 
        data=df,
        markers = ['p', 'd', 's', 'o'],
        hue= 'legend',
        palette=['dimgrey', 'dodgerblue', 'firebrick', 'darkorange'], #['black', 'blue', 'red', 'goldenrod'],
        style= 'legend',
        dashes=False,
        #markers=True,
    )
    
    # somehow the markers are white colored in the edge
    for line in ax.lines:
        # Get the marker face color
        color = line.get_markerfacecolor()
        # Set the marker edge color
        line.set_markeredgewidth(0)
        # get legend name for line


    plt.ylabel(ylabels[i])
    plt.xlabel('Normalized Load')
    plt.ylim(ylims[i])
    plt.xlim([0.4, 0.8])
    
    # set xticks
    #plt.xticks(np.arange(0.4, 0.9, 0.1))
    ax.get_legend().remove()
    
    # set dashed grids
    ax.grid(axis='both', linestyle='--')
    plt.tight_layout()
    #plt.show()
    plt.savefig(rf'{paper_dir}\{fignames[i]}-vs-load.pdf',
                bbox_inches='tight', pad_inches=0, transparent=True)

    # create legend figure
    legendFig = plt.figure(dpi=300, figsize=(10,2))
    labels = [line.get_label() for line in ax.lines if not line.get_label().startswith('_')]
    legendFig.legend(
        ax.lines, 
        labels, 
        loc='center',
        ncol=2, 
        fontsize=18, 
        markerscale=2)
    plt.tight_layout()
    legendFig.savefig(rf'{paper_dir}\legend.pdf',
                        bbox_inches='tight', pad_inches=0, transparent=True)
    