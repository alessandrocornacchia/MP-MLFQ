import bisect
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from includes.functions import *
import pickle
import os
import seaborn as sns
import json

#sns.set()


class DataManager(object):
    def __init__(self):
        super().__init__()
        self.df = None
        self.experiment_id = 0
        self.exp_pars = {}

    def add_data(self, data):
        self.df = pd.DataFrame.from_dict(data)

    def add_parameter(self, key, value):
        self.exp_pars[key] = value

    def merge_experiments(self, data):
        """ Append to dataframe and add 2 additional columns:
            -experiment summary string
            -experiment identifier
        """
        print(data['exp_pars'])
        print(type(data['exp_pars']))
        exp_pars = data['exp_pars']
        df = data['df']
        
        dataset = "{} -- K={}, N={}, CDF={}, Disc={}, id={}".format(
            exp_pars['task'], 
            exp_pars['num_servers'], 
            exp_pars['num_queues'], 
            exp_pars['cdf_name'], 
            exp_pars['disc'],
            self.experiment_id)

        df = df.assign(dataset=dataset)
        #df = df.assign(thresholds = exp_pars['demotion_thresholds'])
        df = self.process_data(df, exp_pars=exp_pars)
        
        id = copy.copy(self.experiment_id)
        df = df.assign(experiment_id=id)
        self.exp_pars[id] = exp_pars

        if self.df is not None:        
            self.df = self.df.append(df)
        else:
            self.df = df
        
        self.experiment_id += 1

    def load_file(self, file_path):
        """ Loads data from a pickle file, containing a dictionary with two keys:
            'exp_pars': experiment parameters,
            'df': a dataframe with experiment results 
        """
        if os.path.isfile(file_path):
            data = pickle.load(open(file_path, "rb"))
            if type(data['exp_pars']) is str:
                data['exp_pars'] = json.loads(data['exp_pars'])
            return data
        else:
            print("File {} does not exist".format(file_path))
            raise Exception("Trying to load a non existing file")

    def get_dataframe(self):
        return copy.deepcopy(self.df)

    def save_data(self, file_path):
        """ Dumps data to a pikle file """
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        if os.path.isfile(file_path):
            import string
            import random
            self.save_data(
                file_path+random.choice(string.ascii_letters), self.exp_pars)
            return
        
        pickle.dump({'df': self.df, 'exp_pars': self.exp_pars},
                    open(file_path+".pickle", "wb"))

    def process_data(self, df, exp_pars):
        """ Post processing of the dataframe """
        categories = [(100, 'mice'), (10000, 'medium'), (np.inf, 'elephant')]
        K = exp_pars['num_servers']
        avg = BoundedPareto(analytical_cdfs[exp_pars['cdf_name']]).avg
        mu = avg / K
        
        df['fct'] = df.completion_time - df.generation_time
        df['fct_n'] = (df.completion_time - df.generation_time) / (df.len/mu)
        # assign mice, medium, elephant
        df['cat'] = df['len'].apply(lambda x: categories[bisect.bisect_left(categories, x, key=lambda y: y[0])][1])
        return df

    def filter(self, ids=None):
        df = self.df
        if ids is not None:
            df = self.df[self.df['experiment_id'].isin(ids)]
        return df


    def get_cdf_objects(self, n=1):
        """ Returns first cdf object found """
        if n != 1:
            raise Exception("Not implemented")
        
        cdfs = [s.split('CDF=')[1].split(',')[0] for s in self.df['dataset'].unique()]
        cdf_name = pd.unique(cdfs)[0]
        return globals()[analytical_cdfs[cdf_name]['class']](analytical_cdfs[cdf_name])
    

    def plot_fct_vs_flow_size(self, _lambda, normalized=False, postfix_title='', out=None, ids=None, legend_col=None):
        
        sns.set_style("ticks")
        
        df = self.filter(ids)
        df['dataset_noid'] = df.loc[:,'dataset'].apply(
            lambda x: ', '.join(x.split(', ')[:-1]))
        
        # if legend column in not provided, we use the dataset column, after
        # removing the experiment ID from the string as a default
        # which accounts for tuple (task, N, K, disc, cdf)
        if legend_col is None:
            legend_col = 'dataset_noid'
        
        # for each curve, we group the flow into log sized bins
        dff = dict(tuple(df.groupby(legend_col)))
        dff_tmp = []
        for k in dff:
            df = dff[k]
            df = df[df['load'] == _lambda]
            flow_size_min = df['len'].min()
            flow_size_max = df['len'].max()
            bins = np.logspace(np.log(flow_size_min), np.log(flow_size_max), 200, endpoint=True)
            df = df.groupby(pd.cut(df["len"], bins=bins)).mean()
            #df = df.assign(dataset=k)
            df[legend_col] = k
            dff_tmp.append(df)
        
        # join in a single dataframe all curves, index will be the flow size bin (x-axis)
        dff = pd.concat(dff_tmp)
        #dff = dff.query('id > 1000')  # discard transient
        # at this point we use seaborn to plot all of them
        y = 'fct'
        if normalized:
            y = 'fct_n'
        g = sns.lineplot(x='len', y=y, data=dff.dropna(subset=[y]), hue=legend_col,
                         style=legend_col, markers=True)
        g.set(xscale='log')
        #g.set(yscale='log')
        g.set(xlabel='Flow length')
        g.set(ylabel='FCT')
        # set font size on legend in axes g
        plt.setp(g.get_legend().get_texts(), fontsize='small')
        # set empty title in legend g
        g.get_legend().set_title(None)
    

        # TODO should use function defined to get cdf object from dataset name
        cdfs = [s.split('CDF=')[1].split(',')[0] for s in self.df['dataset'].unique()]
        cdf_name = pd.unique(cdfs)[0]
        cdf = Cdf(analytical_cdfs[cdf_name])
        
        if normalized:
            g.set(ylabel='nFCT')

        plt.title(f'$\lambda$={_lambda}{postfix_title}')
        plt.tight_layout()
        file = out if out is not None else 'detailed'
        plt.savefig(f"{file}.png")
        plt.savefig(f"{file}.pdf")
        plt.savefig(f"{file}.svg")

    def print_stats(self):
        print("Total generated items: {}".format(len(self.df)))
        print("Mean of nFCT: {}".format(self.df['fct_n'].mean()))

    def plot_fct_vs_load(
            self, 
            log_y=True, 
            normalized=False, 
            ids=None,
            breakdown=False,
            legend_col=None,
            cat=None,
            dump_to=None,
            ylim=None):

        sns.reset_defaults()
        sns.set_style("whitegrid", {'axes.grid' : True}) 

        sns.set_context("talk", rc={
            "lines.linewidth": 2,
            "lines.markersize": 10})
        #sns.set_context("paper", font_scale=1.5, rc={
        #    "lines.linewidth": 2.,
        #    "lines.markersize": 8})

        df = self.filter(ids).copy()
        df['dataset_noid'] = df.loc[:,'dataset'].apply(
            lambda x: ', '.join(x.split(', ')[:-1]))
        
        if legend_col is None:
            legend_col = 'dataset_noid'

        y = 'fct_n' if normalized else 'fct'

        if not breakdown:

            if cat is not None:
                df = df[df['cat'] == cat]
            
            # store to csv the data we plot
            if dump_to:
                print(f'Saving to {dump_to}...')
                # compute variance, percentile and mean
                df.groupby(['load', legend_col, 'N', 'K', 'scheduler'])[y].agg(
                    ['mean', 'std', ('95th', lambda x: np.percentile(x, 95))]
                ).reset_index().to_csv(dump_to)

            plt.figure(figsize=(6,4))

            g = sns.lineplot(
                x='load', 
                y=y, 
                data=df,
                markers =True,
                style=legend_col, 
                hue=legend_col,
                palette=['black', 'blue', 'green'])
            
            
            g.set(ylabel='Normalized FCT' if normalized else 'FCT')
            g.set(xlabel='load')
            if ylim:
                g.set(ylim=ylim)
            if log_y:
                g.set(yscale="log")
            #plt.legend(ncols=1, loc="upper center", bbox_to_anchor=(0.35, 1.3))
            g.get_legend().set_title(None)
            plt.tight_layout()

            plt.savefig("comparison.png")
            plt.savefig("comparison.pdf")
            plt.savefig("comparison.svg")

        elif breakdown:   # subpltos for mice, medium and elephants
            
            # when using relplot plt.figure() parameters are ignored, need to adjust height and aspect
            g = sns.relplot(data=df, 
                        x='load', 
                        y=y, 
                        style=legend_col, 
                        hue=legend_col,
                        col='cat', 
                        kind='line',
                        markers=True,
                        #legend=False,
                        height=3.5, 
                        aspect=1)

            g.legend.set_title(None)
            #sns.move_legend(g, "upper center", bbox_to_anchor=(0.35, 1.25), 
            #                ncol=2
            #                )
            plt.show()
