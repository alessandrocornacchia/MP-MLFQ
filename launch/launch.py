# generate commands and run them distributing over available machine cores

import subprocess
import multiprocessing
import itertools 
import numpy as np
from launch.experiments import *
from argparse import ArgumentParser
from includes.functions import Cdf, analytical_cdfs

available_tasks = [it for it in globals() if 'experiment' in it]
helper_tasks = ['_'.join(t.split('_')[1:]) for t in available_tasks]
parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True, 
                    help='Specify the experiment scenario. Available scenearios are: {}'.format(helper_tasks))
parser.add_argument('--dry-run', action='store_true', default=False,)

def run_command(command):
    print("Running command: {}".format(command))
    subprocess.run(command, shell=True)
    print("Task completed: {}".format(command))


def main(experiments):
  
    # this will return a list of commands
    commands = experiments()

    if args.dry_run:
        print('Dry run, no commands will be executed')
        print('\n'.join(commands))
        return
    else:# run commands
        with multiprocessing.Pool() as pool:
            pool.map(run_command, commands) # run commands in parallel



if __name__ == '__main__':
    args = parser.parse_args()
    task_f = globals()[f'experiment_{args.task}']
    main(task_f)