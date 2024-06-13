import os
from includes.functions import *
from includes.solvers.optimization import *
from includes.solvers.heuristics import *


def parse_arguments():
    available_tasks = [it for it in globals() if 'Solver' in it]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help='Specify the experiment scenario. Available scenearios are: {}'.format(available_tasks))
    parser.add_argument('--num_servers', type=int, default=2,
                        help='Number of servers to use.')
    parser.add_argument('--num_queues', type=int, default=1,
                        help='Number of queues per server.')
    parser.add_argument('--lambda', type=float, help='Arrival rate of flows')
    parser.add_argument('--step', type=float, default=100, help='Step size for brute force search')
    parser.add_argument('--cdf', type=str, default='webSearch.cdf',
                        help='The name of the cdf. For supported analytical cdf and format check functions.py')
    parser.add_argument('--postfix', type=str, default='',
                        help='Postfix to append to the simulation task')
    parser.add_argument('--cdf_cut', type=float, default=0.0,
                        help='Specify the initial cut percentile of the CDF')
    parser.add_argument('--file', '-f', type=str, help='File to write thresholds to')
    parser.add_argument('--lp_threshold', type=float, default=2e3,
                        help='For SD_Circular only, threshold after which flows stop to be served by Q0')
    args = parser.parse_args()

    #args._lambda = [float(it) for it in args._lambda.split(',')]

    if args.cdf in analytical_cdfs:
        args.cdf = globals()[analytical_cdfs[args.cdf]['class']](analytical_cdfs[args.cdf])
    elif ".cdf" in args.cdf:
        args.cdf = Ecdf('ecdf/{}'.format(args.cdf))
    else:
        raise Exception("Invalid CDF specified!")
    
    """ TODO fix this without using Ecdf, can just cut the same cdf obj
    if args.cdf_cut > 0:
        assert args.cdf_cut < 1
        x = np.linspace(args.cdf.Fi(args.cdf_cut), args.cdf.Fi(1), 1000)
        y = [(it - args.cdf_cut)/(1 - args.cdf_cut) for it in args.cdf.F(x)]
        args.cdf = Ecdf({'x':x, 'y':y})
    """

    if args.task not in globals():  # The simulation scenario (i.e., task) is specified as a string on the commandline. Retrieve it from the table of available functions
        raise Exception("Trying to run an unrecognized task!")
    args.task_f = globals()[args.task]

    return args


def write_to_file(r,args):
    if not os.path.exists(os.path.dirname(args.file)):
        os.makedirs(os.path.dirname(args.file))
    print('Writing to file: {}'.format(args.file))
    with open(args.file, 'w') as f:
        f.write('\n'.join(r))
    

def main():
    args = parse_arguments()

    solver = args.task_f(vars(args))
    #solver.plot_load()
    r = solver.solve()
    print('Solver answer:', r)
    if args.file is not None:
        write_to_file(r, args)
    return r
    
    #solver = ScipySolver(_cdf=cdf, _lambda=0.9)
    #solver = PSOSolver(_cdf=cdf, _lambda=0.9)
    # solver.solve(N=7)


if __name__ == '__main__':
    r = main()
