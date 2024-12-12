import os
import random

import numpy as np
import pandas as pd

from Arc.ArcInstance.delunay_instance import DelaunayInstance
from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.GA_CPP.genetic_arc import GeneticArc

'''
    SCRIPT FOR THE NUMERICAL EXPERIMENTS OF THE GA AND THE GA+PL ON THE ARC NPP
'''


n_arcs = 104
dim_grid = (10, 12)
n_locations = (dim_grid[0] - 1) * (dim_grid[1] - 2)
toll_proportion = [10, 15, 20]
n_commodities = [40, 60, 80]
graphs = [DelaunayInstance, GridInstance]

free_path_distribution = []
TIMELIMIT = 3600

ITERATIONS = 10000
POPULATION_SIZE = 128
DIJKSTRA_EVERY = 100
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.1


columns = ['run', 'graphs', 'toll_pr', 'n_com', 'ga_obj', 'ga_time', 'gha_obj', 'gah_time', 'exact_obj', 'exact_time', 'MIP_gap', 'status']
row = 0
df = pd.DataFrame(columns=columns)
for graph in graphs:
    for n_c in n_commodities:
        for t_p in toll_proportion:
            for run in range(10):
                random.seed(run)
                np.random.seed(run)
                instance = graph(n_locations, n_arcs, dim_grid, t_p, n_c, seed=run)
                print("\nProblem ", instance.name, n_c, t_p, run, len(instance.npp.edges))
                # instance.show()

                solver = ArcSolver(instance=instance, symmetric_costs=False)
                solver.solve(time_limit=TIMELIMIT, verbose=False)

                g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, n_threads=None, seed=run)
                g.run(ITERATIONS, verbose=False)

                gh = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, n_threads=None, seed=run)
                gh.run_heuristic(ITERATIONS, DIJKSTRA_EVERY, verbose=False)

                print(g.time, gh.time, solver.time, g.best_val, gh.best_val, solver.obj, (1 - g.best_val / gh.best_val) * 100)
                df.loc[row] = [run, instance.name, t_p, n_c, g.best_val, g.time, gh.best_val, gh.time, solver.obj, solver.time, solver.gap,
                               solver.status]
                row += 1

if not os.path.exists('Results'):
    os.makedirs('Results')
df.to_csv('Results/arc_results.csv', index=False)