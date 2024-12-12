import time

import numpy as np

from Path.GA_CPP.GA.genetic_cpp import GeneticCpp
from Path.Instance.instance import Instance


# from heuristic import improve_solution


class Genetic:
    def __init__(self, npp: Instance, pop_size, offs_size, mutation_rate, recombination_size,
                 verbose=True, n_threads=None, seed=None):
        self.solution = None
        self.time = None
        self.pop_size = pop_size
        self.offs_size = offs_size
        self.mutation_rate = mutation_rate
        self.recombination_size = recombination_size
        self.verbose = verbose
        self.seed = seed
        self.num_threads = n_threads

        self.n_paths = npp.n_paths
        self.npp = npp
        self.upper_bounds = npp.upper_bounds
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.time = None

        self.final_population = None
        self.final_vals = None

        self.genetic = GeneticCpp(npp.upper_bounds, npp.lower_bounds,
                                  npp.commodities_tax_free,
                                  npp.n_users, npp.transfer_costs,
                                  npp.n_commodities, npp.n_paths,
                                  self.pop_size, self.offs_size,
                                  self.mutation_rate, self.recombination_size,
                                  self.verbose, self.num_threads, self.seed)

        self.values = np.array([c.c_od - p for c in self.npp.commodities for p in c.c_p_vector])

    def run(self, iterations, init_population=None):
        self.time = time.time()
        init_population = self.init_values() if init_population is None else init_population
        self.best_val = self.genetic.run(init_population, iterations)
        self.time = time.time() - self.time
        self.final_population, self.final_vals = self.genetic.get_results()
        self.solution = self.final_population[0]

    def init_values(self):
        population = np.zeros((self.pop_size, self.n_paths))
        for i in range(self.n_paths):
            vals = self.values[self.npp.paths[i].L_p <= self.values]
            vals = vals[self.npp.paths[i].N_p >= vals]
            population[:self.pop_size, i] = (
                np.random.choice(vals, size=self.pop_size, replace=True))
        return population
        # self.vals = np.array([self.fitness_fun(sol) for sol in self.population])
