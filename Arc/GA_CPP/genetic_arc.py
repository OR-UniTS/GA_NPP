import copy
import itertools
import multiprocessing
import random
import time

import networkx as nx
import numpy as np

from Arc.ArcInstance.arc_instance import ArcInstance
from Arc.GA_CPP.GA.arc_genetic_cpp import ArcGeneticCpp
from Arc.GA_CPP.GAH.arc_genetic_cpp_heuristic import ArcGeneticCppHeuristic


class GeneticArc:

    def __init__(self, population_size, npp: ArcInstance, offspring_rate=0.5, mutation_rate=0.02,
                 n_threads=None, seed=None):
        self.num_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
        self.seed = -1 if seed is None else seed
        self.adj_solution = None
        self.mat_solution = None
        self.solution = None
        self.time = None
        self.pop_size = population_size
        self.offs_size = int(self.pop_size * offspring_rate)
        self.total_pop_size = self.pop_size + self.offs_size

        self.n_tolls = npp.n_tolls
        self.npp = copy.deepcopy(npp)
        self.upper_bounds = np.array([p.N_p for p in npp.tolls])
        self.lower_bounds = np.zeros_like(self.upper_bounds)
        self.tolls_idxs = [p.idx for p in npp.tolls]
        self.toll_idxs_flat = np.array(self.tolls_idxs).T.flatten()
        self.origins = np.array([commodity.origin for commodity in self.npp.commodities])
        self.destinations = np.array([commodity.destination for commodity in self.npp.commodities])
        self.n_users = np.array([commodity.n_users for commodity in self.npp.commodities])

        self.population = np.zeros((self.total_pop_size, self.n_tolls))
        self.combs = list(itertools.combinations(range(self.pop_size), 2))
        self.idx_range = range(self.n_tolls)
        self.pop_idx = range(self.pop_size)
        self.fitness_fun = npp.compute_obj
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.recombination_size = self.n_tolls // 2

        self.adj = npp.get_adj().copy()
        self.prices = np.zeros_like(npp.get_adj())

        self.genetic_cpp = None

        self.vals = None

    def get_mats(self, sol):
        prices = np.zeros_like(self.adj)
        for i in range(self.n_tolls):
            prices[self.npp.tolls[i].idx] = sol[i]
        adj = self.adj + self.prices

        return adj, prices


    def init_values(self, restart=False):
        start = 1 if restart else 0
        population = np.zeros((self.pop_size - start, self.n_tolls))
        for i in range(start, self.n_tolls):
            population[:self.pop_size, i] = np.random.uniform(self.lower_bounds[i], self.upper_bounds[i], size=self.pop_size - start)
        return population


    def run(self, iterations, verbose):
        self.time = time.time()

        self.genetic_cpp = ArcGeneticCpp(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users, self.origins,
                                         self.destinations, self.npp.n_commodities,
                                         self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate, self.recombination_size,
                                         verbose, self.num_threads, self.seed)
        initial_position = self.init_values()
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = self.population[0]
        self.adj_solution, self.prices = self.get_mats(self.solution)
        self.npp.npp = nx.from_numpy_array(self.adj_solution)
        for c in self.npp.commodities:
            c.solution_path = nx.shortest_path(self.npp.npp, c.origin, c.destination, weight='weight')
            c.solution_edges = [(c.solution_path[i], c.solution_path[i + 1]) for i in range(len(c.solution_path) - 1)]
        self.time = time.time() - self.time


    def run_heuristic(self, iterations, dijkstra_every, verbose):
        self.time = time.time()

        self.genetic_cpp = ArcGeneticCppHeuristic(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users,
                                                  self.origins,
                                                  self.destinations, self.npp.n_commodities,
                                                  self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate,
                                                  self.recombination_size, dijkstra_every,
                                                  verbose, self.num_threads, self.seed)
        initial_position = self.init_values()
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = self.population[0]
        self.adj_solution, self.prices = self.get_mats(self.solution)
        self.npp.npp = nx.from_numpy_array(self.adj_solution)
        for c in self.npp.commodities:
            c.solution_path = nx.shortest_path(self.npp.npp, c.origin, c.destination, weight='weight')
            c.solution_edges = [(c.solution_path[i], c.solution_path[i + 1]) for i in range(len(c.solution_path) - 1)]
        self.time = time.time() - self.time
