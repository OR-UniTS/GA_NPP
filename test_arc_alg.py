from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.GA_CPP.genetic_arc import GeneticArc

# Seed setting
seed = 0

# Network definition
n_arcs = 3*4
dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]
toll_proportion = 5
n_commodities = 10

# instance generation
instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, seed=seed)
instance.show()

# Gurobi exact solver
solver = ArcSolver(instance=instance)
solver.solve(time_limit=3600, verbose=True)


# GA hyperparameter definition
ITERATIONS = 1000
POPULATION_SIZE = 64
OFFSPRING_RATE = 0.5
MUTATION_RATE = 0.1


# GA
g = GeneticArc(npp=instance, population_size=POPULATION_SIZE, offspring_rate=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, n_threads=None, seed=seed)
g.run(ITERATIONS, verbose=True)

# GA+PL
DIJKSTRA_EVERY = 100
gh = GeneticArc(npp=instance, population_size=POPULATION_SIZE, offspring_rate=OFFSPRING_RATE, mutation_rate=MUTATION_RATE, n_threads=None, seed=seed)
gh.run_heuristic(ITERATIONS, dijkstra_every=DIJKSTRA_EVERY, verbose=True)

print(g.time, gh.time, solver.time)
print(g.best_val, gh.best_val, solver.obj)
