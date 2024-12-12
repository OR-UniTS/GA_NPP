from Path.Solver.genetic_heuristic_solver import GeneticHeuristic
from Path.Instance.instance import Instance
from Path.Solver.genetic_solver import Genetic
from Path.Solver.solver import GlobalSolver


# Seed setting
seed = 0

# Network definition
n_commodities = 2
n_paths = 2
recombination_size = int(n_paths / 2)

# instance generation
npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
npp.show()

# Gurobi exact solver
solver = GlobalSolver(npp, verbose=False)
solver.solve()


# GA hyperparameter definition
POPULATION = 256
OFF_SIZE = int(POPULATION / 2)
ITERATIONS = 100
MUTATION_RATE = 0.02

# GA
g = Genetic(npp, pop_size=POPULATION, offs_size=OFF_SIZE, mutation_rate=MUTATION_RATE, recombination_size=recombination_size, verbose=True, seed=seed)
g.run(ITERATIONS)

# GA+PL
HEURISTIC_EVERY = 10
genetic_h = GeneticHeuristic(npp, pop_size=POPULATION, offs_size=OFF_SIZE, mutation_rate=MUTATION_RATE, recombination_size=recombination_size, heuristic_every=HEURISTIC_EVERY,
                             verbose=True, seed=seed)
genetic_h.run(ITERATIONS)


print(g.time, genetic_h.time, solver.time)
print(g.best_val, genetic_h.best_val, solver.obj)

