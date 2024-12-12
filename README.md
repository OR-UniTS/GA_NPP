This repository contains the implementation of the article "A fast GA based approach for the Network Pricing Problem".
Contents of the repository:

- A Python API to handle the problems generation and their solutions with the different algorithms
- The Implementations of the milp models for the solution of the Path and Arc versions of the NPP with Gurobi
- C++ implementation of the GA algorithms and of the heuristics (called via the Python interface) 

# Install

- Install all requirements listed in requirements.txt
- (If not installed) Install openmp library
- run ./install.sh to install cpp all required CPP libraries

# Usage
Turorial test files can be found in the main folder:
- test_path_algs.py
- test_arc_algs.py

# Experiment
All experiments run for the papers results can be replicated running 
- path_experiments.py
- arc_experiments.py
