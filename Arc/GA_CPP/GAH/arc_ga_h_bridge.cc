#include "arc_genetic_h.h"


extern "C" {

    ArcGeneticHeuristic* ArcGeneticHeuristic_(double* upper_bounds_, double* lower_bounds_, double* adj_, int adj_size_, int* tolls_idxs_, 
                            int* n_usr, int* origins_, int* destinations_, short n_commodities_, short n_tolls_, 
                            short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, short d_every,
                            short num_threads_, bool verbose_, short seed) {
        return new ArcGeneticHeuristic(upper_bounds_, lower_bounds_, adj_, adj_size_, tolls_idxs_, n_usr, origins_, destinations_, n_commodities_, n_tolls_, 
                                pop_size_, off_size_, mutation_rate_, recombination_size_, d_every,
                                num_threads_, verbose_, seed);
    }

    void run_arc_genetic_h_(ArcGeneticHeuristic* g, double* population, int iterations) {g -> run(population,iterations);}
    double get_gen_best_val_h_(ArcGeneticHeuristic* g) {return g -> get_best_val();}
    double* get_population_h_(ArcGeneticHeuristic* g) {return g -> get_population();}
    double* get_vals_h_ (ArcGeneticHeuristic* g) {return g-> get_vals();}
    void destroy_h_(ArcGeneticHeuristic* g) {delete g;}

}