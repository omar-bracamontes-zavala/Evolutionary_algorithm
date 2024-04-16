import numpy as np
from multiprocess import Pool
from time import time
from evolutionary_algorithm import evolutionary_algorithm
import evolutionary_functions as ef
from evolutionary_functions import fitness


def calculate_performance_metrics(evolutionary_algorithm, evolutionary_functions_set, sample_size = 20):
    '''
    Calculates performance metrics over several runs.
    This function generates child processes.
    input:
        evolutionary_algorithm (func): Main function for the execution of the evolutive algorithm.
        evolutionary_functions_set (list): Ordered arguments for the evolutionary algorithm to be evaluated.
        sample_size (int): Number of samples to run.
    returns
        Dictionary of metrics from the run.
    '''
    def evolution_evaluation_wrapper(*args):
        starting_time = time()
        best_fitness_history, best_individuals_history = evolutionary_algorithm(*evolutionary_functions_set)
        return {'best_fitness':best_fitness_history[-1], 'runtime':(time()-starting_time)}

    with Pool() as pool:
        best_fitnesses = pool.map(evolution_evaluation_wrapper , [None]*sample_size)
    mean_best_fitness = np.mean([trial['best_fitness'] for trial in best_fitnesses])
    return {'MBF': mean_best_fitness}


if __name__ == '__main__':
    initialize_population = ef.initialize_population
    evaluate_population = ef.evaluate_population
    select_parents = ef.tournament_selection
    crossover = ef.uniform_crossover
    mutate = ef.add_perturbation_mutation
    replacement = ef.complete_replacement
    
    # Parameters (example values)
    population_size = 100
    number_of_particles = 11
    max_generations = 5000
    mutation_rate = 0.01
    mutation_strength = 1. # this could has an adaptative control 
    
    ###########
    # For evaluation testing
    # this fker has to have same order of args as in the definition for now
    evolutionary_args = [
        fitness,
        initialize_population,
        evaluate_population,
        select_parents,
        crossover,
        mutate,
        replacement,
        population_size,
        number_of_particles,
        max_generations,
        mutation_rate,
        mutation_strength
        ]
    
    print(calculate_performance_metrics(evolutionary_algorithm,evolutionary_args))