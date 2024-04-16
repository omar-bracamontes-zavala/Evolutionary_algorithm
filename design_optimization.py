from evolutionary_functions import fitness, available_functions
from evolutionary_algorithm import evolutionary_algorithm
import evolutionary_functions as ef
from multiprocess import Pool
from itertools import product
from time import time
import numpy as np


def calculate_performance_metrics(
                                evolutionary_algorithm, 
                                evolutionary_functions_set, 
                                sample_size = 30, 
                                reference_global_optimum=-128.286,
                                relative_success_threshold = 0.95):
    '''
    Calculates performance metrics over several runs.
    This function generates child processes.
    input:
        evolutionary_algorithm (func): Main function for the execution of the evolutive algorithm.
        evolutionary_functions_set (list): Ordered arguments for the evolutionary algorithm to be evaluated.
        sample_size (int): Number of samples to run.
        reference_global_optimum (float): Fitness from the optimal best solution, for evaluating success.
        relative_success_threshold (float): Proportion of the global optima reached to be considered as a successful run.
    returns
        Dictionary of metrics from the run.
    '''
    def evolution_evaluation_wrapper(*args):
        starting_time = time()
        best_fitness_history, best_individuals_history = evolutionary_algorithm(*evolutionary_functions_set)
        return {'best_fitness':best_fitness_history[-1], 'runtime':(time()-starting_time)}

    with Pool() as pool:
        best_fitnesses = pool.map(evolution_evaluation_wrapper , [None]*sample_size)
    fitnesses_values_list = [trial['best_fitness'] for trial in best_fitnesses]
    mean_best_fitness = np.mean(fitnesses_values_list)
    successful_runs = [trial for trial in fitnesses_values_list if trial < reference_global_optimum*relative_success_threshold]
    peak_performance = min(fitnesses_values_list)
    success_rate = len(successful_runs) / sample_size

    return {'MBF': mean_best_fitness, 'SR': success_rate, 'peak_performance': peak_performance}


def get_design_functions_combinations(available_functions):
    '''
    Obtains the combination of functions to evaluate.
    The enumerate is for later retrieving the resulting optimal combination of functions.
    Returns combinations as lists of tuples [ (index_of_function,function), ...]
    '''
    indexed_selections = list(enumerate(available_functions['selection']))
    indexed_crossovers = list(enumerate(available_functions['crossover']))
    indexed_mutations = list(enumerate(available_functions['mutation']))
    indexed_replacement = list(enumerate(available_functions['replacement']))
    
    return product(indexed_selections, indexed_crossovers, indexed_mutations, indexed_replacement)
    


if __name__ == '__main__':
    ## General functions
    initialize_population = ef.initialize_population
    evaluate_population = ef.evaluate_population
    
    # Parameters (example values)
    population_size = 100
    number_of_particles = 30
    max_generations = 500
    mutation_rate = 0.01
    mutation_strength = 1. # this could has an adaptative control 

    select_parents = ef.tournament_selection
    crossover = ef.uniform_crossover
    mutate = ef.add_perturbation_mutation
    replacement = ef.complete_replacement


    general_evolutionary_functions = [
        fitness,
        initialize_population,
        evaluate_population
        ]

    parameters = [
        population_size,
        number_of_particles,
        max_generations,
        mutation_rate,
        mutation_strength
        ]

    combination_performances = []

    design_function_option_combinations = get_design_functions_combinations(available_functions)
    for function_combination in design_function_option_combinations:
        try:
            ## Functions and indexes have to be unpacked
            design_functions = [element[1] for element in function_combination]
            design_functions_indexes = [element[0] for element in function_combination]

            evolutionary_algorithm_args = general_evolutionary_functions + design_functions + parameters
            performances = calculate_performance_metrics(evolutionary_algorithm,evolutionary_algorithm_args)
            combination_performances.append({
                'combination_indexes':design_functions_indexes,
                'performance_metrics': performances,
                })

        except Exception as err:
            combination_performances.append({
                'combination_indexes':design_functions_indexes,
                'performance_metrics': str(err),
                })
    import json
    with open('results.json','w') as json_file:
        json.dump(combination_performances,json_file)