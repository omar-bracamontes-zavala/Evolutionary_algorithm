import json
import numpy as np
from helpers.get_top_designs import get_merged_tops
from evolutionary_algorithm import evolutionary_algorithm
from design_optimization import calculate_performance_metrics
from evolutionary_functions import available_functions,fitness
from itertools import product
import evolutionary_functions as ef
from time import time 

#
# Designs
#
# top_designs = get_merged_tops('results/cleaned_results.json', 3)


#
# Hyper-parameters
def generate_parameters_set(default, start_rate=0.5, end_rate=5, step_rate=1.5):
    if isinstance(default,int):
        return np.arange( int(default*start_rate), int(default*end_rate), int(default*step_rate)) 
    return np.arange( default*start_rate, default*end_rate, default*step_rate) 

# population_size_set = generate_parameters_set(100)
# max_generations_set = generate_parameters_set(500)
# mutation_rate_set = generate_parameters_set(0.01)
#mutation_strength_set =  generate_parameters_set(1.)# this could has an adaptative control   

population_size_set = [300]
max_generations_set = [1000]

# mutation_rate_set = generate_parameters_set(0.01)
#mutation_strength_set =  generate_parameters_set(1.)# this could has an adaptative control   

if __name__ == '__main__':
    initialize_population = ef.initialize_population
    evaluate_population = ef.evaluate_population
    # number_of_particles = 11 # este se barrera despues de tunnear

    number_of_particles = [x for x in range(2,30+1)]
    population_size = 300
    max_generations = 1000
    mutation_rate = 0.01
    mutation_strength = 1.

    design_selection = [1, 0, 2, 2]

    general_evolutionary_functions = [
        fitness,
        initialize_population,
        evaluate_population,
    ]
    
    design_functions = [available_functions[x][option] for x,option in zip(['selection','crossover','mutation','replacement'], design_selection)]



    combination_performances = []
    
    for number_of_particles in range(2,31):
        parameters = [
            population_size,
            number_of_particles,
            max_generations,
            mutation_rate,
            mutation_strength
        ]
        time_start = time()
        try:

            print(f'Particles: {number_of_particles}')
            evolutionary_algorithm_args = general_evolutionary_functions + design_functions + parameters
            performances = calculate_performance_metrics(evolutionary_algorithm,evolutionary_algorithm_args,sample_size=5)
            combination_performances.append({
                'number_of_particles':number_of_particles,
                'performance_metrics': performances,
                })

        except Exception as err:
            print(f'Particles: {number_of_particles}')
            combination_performances.append({
                'number_of_particles':number_of_particles,
                'performance_metrics': str(err),
                })

        print(f'\tFinished in {round(time()-time_start)}s')
        with open('results/num_particles_results.json','w') as json_file:
            json.dump(combination_performances,json_file)