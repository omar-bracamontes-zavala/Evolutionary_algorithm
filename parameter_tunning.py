import numpy as np
from helpers.get_top_designs import get_merged_tops

#
# Designs
#
top_designs = get_merged_tops('results/cleaned_results.json', 3)

#
# Hyper-parameters
def generate_parameters_set(default, start_rate=0.5, end_rate=5, step_rate=0.5):
    return np.arange( int(default*start_rate), int(default*end_rate), int(default*step_rate)) 

number_of_particles = 11 # este se barrera despues de tunnear
population_size_set = generate_parameters_set(100)
max_generations_set = generate_parameters_set(500)
mutation_rate_set = generate_parameters_set(0.01)
mutation_strength_set =  generate_parameters_set(1.)# this could has an adaptative control   

