import numpy as np

number_of_particles = 11
# Parameters (example values)
def generate_parameters_set(default, start_rate=0.5, end_rate=5, step_rate=0.5):
    return np.arange( int(default*start_rate), int(default*end_rate), int(default*step_rate)) 

population_size_default = 100
population_size_set = generate_parameters_set(100)

max_generation_default = 500
max_generations_set = generate_parameters_set(500)

mutation_rate_default = 0.01
mutation_rate_set = generate_parameters_set(0.01)

mutation_strength_set =  generate_parameters_set(1.)# this could has an adaptative control   