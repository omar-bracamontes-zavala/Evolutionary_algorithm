import numpy as np
import matplotlib.pyplot as plt
import evolutionary_functions as ef
from evolutionary_functions import fitness
from multiprocess import Pool
from time import time

#
# Evaluation functions
#

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


#
# Helpers
#
def _get_best_individual_and_fitness(fitnesses, population):
    min_fitness_index = np.argmin(fitnesses)
            
    best_fitness = fitnesses[min_fitness_index]
    best_individual = population[min_fitness_index]
    
    return best_fitness, best_individual

def _register_best_individual_and_fitness(fitnesses, population, best_fitness_history, best_individuals_history):
    best_fitness, best_individual = _get_best_individual_and_fitness(fitnesses, population)
            
    best_fitness_history.append(best_fitness)
    best_individuals_history.append(best_individual)
    

#
# Main
#
def evolutionary_algorithm(
        # Main functions
        fitness,
        initialize_population,
        evaluate_population,
        select_parents,
        crossover,
        mutate,
        replacement,
        # Algorithm Hyperparameters
        population_size,
        number_of_particles,
        max_generations,
        mutation_rate,
        mutation_strength,
        # Hyperparameters
        dimension=3,
        simulation_box_initial_length=1.,
    ):
    """
    Main evolutionary algorithm.
    """
    # Initialize population
    population = initialize_population(population_size, number_of_particles, dimension, simulation_box_initial_length)
    best_fitness_history = []
    best_individuals_history = []
    
    for _ in range(max_generations):
        # Evaluate population
        fitnesses = evaluate_population(population, dimension, fitness)
        
        # Log
        _register_best_individual_and_fitness(fitnesses, population, best_fitness_history, best_individuals_history)
        
        # Generate a new population
        population = replacement(
            select_parents,
            crossover,
            mutate,
            population,
            fitnesses,
            population_size,
            mutation_rate,
            dimension,
            mutation_strength
        )
        
        # Optionally, here you can implement elitism to directly pass the best individual(s) to the next generation
        
    # Final evaluation to find the best solution
    final_fitnesses = evaluate_population(population, dimension, fitness)
    
    # Log
    _register_best_individual_and_fitness(final_fitnesses, population, best_fitness_history, best_individuals_history)
    
    return best_fitness_history, best_individuals_history

if __name__=='__main__':
    from time import time
    start = time()
    
    # Design
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
    # # this fker has to have same order of args as in the definition for now
    # evolutionary_args = [
    #     fitness,
    #     initialize_population,
    #     evaluate_population,
    #     select_parents,
    #     crossover,
    #     mutate,
    #     replacement,
    #     population_size,
    #     number_of_particles,
    #     max_generations,
    #     mutation_rate,
    #     mutation_strength
    #     ]
    
    # print(calculate_performance_metrics(evolutionary_algorithm,evolutionary_args))
    ############

    # Run the algorithm
    best_fitness_history, best_individuals_history = evolutionary_algorithm(
        # Main functions
        fitness=fitness,
        initialize_population=initialize_population,
        evaluate_population=evaluate_population,
        select_parents=select_parents,
        crossover=crossover,
        mutate=mutate,
        # Algorithm Hyperparameters
        population_size=population_size,
        number_of_particles=number_of_particles,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        replacement=replacement,
        mutation_strength=mutation_strength,
        # Hyperparameters
    )
    print('Runtime: ',time()-start,'s' )
    print('Particles: ', number_of_particles)
    print("Energy:", best_fitness_history[-1])
    # print("Best Fitness:", best_individuals_history[-1])

    # Plot the best fitness score across generations
    plt.plot(best_fitness_history)
    plt.title('Best Fitness Score over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.show()

    from particles_animation import visualize_particle_system
    print('Finished')
    visualize_particle_system(best_individuals_history,best_fitness_history,number_of_particles)
