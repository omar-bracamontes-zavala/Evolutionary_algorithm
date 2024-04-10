import numpy as np
import matplotlib.pyplot as plt
import evolutionary_functions as ef
from evolutionary_functions import fitness

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
        generate_new_population,
        # Algorithm Hyperparameters
        population_size,
        number_of_particles,
        max_generations,
        mutation_rate,
        mutation_strength,
        # Hyperparameters
        dimension=3,
        simulation_box_initial_length=10,
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
        population = generate_new_population(
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
    generate_new_population = ef.replace_all_generate_new_population#replace_all_generate_new_population
    
    # Parameters (example values)
    population_size = 100
    number_of_particles = 30
    max_generations = 5000
    mutation_rate = 0.01
    mutation_strength = 1. # this could has an adaptative control 
    

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
        generate_new_population=generate_new_population,
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
