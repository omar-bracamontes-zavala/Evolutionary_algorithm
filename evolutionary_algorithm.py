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
        initialize_population,
        evaluate_population,
        select_parents,
        crossover,
        mutate,
        # Algorithm Hyperparameters
        population_size,
        number_of_particles,
        dimension,
        max_generations,
        mutation_rate,
        # Hyperparameters
        simulation_box_bounds,
    ):
    """
    Main evolutionary algorithm.
    """
    # Initialize population
    population = initialize_population(population_size, number_of_particles, dimension, simulation_box_bounds)
    best_fitness_history = []
    best_individuals_history = []
    
    for _ in range(max_generations):
        # Evaluate population
        fitnesses = evaluate_population(population, dimension, fitness)
        
        # Log
        _register_best_individual_and_fitness(fitnesses, population, best_fitness_history, best_individuals_history)
        
        # Generate a new population using tournament selection, crossover, and mutation
        # 
        new_population = []
        while len(new_population) < population_size:
            # Select parents
            parent1 = select_parents(population, fitnesses)
            parent2 = select_parents(population, fitnesses)
            # Ensure parent2 is different from parent1
            while np.array_equal(parent1, parent2):
                parent2 = select_parents(population, fitnesses)
            # Crossover
            offspring1, offspring2 = crossover(parent1, parent2)
            # Mutate
            offspring1 = mutate(offspring1, mutation_rate, dimension)
            offspring2 = mutate(offspring2, mutation_rate, dimension)
            new_population.extend([offspring1, offspring2])
        
        population = new_population[:population_size]
        
        # Optionally, here you can implement elitism to directly pass the best individual(s) to the next generation
        
    # Final evaluation to find the best solution
    final_fitnesses = evaluate_population(population, dimension)
    
    # Log
    _register_best_individual_and_fitness(final_fitnesses, population, best_fitness_history, best_individuals_history)
    
    return best_fitness_history, best_individuals_history

if __name__=='__main__':
    
    initialize_population = ef.initialize_population
    evaluate_population = ef.evaluate_population
    select_parents = ef.tournament_selection
    crossover = ef.uniform_crossover
    mutate = ef.add_perturbation_mutation
    
    # Parameters (example values)
    simulation_box_limits = (-1,1)
    population_size = 100
    number_of_particles = 20
    dimension = 3
    max_generations = 500
    mutation_rate = 0.05

    # Run the algorithm
    best_fitness_history, best_individuals_history = evolutionary_algorithm(
        # Main functions
        initialize_population=initialize_population,
        evaluate_population=evaluate_population,
        select_parents=select_parents,
        crossover=crossover,
        mutate=mutate,
        # Algorithm Hyperparameters
        population_size=population_size,
        number_of_particles=number_of_particles,
        dimension=dimension,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        # Hyperparameters
        simulation_box_bounds=simulation_box_limits,
    )
    
    print('Particles: ', number_of_particles)
    print("Best Individual:", best_fitness_history[-1])
    print("Best Fitness:", best_individuals_history[-1])

    # Plot the best fitness score across generations
    # plt.plot(best_fitness_history)
    # plt.title('Best Fitness Score over Generations')
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness Score')
    # plt.show()

    from particles_animation import visualize_particle_system
    print('Finished')
    visualize_particle_system(best_individuals_history, number_of_particles)