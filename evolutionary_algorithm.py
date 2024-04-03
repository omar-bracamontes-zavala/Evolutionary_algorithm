import numpy as np
import matplotlib.pyplot as plt
import evolutionary_functions as ef

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
        # Hyperparameters
        population_size,
        number_of_particles,
        dimension,
        generations,
        mutation_rate,
    ):
    """
    Main evolutionary algorithm.
    """
    # Initialize population
    population = initialize_population(population_size, number_of_particles, dimension)
    best_fitness_history = []
    best_genomes_history = []
    
    for generation in range(generations):
        # Evaluate population
        fitnesses = evaluate_population(population, dimension)
        
        min_fitness_index = np.argmin(fitnesses)
            
        best_fitness_history.append(fitnesses[min_fitness_index])
        best_genomes_history.append(population[min_fitness_index])
        
        # Generate a new population using tournament selection, crossover, and mutation
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
    
    min_fitness_index = np.argmin(final_fitnesses)
    best_fitness_history.append(fitnesses[min_fitness_index])
    best_genomes_history.append(population[min_fitness_index])
    
    return best_fitness_history, best_genomes_history

if __name__=='__main__':
    initialize_population = ef.initialize_population
    evaluate_population = ef.evaluate_population
    select_parents = ef.tournament_selection
    crossover = ef.uniform_crossover
    mutate = ef.add_perturbation_mutation
    
    # Parameters (example values)
    box_limits = (-1,1)
    population_size = 100
    number_of_particles = 20
    dimension = 3
    generations = 500
    mutation_rate = 0.05

    # Run the algorithm
    best_fitness_history, best_genomes_history = evolutionary_algorithm(population_size, number_of_particles, dimension, generations, mutation_rate)
    print('Particles: ', number_of_particles)
    print("Best Individual:", best_fitness_history[-1])
    print("Best Fitness:", best_genomes_history[-1])

    # Plot the best fitness score across generations
    # plt.plot(best_fitness_history)
    # plt.title('Best Fitness Score over Generations')
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness Score')
    # plt.show()

    from particles_animation import visualize_particle_system
    print('Finished')
    visualize_particle_system(best_genomes_history, number_of_particles)