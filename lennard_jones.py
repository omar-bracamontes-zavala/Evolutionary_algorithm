
import numpy as np
import matplotlib.pyplot as plt

#
# Evolutivo cosas
#
def generate_individual(number_of_particles, dimension):
    '''
    Generate a random individual representing a molecule's structure.   
    '''
    return np.random.uniform(low=-1, high=1, size=( number_of_particles * dimension ))

def initialize_population(population_size, number_of_particles, dimension):
    """
    Initialize a population of random individuals.
    """
    return [generate_individual(number_of_particles, dimension) for _ in range(population_size)]

def evaluate_population(population, dimension):
    """
    Evaluate the fitness of each individual in the population.
    """
    return [fitness(individual, dimension) for individual in population]

# Selection
def select_parents(population, fitnesses, tournament_size=3):
    """
    Select an individual from the population using tournament selection.

    Parameters:
        population (list of numpy.ndarray): The current population.
        fitnesses (list of float): The fitness of each individual in the population.
        tournament_size (int): The number of individuals competing in each tournament.

    Returns:
        numpy.ndarray: The selected individual.
    """
    # Randomly select tournament_size individuals from the population
    participants_idx = np.random.choice(len(population), tournament_size, replace=False)
    participants_fitnesses = [fitnesses[idx] for idx in participants_idx]
    # Select the individual with the best fitness
    winner_idx = participants_idx[np.argmin(participants_fitnesses)]
    return population[winner_idx]

# Crossover
def crossover(parent1, parent2):
    """
    Perform uniform crossover between two parents to produce two offspring.

    Parameters:
        parent1, parent2 (numpy.ndarray): The parent individuals.

    Returns:
        numpy.ndarray, numpy.ndarray: Two offspring individuals.
    """
    # Initialize offspring as copies of the parents
    offspring1, offspring2 = np.copy(parent1), np.copy(parent2)
    
    # Iterate over each gene position
    for i in range(len(parent1)):
        # With 50% chance, swap the genes at position i
        if np.random.rand() < 0.5:
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
    
    return offspring1, offspring2

# Mutation
def mutate(individual, mutation_rate, mutation_strength=0.1):
    """
    Mutate an individual's genes with a given mutation rate.

    Parameters:
        individual (numpy.ndarray): The individual to mutate.
        mutation_rate (float): The probability of each gene to mutate.
        dimension (int): The dimensionality of the space.
        mutation_strength (float): The magnitude of mutation.

    Returns:
        numpy.ndarray: The mutated individual.
    """
    # Iterate over each gene in the individual
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Apply a small, random change
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            individual[i] += mutation
            # For example, if your genes should be in the range [-1, 1], you could use:
            individual[i] = np.clip(individual[i], -1, 1)
    return individual

#
# Fitness cosas
#

def _particle_lennard_jones(vector_1, vector_2, sigma=1.0, epsilon=1.0):
    '''
    vector_1: position vector of the particule 1
    sigma: the zero-potential distance.
    epsilon: the depth of the potential well
    '''
    distance = np.linalg.norm(vector_1 - vector_2)
    if distance==0:
        # Penailzar
        return 100
    return 4 * epsilon * ( (sigma/distance)**12 - (sigma/distance)**6 )
        

def fitness(molecule, dimension, sigma=1.0, epsilon=1.0):
    """
    Calculate the total Lennard-Jones potential energy of a molecule.

    Parameters:
        molecule (numpy.ndarray): Flattened array of particle coordinates.
        dimension (int): The dimensionality of the space (e.g., 3 for three-dimensional space).
        sigma (float): Distance at which the potential is zero (collision diameter).
        epsilon (float): Depth of the potential well.

    Returns:
        float: Total Lennard-Jones potential energy of the molecule.
    """
    total_energy = 0.0
    num_particles = len(molecule) // dimension

    for i in range(num_particles):
        for j in range(i + 1, num_particles):  # Avoid counting pairs twice
            # Extract particle coordinates from the flattened array
            particle_i = molecule[i * dimension:(i + 1) * dimension]
            particle_j = molecule[j * dimension:(j + 1) * dimension]

            # Calculate and sum the pairwise Lennard-Jones potential
            interaction_energy = _particle_lennard_jones(particle_i, particle_j, sigma, epsilon)
            total_energy += interaction_energy

    return total_energy

#
# Main
#
def evolutionary_algorithm(population_size, number_of_particles, dimension, generations, mutation_rate):
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
    return best_fitness_history, best_genomes_history

# Parameters (example values)
population_size = 100
number_of_particles = 4
dimension = 3
generations = 500
mutation_rate = 0.05

# Run the algorithm
best_fitness_history, best_genomes_history = evolutionary_algorithm(population_size, number_of_particles, dimension, generations, mutation_rate)
print('Particles: ', number_of_particles)
print("Best Individual:", best_fitness_history[-1])
print("Best Fitness:", best_genomes_history[-1])

# Plot the best fitness score across generations
plt.plot(best_fitness_history)
plt.title('Best Fitness Score over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Score')
plt.show()
