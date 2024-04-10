'''
    This file is for storing all the available strategies for each evolutionary step
    to be used in the evolutionary algorithm
'''

import numpy as np
from numba import jit, njit, prange

#
# Fitness (Lennard-Jones potential)
#
@jit(nopython=True)
def _particle_lennard_jones_optimized(distances, sigma=1.0, epsilon=1.0, penalization_strength=1e3):
    '''
    Calculate the Lennard-Jones potential for a matrix of distances.
    This function is vectorized for efficiency and uses Numba for JIT compilation.

    Parameters:
        distances (numpy.ndarray): Matrix of distances between particles.
        sigma (float): The zero-potential distance.
        epsilon (float): The depth of the potential well.

    Returns:
        numpy.ndarray: Matrix of Lennard-Jones potentials.
    '''
    # Precompute to avoid division by zero
    sigma_over_distance = np.where(distances != 0, sigma / distances, penalization_strength) # penaliza las particulas en la misma posicion pero no hace que se vaya a inf para ahorrar memoria (?)
    sigma_over_distance = np.where(distances > 2*sigma, sigma / distances, 0) # penaliza ligeramente las particulas que estan 'muy' lejos
    sigma_over_distance_6 = sigma_over_distance ** 6
    sigma_over_distance_12 = sigma_over_distance_6 ** 2
    potentials = 4 * epsilon * (sigma_over_distance_12 - sigma_over_distance_6)

    # Apply penalization for zero distances
    # Using np.where is compatible with Numba and avoids manual loop to handle NaN or inf values
    potentials = np.where(np.isnan(potentials) | np.isinf(potentials) , penalization_strength, potentials)

    return potentials

@jit(nopython=True)
def fitness(molecule, dimension, sigma=1.0, epsilon=1.0):
    """
    Calculate the total Lennard-Jones potential energy of a molecule using optimized operations and JIT compilation with Numba.

    Parameters:
        molecule (numpy.ndarray): Flattened array of particle coordinates.
        dimension (int): The dimensionality of the space (e.g., 3 for three-dimensional space).
        sigma (float): Distance at which the potential is zero.
        epsilon (float): Depth of the potential well.

    Returns:
        float: Total Lennard-Jones potential energy of the molecule.
    """
    num_particles = len(molecule) // dimension
    reshaped_molecule = molecule.reshape(num_particles, dimension)

    # Efficiently compute the matrix of distances between all pairs of particles
    diff = reshaped_molecule[:, np.newaxis, :] - reshaped_molecule[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    # Calculate the pairwise Lennard-Jones potential
    potentials = _particle_lennard_jones_optimized(distances, sigma, epsilon)

    # Sum over the upper triangle, excluding the diagonal, to avoid double-counting pairs
    total_energy = np.sum(np.triu(potentials, k=1))

    return total_energy


#
# These will remain static for the design optimization
#
@njit
def _generate_individual(number_of_particles, dimension, simulation_box_initial_length):
    """
    Generate a random individual representing a molecule's structure.
    """
    return np.random.uniform(
        low=0,
        high=simulation_box_initial_length,
        size=(number_of_particles * dimension)
    )

@njit
def initialize_population(population_size, number_of_particles, dimension, simulation_box_initial_length):
    """
    Initialize a population of random individuals.
    """
    population = np.empty((population_size, number_of_particles * dimension))
    for i in range(population_size):
        population[i] = _generate_individual(number_of_particles, dimension, simulation_box_initial_length)
    return population

@jit(nopython=True, parallel=True)
def evaluate_population(population, dimension, fitness_function_jitted):
    """
    Evaluate the fitness of each individual in the population using JIT compilation and parallel execution.

    Parameters:
        population (list of numpy.ndarray): The population to evaluate.
        dimension (int): The dimensionality of the space (e.g., 3 for three-dimensional space).
        fitness_function_jitted (function): A JIT-compiled fitness function that calculates the fitness of an individual.

    Returns:
        numpy.ndarray: An array containing the fitness of each individual in the population.
    """
    # Preallocate an array for fitness values
    fitness_values = np.empty(len(population), dtype=np.float64)

    # Parallel loop through the population
    for i in prange(len(population)):
        fitness_values[i] = fitness_function_jitted(population[i], dimension)

    return fitness_values

#
# Parent Selection (<selection_strategy>_selection)
#
def tournament_selection(population, fitnesses, tournament_size=3):
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

#
# Crossover ( <crossover_strategy>_crossover )
#
@njit
def uniform_crossover(parent1, parent2):
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

#
# Mutation ( <mutation_strategy>_mutation )
#
@njit
def add_perturbation_mutation(individual, mutation_rate, mutation_strength=0.1):
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
            individual[i] +=  np.random.uniform(-mutation_strength, mutation_strength)
    return individual

#
# Generate new population ( <generate_new_population_strategy>_generate_new_population )
#
def replace_all_generate_new_population(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength):
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
        offspring1 = mutate(offspring1, mutation_rate, mutation_strength)
        offspring2 = mutate(offspring2, mutation_rate, mutation_strength)
        # Here we ensure the new population does not exceed the intended population size
        # This is necessary since we're adding two offspring at a time
        if len(new_population) < population_size:
            new_population.append(offspring1)
        if len(new_population) < population_size:
            new_population.append(offspring2)
    return new_population


#
# For an easier implementation on design
#
available_functions = {
    'selection': [
        tournament_selection,
    ],
    'crossover': [
        uniform_crossover,
    ],
    'mutation': [
        add_perturbation_mutation,
    ],
    'generate_new_population': [
        replace_all_generate_new_population,
    ],
}

if __name__=='__main__':
    from time import time
    a = _generate_individual(100, 3, [-10,10])
    
    start = time()
    b = fitness(a, 3)
    print( time()-start )
    print(b)
