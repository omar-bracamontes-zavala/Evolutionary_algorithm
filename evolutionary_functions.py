'''
    This file is for storing all the available strategies for each evolutionary step
    to be used in the evolutionary algorithm
'''

import numpy as np
from numba import jit, njit, prange
import heapq # para parent selection con probabulidades

#
# Fitness (Lennard-Jones potential)
#
@jit(nopython=True)
def _particle_lennard_jones_optimized(distances, sigma=1.0, epsilon=1.0, penalization_strength=1e10):
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
    sigma_over_distance = np.where(distances != 0, sigma / distances, penalization_strength) # penaliza las particulas en la misma posicion pero no hace que se vaya a inf para ahorrar memoria (?)
    # sigma_over_distance = np.where(distances < 2*sigma, sigma / distances, 0) # penaliza ligeramente las particulas que estan 'muy' lejos
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
def tournament_selection(population, fitnesses, tournament_size=3, num_parents=2):
    """
    Perform tournament selection to choose multiple parents from the population for a minimization problem.

    Parameters:
        population (list of numpy.ndarray): The current population.
        fitnesses (list of float): The fitness of each individual in the population.
        tournament_size (int): The number of individuals competing in each tournament.
        num_parents (int): The number of parents to select.

    Returns:
        list of numpy.ndarray: The selected parents.
    """
    selected_parents = []
    population_size = len(population)
    while len(selected_parents) < num_parents:
        # Randomly select tournament_size individuals from the population
        participants_idx = np.random.choice(population_size, tournament_size, replace=False)
        participants_fitnesses = [fitnesses[idx] for idx in participants_idx]

        # Select the individual with the best (minimum) fitness
        winner_idx = participants_idx[np.argmin(participants_fitnesses)]

        # Ensure unique parents are selected if the population size allows it
        if population[winner_idx] not in selected_parents or len(set(fitnesses)) < num_parents:
            selected_parents.append(population[winner_idx])
    
    return selected_parents

# Fitness proportional selection pagina 80
def fps_selection(population, fitnesses, num_parents=2):
    population_size = len(population)
    
    # Invert fitness values to prioritize lower fitnesses
    # Ensure all fitness values are positive and non-zero to avoid division by zero
    inverted_fitnesses = [1.0 / f for f in fitnesses]

    # Calculate the sum of inverted fitnesses
    inverted_fitness_sum = sum(inverted_fitnesses)

    # Calculate selection probabilities based on inverted fitnesses
    probabilities = [f / inverted_fitness_sum for f in inverted_fitnesses]

    # Select the parents based on these probabilities
    selected_indices = np.random.choice(population_size, size=num_parents, replace=False, p=probabilities)
    selected_parents = [population[i] for i in selected_indices]

    return selected_parents
    
# Ranking selection pagina 82
def ranking_selection(population, fitnesses, s=1.5,  num_parents=2):
    '''
    Parameters:
        population (list of numpy.ndarray): The current population. Note: individuals are also np array
        fitnesses (list of numpy.ndarray): The fitness of each individual in the population.
    '''
    population_size = len(population)
        
    # Sort the population and fitnesses by descending fitness
    sorted_indices = np.argsort(fitnesses)[::-1]  # This gets indices to sort array in descending order
    sorted_population = np.array(population)[sorted_indices]
    print(sorted_indices)

    # Calculate probabilities based on linear ranking
    probabilities = np.array([(2-s)/population_size + 2*i*(s-1)/(population_size*(population_size-1)) for i in range(population_size)])

    # Select the parents based on probabilities
    selected_indices = np.random.choice(population_size, size=num_parents, replace=False, p=probabilities)
    selected_parents = sorted_population[selected_indices]

    return selected_parents
    
# Stochastic Universal Sampling (or roulette with chochos)
def sus_selection(population, fitnesses, num_parents=2):
    '''
    Perform stochastic universal sampling selection on the given population.

    Parameters:
        population (list of numpy.ndarray): The current population, where individuals are also numpy arrays.
        fitnesses (list): The fitness of each individual in the population.

    Returns:
        tuple: The selected parent, and the sorted population and fitness arrays.
    '''
    population_size = len(population)

    # Invert fitness values for minimization (assuming all fitnesses are positive)
    inverted_fitnesses = 1.0 / np.array(fitnesses)

    # Sort the population by these inverted fitnesses (higher is better now)
    sorted_indices = np.argsort(inverted_fitnesses)[::-1]
    sorted_population = np.array(population)[sorted_indices]
    sorted_inverted_fitnesses = inverted_fitnesses[sorted_indices]

    # Calculate the cumulative sum of the inverted fitnesses
    cumulative_fitnesses = np.cumsum(sorted_inverted_fitnesses)

    # Calculate the total fitness and distance between pointers
    total_fitness = cumulative_fitnesses[-1]
    pointer_distance = total_fitness / num_parents

    # Pick a random start for the pointer between 0 and pointer_distance
    start_point = np.random.uniform(0, pointer_distance)

    # Select individuals
    selected_parents = []
    pointers = [start_point + i * pointer_distance for i in range(num_parents)]
    pointer_idx = 0
    current_member = 0
    while current_member < population_size and pointer_idx < num_parents:
        if cumulative_fitnesses[current_member] >= pointers[pointer_idx]:
            selected_parents.append(sorted_population[current_member])
            pointer_idx += 1
        else:
            current_member += 1

    return selected_parents

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
# Replacement/Survivor Selection: Generate new population ( <replacement_strategy>_replacement )
#
def complete_replacement(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength):
    new_population = []
    while len(new_population) < population_size:
        # Select parents
        parent1, parent2 = select_parents(population, fitnesses, num_parents=2)
        # parent2, population, fitnesses = select_parents(population, fitnesses)
        # Ensure parent2 is different from parent1
        # while np.array_equal(parent1, parent2):
        #     parent2, population, fitnesses = select_parents(population, fitnesses)
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

# Age-Based Replacement
# Fitness-Based Replacement
# ??? Replace worst (GENITOR)
# ??? Elitism
# ??? Round-robin tournament
# (μ + λ) Selection
# (μ, λ) Selection

#
# For an easier implementation on design
#
available_functions = {
    'selection': [
        tournament_selection,
        fps_selection,
    ],
    'crossover': [
        uniform_crossover,
    ],
    'mutation': [
        add_perturbation_mutation,
    ],
    'replacement': [
        complete_replacement,
    ],
}

if __name__=='__main__':
    # from time import time
    # a = _generate_individual(100, 3, [-10,10])
    
    # start = time()
    # b = fitness(a, 3)
    # print( time()-start )
    # print(b)
    p = [ [1,1], [2,2], [3,3], [4,4], [5,5]] # ideal es 3, 5, 1, 2, 4
    f = [  0.01,   0.02,  0.03,  0.4,   0.5] 
    # Select parents
    parent1, parent2 = tournament_selection(p, f)
    
    print('parent1',parent1)
    print('parent2',parent2)
