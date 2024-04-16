'''
    This file is for storing all the available strategies for each evolutionary step
    to be used in the evolutionary algorithm
'''

import numpy as np
from numba import jit, njit, prange

#
# Fitness (Lennard-Jones potential)
#
@njit
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

@njit
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
def evaluate_population(population, dimension):#, fitness_function_jitted):
    """
    Evaluate the fitness of each individual in the population using JIT compilation and parallel execution.

    Parameters:
        population (list of numpy.ndarray): The population to evaluate.
        dimension (int): The dimensionality of the space (e.g., 3 for three-dimensional space).
        # fitness_function_jitted (function): A JIT-compiled fitness function that calculates the fitness of an individual.

    Returns:
        numpy.ndarray: An array containing the fitness of each individual in the population.
    """
    # Preallocate an array for fitness values
    fitness_values = np.empty(len(population), dtype=np.float64)

    # Parallel loop through the population
    for i in prange(len(population)):
        fitness_values[i] = fitness(population[i], dimension) # fitness se toma de aqui mismo dado que es estatico el objetivo de la optimizacion

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
        participants_fitnesses = np.array([fitnesses[idx] for idx in participants_idx])

        # Select the individual with the best (minimum) fitness
        winner_idx = participants_idx[np.argmin(participants_fitnesses)]

        # Check for unique selection
        if all(not np.array_equal(population[winner_idx], parent) for parent in selected_parents):
            selected_parents.append(population[winner_idx])

        # Handling non-unique cases might require a different approach or relax the constraint in Numba
        if len(selected_parents) == num_parents:
            break

    return selected_parents

# Fitness proportional selection pagina 80
def fps_selection(population, fitnesses, num_parents=2):
    population_size = len(population)

    # Transform fitness values for minimization (handle negatives and zero)
    # Shift fitness values so that the lowest fitness has the highest selection probability
    max_fitness = max(fitnesses)
    shifted_fitnesses = [max_fitness - f + 1 for f in fitnesses]

    # Calculate the sum of the shifted fitnesses
    total_shifted_fitness = sum(shifted_fitnesses)

    # Calculate selection probabilities based on shifted fitnesses
    probabilities = [f / total_shifted_fitness for f in shifted_fitnesses]

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

@njit  # Use nopython mode for better performance
def simple_arithmetic_crossover(parent1, parent2, k=3):
    '''
    Se deja fijo en k=3 porque es fijar la primera particula.
    '''
    # Initialize offspring as copies of the parents
    offspring1, offspring2 = np.copy(parent1), np.copy(parent2)
    # Perform the simple arithmetic crossover
    alpha = np.random.rand()  # Randomly choose a mixing ratio
    offspring1[k:] = alpha * parent1[k:] + (1 - alpha) * parent2[k:]
    offspring2[k:] = alpha * parent2[k:] + (1 - alpha) * parent1[k:]
    
    return offspring1, offspring2

@njit
def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    Perform Blend Crossover (BLX-α) on two parents to produce a single offspring.

    alpha: The alpha parameter controls the range of the blend.
    return: An array representing the offspring.
    """
    # Ensure the parents are numpy arrays
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    # Calculate the range between the parents
    I = np.abs(parent2 - parent1)

    # Generate offspring with genes within the range extended by alpha
    min_genes = np.minimum(parent1, parent2) - I * alpha
    max_genes = np.maximum(parent1, parent2) + I * alpha

    # Generate two offspring within the range
    offspring1 = min_genes + np.random.rand(*parent1.shape) * (max_genes - min_genes)
    offspring2 = min_genes + np.random.rand(*parent2.shape) * (max_genes - min_genes)

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

@njit
def uniform_mutation(individual, mutation_rate, simulation_box_length=1.):
    """
    Apply uniform mutation to an individual based on a given mutation rate.
    Each gene has a chance to mutate based on the mutation rate, with the new value
    randomly chosen uniformly within a range defined by the simulation box length.

    individual: NumPy array representing the individual's genes.
    mutation_rate: Probability of each gene undergoing mutation.
    simulation_box_length: Defines the range for the mutation.
    return: The mutated individual.
    """
    # Generate mutation probabilities for each gene
    mutation_probabilities = np.random.rand(len(individual))
    
    # Determine which genes will mutate
    mutations = mutation_probabilities < mutation_rate
    
    # Apply mutations
    if np.any(mutations):
        individual[mutations] = np.random.uniform(-simulation_box_length, simulation_box_length, size=np.sum(mutations))
    
    return individual

@njit
def non_uniform_mutation(individual, mutation_rate, mu=0, sigma=1.):
    """
    Apply non-uniform mutation to an individual based on a given mutation rate.
    Each gene that mutates is assigned a new value generated from a normal distribution
    centered at 'mu' with standard deviation 'sigma'.

    individual: NumPy array representing the individual's genes.
    mutation_rate: Probability of each gene undergoing mutation.
    mu: Mean of the normal distribution used for mutation.
    sigma: Standard deviation of the normal distribution, represents mutation step size.
    return: The mutated individual.
    """
    # Generate mutation probabilities for each gene
    mutation_probabilities = np.random.rand(len(individual))
    
    # Determine which genes will mutate
    mutations = mutation_probabilities < mutation_rate
    
    # Apply mutations
    if np.any(mutations):
        individual[mutations] = np.random.normal(mu, sigma, size=np.sum(mutations))
    
    return individual


#
# Replacement/Survivor Selection: Generate new population ( <replacement_strategy>_replacement )
#
def aged_based_replacement(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength, evaluate_population):
    '''
    pag 88
    '''
    new_population = []
    while len(new_population) < population_size:
        # Select parents
        parent1, parent2 = select_parents(population, fitnesses, num_parents=2)
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

def genitor_replacement(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength, evaluate_population, lambda_ratio=0.5):
    '''
    Replace worst  (tha half)
    '''
    # Sort the population by fitness in ascending order (for minimization)
    sorted_indices = np.argsort(fitnesses)
    sorted_population = np.array(population)[sorted_indices]

    # Split the population into the better and worse parts
    replacement_num = int(population_size*lambda_ratio)
    better_half = sorted_population[:replacement_num]
    worse_half = sorted_population[replacement_num:]
    
    # Generate new offspring to replace the worse half
    new_offspring = []
    while len(new_offspring) < len(worse_half):
        # Select parents
        parent1, parent2 = select_parents(population, fitnesses, num_parents=2)
        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2)
        # Mutate
        offspring1 = mutate(offspring1, mutation_rate, mutation_strength)
        offspring2 = mutate(offspring2, mutation_rate, mutation_strength)
        # Add the new offspring to the new offspring list
        new_offspring.append(offspring1)
        if len(new_offspring) < len(worse_half):
            new_offspring.append(offspring2)

    # Replace the worse half with new offspring
    new_population = np.concatenate((better_half, new_offspring[:len(worse_half)]))

    # Now we shuffle the new population to maintain genetic diversity
    np.random.shuffle(new_population)

    return new_population

# ??? Round-robin tournament descartadop orque es demasiado caro 32x
# def _round_robin_tournament(population, fitnesses, q):
#     # Calculate the number of individuals (μ)
#     mu = len(population)

#     # Initialize win counts
#     win_counts = np.zeros(mu)

#     # Perform round-robin tournament
#     for i in range(mu):
#         for _ in range(q):
#             # Select a random opponent
#             opponent_index = np.random.randint(mu)
#             # Assign a win if the individual is better than the randomly selected opponent
#             if fitnesses[i] < fitnesses[opponent_index]:
#                 win_counts[i] += 1

#     # Select the μ individuals with the greatest number of wins
#     selected_indices = np.argsort(win_counts)[-mu:]
#     return population[selected_indices], fitnesses[selected_indices]
# def round_robin_replacement(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength, evaluate_population,q=10):
    # Create a list to store offspring
    offspring_list = []
    while len(offspring_list) < population_size:
        # Select parents and produce offspring
        parent1, parent2 = select_parents(population, fitnesses)
        offspring1, offspring2 = crossover(parent1, parent2)
        offspring1 = mutate(offspring1, mutation_rate, mutation_strength)
        offspring2 = mutate(offspring2, mutation_rate, mutation_strength)
        offspring_list.extend([offspring1, offspring2])

    # Truncate the list if we have too many offspring
    offspring_list = offspring_list[:population_size]
    
    # Combine the offspring with the current population
    combined_population = np.vstack((population, offspring_list))
    combined_fitnesses = np.concatenate((fitnesses, [None] * population_size))  # Placeholder for offspring fitnesses

    # Recalculate fitnesses for the combined population
    # Assuming there is a function calculate_fitnesses that calculates the fitness for each individual
    combined_fitnesses = evaluate_population(combined_population, dimension)

    # Perform round-robin tournament to select the new population
    new_population, new_fitnesses = _round_robin_tournament(combined_population, combined_fitnesses, q)

    return new_population

def age_based_replacement_with_elitism(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength, evaluate_population):
    '''
    Perform age-based replacement with elitism.
    '''
    # Identify the elite individual (the one with the best fitness)
    elite_index = np.argmin(fitnesses)
    elite_individual = population[elite_index]
    elite_fitness = fitnesses[elite_index]

    new_population = []
    new_fitnesses = []

    while len(new_population) < population_size:
        # Select parents
        parent1, parent2 = select_parents(population, fitnesses, num_parents=2)
        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2)
        # Mutate
        offspring1 = mutate(offspring1, mutation_rate, mutation_strength)
        offspring2 = mutate(offspring2, mutation_rate, mutation_strength)
        # Evaluate the fitness of the offspring
        fitness1 = evaluate_population([offspring1], dimension)
        fitness2 = evaluate_population([offspring2], dimension)

        # Add offspring to the new population, ensuring we do not exceed the population size
        if len(new_population) < population_size:
            new_population.append(offspring1)
            new_fitnesses.append(fitness1)
        if len(new_population) < population_size:
            new_population.append(offspring2)
            new_fitnesses.append(fitness2)

    # Now we replace the worst individual with the elite if necessary
    # First, find the worst individual's index in the new population
    worst_index = np.argmax(new_fitnesses)
    if new_fitnesses[worst_index] > elite_fitness:
        # Replace the worst individual with the elite individual
        new_population[worst_index] = elite_individual
        new_fitnesses[worst_index] = elite_fitness

    return new_population

def mu_plus_lambda_replacement(select_parents, crossover, mutate, population, fitnesses, population_size, mutation_rate, dimension, mutation_strength, evaluate_population, lambda_ratio=7):
    '''
    Replace worst  (tha half)
    '''
    # Generate lambda offspring
    lambda_offspring = []
    for _ in range(int(population_size * lambda_ratio)):
        parent1, parent2 = select_parents(population, fitnesses, num_parents=2)
        offspring1, offspring2 = crossover(parent1, parent2)
        offspring1 = mutate(offspring1, mutation_rate, mutation_strength)
        offspring2 = mutate(offspring2, mutation_rate, mutation_strength)
        lambda_offspring.extend([offspring1, offspring2])
    
    # Evaluate the fitness of new offspring if not already evaluated
    offspring_fitnesses = evaluate_population(lambda_offspring, dimension)
    
    # Merge offspring with the original population
    combined_population = np.concatenate((population, lambda_offspring))
    combined_fitnesses = np.concatenate((fitnesses, offspring_fitnesses))
    
    # Sort combined population by fitness
    sorted_indices = np.argsort(combined_fitnesses)
    sorted_combined_population = combined_population[sorted_indices]
    
    # Select the top mu individuals to form the next generation
    new_population = sorted_combined_population[:population_size]
    
    return new_population

# (μ, λ) Selection descartado porque es mas para multimodal

#
# For an easier implementation on design
#
available_functions = {
    'selection': [
        tournament_selection,
        fps_selection,
        ranking_selection,
        sus_selection,
    ],
    'crossover': [
        uniform_crossover,
        simple_arithmetic_crossover,
        blx_alpha_crossover,
    ],
    'mutation': [
        add_perturbation_mutation,
        uniform_mutation,
        non_uniform_mutation,
    ],
    'replacement': [
        aged_based_replacement,
        genitor_replacement,
        mu_plus_lambda_replacement,
        age_based_replacement_with_elitism
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
    parent1, parent2 = fps_selection(p, f)
    
    print('parent1',parent1)
    print('parent2',parent2)
