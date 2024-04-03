'''
    This file is for storing all the available strategies for each evolutionary step
    to be used in the evolutionary algorithm
'''

import numpy as np

#
# Fitness (Lennard-Jones potential)
#
def _particle_lennard_jones(vector_1, vector_2, sigma=1.0, epsilon=1.0, penalization_strenght=1e3):
    '''
    vector_1: position vector of the particule 1
    sigma: the zero-potential distance.
    epsilon: the depth of the potential well
    '''
    distance = np.linalg.norm(vector_1 - vector_2)
    if distance==0:
        # Penailzation
        return penalization_strenght
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
# These will remain static for the design optimization
#
def _generate_individual(number_of_particles, dimension, simulation_box_bounds):
    '''
    Generate a random individual representing a molecule's structure.   
    '''
    return np.random.uniform(
        low=simulation_box_bounds[0],
        high=simulation_box_bounds[1],
        size=( number_of_particles * dimension )
    )

def initialize_population(population_size, number_of_particles, dimension, simulation_box_bounds):
    """
    Initialize a population of random individuals.
    """
    return [
        _generate_individual(number_of_particles, dimension, simulation_box_bounds) for _ in range(population_size)
    ]

def evaluate_population(population, dimension, fitness):
    """
    Evaluate the fitness of each individual in the population.
    """
    return [fitness(individual, dimension) for individual in population]

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
            mutation = np.random.uniform(-mutation_strength, mutation_strength)
            individual[i] += mutation
    return individual


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
}