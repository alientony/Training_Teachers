import numpy as np
import random
import multiprocessing
import csv
from itertools import permutations

# sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_tsp_path(n_cities):
    """Generate a random TSP path."""
    return random.sample(range(n_cities), n_cities)

def calculate_fitness(individual):
    """Calculate fitness of an individual as the inverse of the total distance of the path."""
    total_distance = sum(np.sqrt((individual[i]-individual[i-1])**2) for i in range(1, len(individual)))
    return -total_distance

def crossover(parent1, parent2):
    """Perform crossover between two parents to create two children."""
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    """Perform mutation on an individual with a given mutation rate."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)  # Select two cities
        individual[i], individual[j] = individual[j], individual[i]  # Swap the two cities
    return individual

def calculate_fitness_and_weights(individual, nets):
    fitness = calculate_fitness(individual)
    input_data = [city/len(individual) for city in individual]  # Normalize to range [0, 1]
    total_weight = 0
    for net in nets:
        output = sigmoid(np.dot(input_data, net))
        yes_prob = output[1] / (output[0] + output[1])
        total_weight += yes_prob * fitness
    return (individual, fitness, total_weight)

def weighted_random_choice(population, weights):
    unique_population = list(set(tuple(individual) for individual in population))
    return random.choices(population, weights=[weights[unique_population.index(tuple(individual))] for individual in population], k=2)

def genetic_algorithm(n_cities, nets, population_size=10000, generations=10000):
    # Initialize population
    population = [generate_tsp_path(n_cities) for _ in range(population_size)]
    learning_rate = 0.01
    
    # Create a list to store the best individual of each generation
    best_individuals = []

    for generation in range(generations):
        # Calculate fitness and weights
        with multiprocessing.Pool() as pool:
            results = pool.starmap(calculate_fitness_and_weights, [(individual, nets) for individual in population])
        fitnesses = {tuple(result[0]): result[1] for result in results}

        # Convert raw fitnesses to ranks
        ranked_population = sorted(population, key=lambda individual: fitnesses[tuple(individual)])
        rank_fitnesses = {tuple(individual): rank for rank, individual in enumerate(ranked_population, start=1)}
        
        # Print the maximum fitness at each generation
        print(f"Generation {generation}, Max Fitness: {max(fitnesses.values())}")

        # Store the best individual of the current generation
        best_individual = max(fitnesses, key=fitnesses.get)
        best_individuals.append(best_individual)

        new_population = []
        for net in nets:
            parent1, parent2 = weighted_random_choice(ranked_population, list(rank_fitnesses.values()))
            child1, child2 = crossover(parent1, parent2)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])

            child_fitness = (calculate_fitness(child1) + calculate_fitness(child2)) / 2
            if child_fitness > np.median(list(fitnesses.values())):
                # Update weights
                for individual in [parent1, parent2]:
                    input_data = np.array([city/len(individual) for city in individual])
                    output = sigmoid(np.dot(input_data, net))
                    yes_prob = output[1] / (output[0] + output[1])
                    grad = 2 * (yes_prob - 0.5) * sigmoid_derivative(output)
                    net -= learning_rate * np.outer(input_data, grad)

        population = new_population

    # Write the best individual of each generation to a CSV file
    with open('best_individuals.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'Best Individual'])
        for i, individual in enumerate(best_individuals):
            writer.writerow([i, individual])

    # Return the list of best individuals along with the final population
    return best_individuals, population

# Run the genetic algorithm
nets = [2 * np.random.random((10, 2)) - 1 for _ in range(10)]
genetic_algorithm(10, nets)


