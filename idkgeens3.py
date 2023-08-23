import numpy as np
import random
import multiprocessing
import csv

# sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_number():
    """Generate a random floating-point number between -10 and 10."""
    return random.uniform(-10, 10)

def calculate_fitness(individual):
    """Calculate fitness of an individual as the value of the function f(x) = -(x-2)^2."""
    return -(individual-2)**2

def crossover(parent1, parent2):
    """Perform crossover between two parents to create two children."""
    alpha = random.uniform(0, 1)
    child1 = alpha*parent1 + (1-alpha)*parent2
    child2 = alpha*parent2 + (1-alpha)*parent1
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    """Perform mutation on an individual with a given mutation rate."""
    return individual + mutation_rate*random.uniform(-1, 1)

def calculate_fitness_and_weights(individual, nets):
    fitness = calculate_fitness(individual)
    input_data = [individual/10]  # Normalize to range [-1, 1]
    total_weight = 0
    for net in nets:
        output = sigmoid(np.dot(input_data, net))
        yes_prob = output[1] / (output[0] + output[1])
        total_weight += yes_prob * fitness
    return (individual, fitness, total_weight)

def weighted_random_choice(population, weights):
    return random.choices(population, weights=weights, k=2)

def genetic_algorithm(nets, population_size=10000, generations=1000):
    # Initialize population
    population = [generate_number() for _ in range(population_size)]
    learning_rate = 0.01

    # Open the CSV file for writing
    with open('evolution_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['Generation', 'Max Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        for generation in range(generations):
            # Calculate fitness and weights
            with multiprocessing.Pool() as pool:
                results = pool.starmap(calculate_fitness_and_weights, [(individual, nets) for individual in population])
            fitnesses = {result[0]: result[1] for result in results}

            # Convert raw fitnesses to ranks
            ranked_population = sorted(population, key=fitnesses.get)
            rank_fitnesses = {individual: rank for rank, individual in enumerate(ranked_population, start=1)}

            # Print the maximum fitness at each generation
            print(f"Generation {generation}, Max Fitness: {max(fitnesses.values())}")

            # Write data to the CSV file
            writer.writerow({'Generation': generation, 'Max Fitness': max(fitnesses.values())})

            new_population = []
            for net in nets:
                parent1, parent2 = weighted_random_choice(ranked_population, rank_fitnesses.values())
                child1, child2 = crossover(parent1, parent2)
                child1, child2 = mutate(child1), mutate(child2)
                new_population.extend([child1, child2])

                child_fitness = (calculate_fitness(child1) + calculate_fitness(child2)) / 2
                if child_fitness > np.median(list(fitnesses.values())):
                    # Update weights
                    for individual in [parent1, parent2]:
                        input_data = np.array([individual/10])
                        output = sigmoid(np.dot(input_data, net))
                        yes_prob = output[1] / (output[0] + output[1])
                        grad = 2 * (yes_prob - 0.5) * sigmoid_derivative(output)
                        net -= learning_rate * np.outer(input_data, grad)

            population = new_population

    return population

# Run the genetic algorithm
nets = [2 * np.random.random((1, 2)) - 1 for _ in range(10)]
genetic_algorithm(nets)

