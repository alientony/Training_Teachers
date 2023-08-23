import numpy as np
import random
import string
import multiprocessing

# sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_string(length=10):
    """Generate a random string of fixed length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def calculate_fitness(individual, target):
    """Calculate fitness of an individual as the inverse of the hamming distance to the target."""
    return sum(individual[i] == target[i] for i in range(len(target)))

def calculate_fitness_and_weights(individual, target, nets):
    fitness = calculate_fitness(individual, target)
    input_data = [ord(char)/256 for char in individual] + [0]*(20 - len(individual))
    total_weight = 0
    for net in nets:
        output = sigmoid(np.dot(input_data, net))
        yes_prob = output[1] / (output[0] + output[1])
        total_weight += yes_prob * fitness
    return (individual, fitness, total_weight)

def crossover(parent1, parent2):
    """Perform crossover between two parents to create two children."""
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    """Perform mutation on an individual with a given mutation rate."""
    return ''.join(char if random.random() > mutation_rate else random.choice(string.ascii_letters + string.digits) for char in individual)



def weighted_random_choice(population, fitnesses, net):
    weights = []
    for individual in population:
        input_data = [ord(char)/256 for char in individual] + [0]*(20 - len(individual))
        output = sigmoid(np.dot(input_data, net))
        yes_prob = output[1] / (output[0] + output[1])
        weights.append(yes_prob * fitnesses[individual])
    return random.choices(population, weights=weights, k=2)

def genetic_algorithm(nets, target='HelloWorld', population_size=10000, generations=1000):
    population = [generate_string(len(target)) for _ in range(population_size)]
    learning_rate = 0.01

    for generation in range(generations):
        with multiprocessing.Pool() as pool:
            results = pool.starmap(calculate_fitness_and_weights, [(individual, target, nets) for individual in population])
        fitnesses = {result[0]: result[1] for result in results}
        avg_fitness = sum(fitnesses.values()) / len(fitnesses)

        # Print the maximum fitness at each generation
        print(f"Generation {generation}, Max Fitness: {max(fitnesses.values())}")

        new_population = []
        for net in nets:
            parent1, parent2 = weighted_random_choice(population, fitnesses, net)
            child1, child2 = crossover(parent1, parent2)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])

            child_fitness = (calculate_fitness(child1, target) + calculate_fitness(child2, target)) / 2
            if child_fitness > avg_fitness:
                # Update weights
                for individual in [parent1, parent2]:
                    input_data = np.array([ord(char)/256 for char in individual] + [0]*(20 - len(individual)))
                    output = sigmoid(np.dot(input_data, net))
                    yes_prob = output[1] / (output[0] + output[1])
                    grad = 2 * (yes_prob - 0.5) * sigmoid_derivative(output)
                    net -= learning_rate * np.outer(input_data, grad)

        population = new_population

    return population


# Run the genetic algorithm
nets = [2 * np.random.random((20, 2)) - 1 for _ in range(10)]
genetic_algorithm(nets)

