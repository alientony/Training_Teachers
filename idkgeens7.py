import pygame
import numpy as np
import random
import multiprocessing
import csv
from itertools import permutations

# Initialize Pygame
pygame.init()

# Set the width and height of the screen (width, height)
size = (700, 500)
screen = pygame.display.set_mode(size)

# Set title of screen
pygame.display.set_caption("Traveling Salesman Problem")

# sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_tsp_path(n_cities):
    """Generate a random TSP path."""
    return random.sample(range(n_cities), n_cities)

def calculate_fitness(individual, cities):
    """Calculate fitness of an individual as the inverse of the total distance of the path."""
    total_distance = sum(np.sqrt((cities[individual[i]][0]-cities[individual[i-1]][0])**2 + 
                                 (cities[individual[i]][1]-cities[individual[i-1]][1])**2) 
                         for i in range(1, len(individual)))
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

def calculate_fitness_and_weights(individual, nets, cities):
    fitness = calculate_fitness(individual, cities)
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

def draw_path(individual, cities):
    """Draw the path represented by the given individual using Pygame."""
    # Clear the screen to white
    screen.fill((255, 255, 255))

    # Draw the cities
    for city in cities:
        pygame.draw.circle(screen, (0, 0, 0), (int(city[0]*700), int(city[1]*500)), 5)

    # Draw the path
    for i in range(len(individual) - 1):
        pygame.draw.line(screen, (255, 0, 0), (int(cities[individual[i]][0]*700), int(cities[individual[i]][1]*500)), 
                                             (int(cities[individual[i+1]][0]*700), int(cities[individual[i+1]][1]*500)))

    # Update the screen
    pygame.display.flip()

def genetic_algorithm(n_cities, nets, population_size=10000, generations=1000):
    # Initialize population
    population = [generate_tsp_path(n_cities) for _ in range(population_size)]
    learning_rate = 0.01

    # Create a list to store the best individual of each generation
    best_individuals = []

    # Generate random city locations
    cities = np.random.rand(n_cities, 2)

    for generation in range(generations):
        # Calculate fitness and weights
        with multiprocessing.Pool() as pool:
            results = pool.starmap(calculate_fitness_and_weights, [(individual, nets, cities) for individual in population])
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

            child_fitness = (calculate_fitness(child1, cities) + calculate_fitness(child2, cities)) / 2
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
    return best_individuals, population, cities

# Run the genetic algorithm
n_cities = 10  # or any other number of cities you want to use
nets = [2 * np.random.random((n_cities, 2)) - 1 for _ in range(10)]
best_individuals, _, cities = genetic_algorithm(n_cities, nets)

best_overall_individual = max(best_individuals, key=lambda individual: calculate_fitness(individual, cities))
draw_path(best_overall_individual, cities)


# Draw the path of the best overall individual
best_overall_individual = max(best_individuals, key=lambda individual: calculate_fitness(individual, cities))
draw_path(best_overall_individual, cities)

# Save the image
pygame.image.save(screen, 'best_overall_individual.png')

# Wait until the user closes the window
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

