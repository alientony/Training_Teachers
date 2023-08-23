import pygame
import random

# Define the Dot class
class Dot:
    def __init__(self):
        self.x = random.randint(0, 600)
        self.y = random.randint(0, 600)
        self.size = 5
        self.color = (0, 255, 0)  # green
        self.lifespan = random.randint(200, 500)  # lifespan in frames

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)
        self.lifespan -= 1
        self.x += random.randint(-1, 1)
        self.y += random.randint(-1, 1)

# Define the Predator class
class Predator:
    def __init__(self):
        self.x = random.randint(0, 600)
        self.y = random.randint(0, 600)
        self.size = 10
        self.color = (255, 0, 0)  # red
        self.lifespan = random.randint(200, 500)  # lifespan in frames

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)
        self.lifespan -= 1
        self.x += random.randint(-1, 1)
        self.y += random.randint(-1, 1)

# Initialize Pygame
pygame.init()

# Set the screen size
screen = pygame.display.set_mode((600, 600))

# Create a list to hold the dots and predators
dots = []
predators = []

# Main game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Create a new dot and predator
    if len(dots) < 100:  # limit number of dots
        dots.append(Dot())
    if len(predators) < 10:  # limit number of predators
        predators.append(Predator())

    # Draw the dots and predators
    screen.fill((0, 0, 0))
    for dot in dots:
        dot.draw(screen)
        # Remove the dot if it has "died"
        if dot.lifespan <= 0:
            dots.remove(dot)
    for predator in predators:
        predator.draw(screen)
        # Remove the predator if it has "died"
        if predator.lifespan <= 0:
            predators.remove(predator)

    # Check for collisions
    for dot in dots:
        for predator in predators:
            distance = ((predator.x - dot.x)**2 + (predator.y - dot.y)**2)**0.5
            if distance < predator.size:
                dots.remove(dot)
                break

    # Update the screen
    pygame.display.flip()

pygame.quit()

