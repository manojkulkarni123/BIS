import random

# Panel types (cost, power)
PANEL_TYPES = [
    {"cost": 100, "power": 250},
    {"cost": 150, "power": 300},
    {"cost": 200, "power": 400}
]

# GEA parameters
POP_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.2

# Gene = [num_panels (1–10), panel_type (0–2)]


def random_gene():
    return [random.randint(1, 10), random.randint(0, 2)]


def gene_expression(gene):
    num, p_type = gene
    cost = num * PANEL_TYPES[p_type]["cost"]
    power = num * PANEL_TYPES[p_type]["power"]
    efficiency = power / cost
    return efficiency


def mutate(gene):
    if random.random() < MUTATION_RATE:
        gene[0] = random.randint(1, 10)  # mutate number of panels
    if random.random() < MUTATION_RATE:
        gene[1] = random.randint(0, 2)   # mutate panel type
    return gene


def crossover(parent1, parent2):
    return [parent1[0], parent2[1]]  # simple crossover


def select(pop, fitnesses):
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    return random.choices(pop, weights=probs, k=2)


# Initialize population
population = [random_gene() for _ in range(POP_SIZE)]

best_gene = None
best_score = -1

for gen in range(GENERATIONS):
    # Gene expression and fitness
    fitnesses = [gene_expression(gene) for gene in population]

    # Save best
    max_fit = max(fitnesses)
    if max_fit > best_score:
        best_score = max_fit
        best_gene = population[fitnesses.index(max_fit)]

    # New population
    new_pop = []
    while len(new_pop) < POP_SIZE:
        parent1, parent2 = select(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)

    population = new_pop

# Final Output
print("Best Design Found:")
print(f"  Number of Panels: {best_gene[0]}")
print(f"  Panel Type: {best_gene[1]}")
print(f"  Efficiency (Power per Dollar): {best_score:.4f}")


//Output:
/*
Best Design Found:
  Number of Panels: 3
  Panel Type: 0
  Efficiency (Power per Dollar): 2.5000
*/
