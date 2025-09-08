//Ant Colony Optimization for TSP

import numpy as np
import random

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=2):
        self.distances = distances                    # Distance matrix
        self.pheromone = np.ones(self.distances.shape) / len(distances)  # Initial pheromone levels
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay                          # Pheromone evaporation rate
        self.alpha = alpha                          # Influence of pheromone
        self.beta = beta                            # Influence of distance
        self.n_cities = len(distances)

    def run(self):
        best_distance = float('inf')
        best_path = None
        
        for iteration in range(self.n_iterations):
            all_paths = self.generate_paths()
            self.evaporate_pheromone()
            
            for path, distance in all_paths:
                self.update_pheromone(path, distance)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            
            print(f"Iteration {iteration+1}: Best Distance = {best_distance}")
        
        return best_path, best_distance

    def generate_paths(self):
        paths = []
        for _ in range(self.n_ants):
            path = self.generate_single_path()
            distance = self.calculate_distance(path)
            paths.append((path, distance))
        return paths

    def generate_single_path(self):
        path = []
        visited = set()
        
        start = random.randint(0, self.n_cities - 1)
        path.append(start)
        visited.add(start)
        
        for _ in range(self.n_cities - 1):
            current = path[-1]
            probabilities = self.get_probabilities(current, visited)
            next_city = self.select_city(probabilities)
            path.append(next_city)
            visited.add(next_city)
        
        return path

    def get_probabilities(self, current, visited):
        pheromone = self.pheromone[current]
        heuristic = 1 / (self.distances[current] + 1e-10)  # Avoid division by zero
        prob = []
        
        for i in range(self.n_cities):
            if i not in visited:
                value = (pheromone[i] ** self.alpha) * (heuristic[i] ** self.beta)
                prob.append(value)
            else:
                prob.append(0)
        
        total = sum(prob)
        return [p / total if total > 0 else 0 for p in prob]

    def select_city(self, probabilities):
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        return len(probabilities) - 1

    def calculate_distance(self, path):
        distance = 0
        for i in range(len(path)):
            start = path[i]
            end = path[(i + 1) % self.n_cities]  # Wrap around to starting city
            distance += self.distances[start][end]
        return distance

    def evaporate_pheromone(self):
        self.pheromone *= (1 - self.decay)

    def update_pheromone(self, path, distance):
        contribution = 1 / distance
        for i in range(len(path)):
            start = path[i]
            end = path[(i + 1) % self.n_cities]
            self.pheromone[start][end] += contribution
            self.pheromone[end][start] += contribution


if __name__ == "__main__":
    # Example distance matrix (symmetric)
    distances = np.array([
        [0, 3, 4, 2, 7, 3],
        [3, 0, 4, 6, 3, 5],
        [4, 4, 0, 5, 8, 6],
        [2, 6, 5, 0, 6, 4],
        [7, 3, 8, 6, 0, 5],
        [3, 5, 6, 4, 5, 0]
    ])

    ant_colony = AntColony(distances, n_ants=10, n_iterations=100, decay=0.1, alpha=0.5, beta=3)
    best_path, best_distance = ant_colony.run()

    print("\nBest path found:", best_path)
    print("Total distance:", best_distance)

"""
Output:
Iteration 1: Best Distance = 24
Iteration 2: Best Distance = 22
Iteration 3: Best Distance = 22
Iteration 4: Best Distance = 22
Iteration 5: Best Distance = 22
Iteration 6: Best Distance = 22
Iteration 7: Best Distance = 22
Iteration 8: Best Distance = 22
Iteration 9: Best Distance = 22
Iteration 10: Best Distance = 22
Iteration 11: Best Distance = 22
Iteration 12: Best Distance = 22
Iteration 13: Best Distance = 22
Iteration 14: Best Distance = 22
Iteration 15: Best Distance = 22
Iteration 16: Best Distance = 22
Iteration 17: Best Distance = 22
Iteration 18: Best Distance = 22
Iteration 19: Best Distance = 22
Iteration 20: Best Distance = 22
Iteration 21: Best Distance = 22
Iteration 22: Best Distance = 22
Iteration 23: Best Distance = 22
Iteration 24: Best Distance = 22
Iteration 25: Best Distance = 22
Iteration 26: Best Distance = 22
Iteration 27: Best Distance = 22
Iteration 28: Best Distance = 22
Iteration 29: Best Distance = 22
Iteration 30: Best Distance = 22
Iteration 31: Best Distance = 22
Iteration 32: Best Distance = 22
Iteration 33: Best Distance = 22
Iteration 34: Best Distance = 22
Iteration 35: Best Distance = 22
Iteration 36: Best Distance = 22
Iteration 37: Best Distance = 22
Iteration 38: Best Distance = 22
Iteration 39: Best Distance = 22
Iteration 40: Best Distance = 22
Iteration 41: Best Distance = 22
Iteration 42: Best Distance = 22
Iteration 43: Best Distance = 22
Iteration 44: Best Distance = 22
Iteration 45: Best Distance = 22
Iteration 46: Best Distance = 22
Iteration 47: Best Distance = 22
Iteration 48: Best Distance = 22
Iteration 49: Best Distance = 22
Iteration 50: Best Distance = 22
Iteration 51: Best Distance = 22
Iteration 52: Best Distance = 22
Iteration 53: Best Distance = 22
Iteration 54: Best Distance = 22
Iteration 55: Best Distance = 22
Iteration 56: Best Distance = 22
Iteration 57: Best Distance = 22
Iteration 58: Best Distance = 22
Iteration 59: Best Distance = 22
Iteration 60: Best Distance = 22
Iteration 61: Best Distance = 22
Iteration 62: Best Distance = 22
Iteration 63: Best Distance = 22
Iteration 64: Best Distance = 22
Iteration 65: Best Distance = 22
Iteration 66: Best Distance = 22
Iteration 67: Best Distance = 22
Iteration 68: Best Distance = 22
Iteration 69: Best Distance = 22
Iteration 70: Best Distance = 22
Iteration 71: Best Distance = 22
Iteration 72: Best Distance = 22
Iteration 73: Best Distance = 22
Iteration 74: Best Distance = 22
Iteration 75: Best Distance = 22
Iteration 76: Best Distance = 22
Iteration 77: Best Distance = 22
Iteration 78: Best Distance = 22
Iteration 79: Best Distance = 22
Iteration 80: Best Distance = 22
Iteration 81: Best Distance = 22
Iteration 82: Best Distance = 22
Iteration 83: Best Distance = 22
Iteration 84: Best Distance = 22
Iteration 85: Best Distance = 22
Iteration 86: Best Distance = 22
Iteration 87: Best Distance = 22
Iteration 88: Best Distance = 22
Iteration 89: Best Distance = 22
Iteration 90: Best Distance = 22
Iteration 91: Best Distance = 22
Iteration 92: Best Distance = 22
Iteration 93: Best Distance = 22
Iteration 94: Best Distance = 22
Iteration 95: Best Distance = 22
Iteration 96: Best Distance = 22
Iteration 97: Best Distance = 22
Iteration 98: Best Distance = 22
Iteration 99: Best Distance = 22
Iteration 100: Best Distance = 22

Best path found: [0, 3, 2, 1, 4, 5]
Total distance: 22
"""
