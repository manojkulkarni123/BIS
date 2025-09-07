"""
This code uses Particle Swarm Optimization (PSO) to solve a Job Scheduling problem where tasks are assigned 
to machines to minimize the total processing time (makespan).
"""

import random

# --------- Problem Definition ---------
# List of task processing times
processing_times = [5, 2, 7, 3, 8, 4]

num_tasks = len(processing_times)
num_machines = 3

# --------- PSO Parameters ---------
num_particles = 10
max_iterations = 50
w = 0.5       # inertia weight
c1 = 1.5      # cognitive coefficient
c2 = 1.5      # social coefficient

# --------- Fitness Function ---------
def fitness(position):
    machine_times = [0] * num_machines
    for task_index, machine in enumerate(position):
        machine_times[machine] += processing_times[task_index]
    return max(machine_times)

# --------- Initialize particles ---------
def initialize_particles():
    particles = []
    velocities = []
    pbest_positions = []
    pbest_scores = []

    for _ in range(num_particles):
        # Random assignment of tasks to machines
        position = [random.randint(0, num_machines - 1) for _ in range(num_tasks)]
        velocity = [0] * num_tasks  # will store tendency to move tasks
        particles.append(position)
        velocities.append(velocity)
        pbest_positions.append(position[:])
        pbest_scores.append(fitness(position))

    return particles, velocities, pbest_positions, pbest_scores

# --------- Update velocity ---------
def update_velocity(velocity, position, pbest, gbest):
    new_velocity = []
    for i in range(num_tasks):
        r1 = random.random()
        r2 = random.random()

        cognitive = c1 * r1 * (pbest[i] - position[i])
        social = c2 * r2 * (gbest[i] - position[i])
        
        inertia = w * velocity[i]
        vel = inertia + cognitive + social
        
        # Discrete handling: treat positive as tendency to increment, negative as decrement
        if vel > 0.5:
            new_velocity.append(1)
        elif vel < -0.5:
            new_velocity.append(-1)
        else:
            new_velocity.append(0)
    return new_velocity

# --------- Update position ---------
def update_position(position, velocity):
    new_position = position[:]
    for i in range(num_tasks):
        new_val = position[i] + velocity[i]
        if new_val < 0:
            new_val = 0
        if new_val >= num_machines:
            new_val = num_machines - 1
        new_position[i] = new_val
    return new_position

# --------- Main PSO Loop ---------
particles, velocities, pbest_positions, pbest_scores = initialize_particles()

# Initialize global best
gbest_index = pbest_scores.index(min(pbest_scores))
gbest_position = pbest_positions[gbest_index][:]
gbest_score = pbest_scores[gbest_index]

print(f"Initial best makespan: {gbest_score}")

for iteration in range(max_iterations):
    for i in range(num_particles):
        # Evaluate fitness
        score = fitness(particles[i])

        # Update personal best
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = particles[i][:]

        # Update global best
        if score < gbest_score:
            gbest_score = score
            gbest_position = particles[i][:]

    # Update velocities and positions
    for i in range(num_particles):
        velocities[i] = update_velocity(velocities[i], particles[i], pbest_positions[i], gbest_position)
        particles[i] = update_position(particles[i], velocities[i])

    print(f"Iteration {iteration+1}: Best makespan = {gbest_score}")

# Final Output
print("\nFinal Best Makespan:", gbest_score)
print("Task assignment to machines:")
for machine in range(num_machines):
    tasks = [i for i in range(num_tasks) if gbest_position[i] == machine]
    total_time = sum(processing_times[i] for i in tasks)
    print(f"Machine {machine+1}: Tasks {tasks}, Total time {total_time}")

"""
Output:

Initial best makespan: 13
Iteration 1: Best makespan = 13
Iteration 2: Best makespan = 13
Iteration 3: Best makespan = 12
Iteration 4: Best makespan = 12
Iteration 5: Best makespan = 12
Iteration 6: Best makespan = 12
Iteration 7: Best makespan = 12
Iteration 8: Best makespan = 12
Iteration 9: Best makespan = 12
Iteration 10: Best makespan = 11
Iteration 11: Best makespan = 11
Iteration 12: Best makespan = 11
Iteration 13: Best makespan = 11
Iteration 14: Best makespan = 11
Iteration 15: Best makespan = 11
Iteration 16: Best makespan = 11
Iteration 17: Best makespan = 11
Iteration 18: Best makespan = 11
Iteration 19: Best makespan = 11
Iteration 20: Best makespan = 11
Iteration 21: Best makespan = 11
Iteration 22: Best makespan = 11
Iteration 23: Best makespan = 11
Iteration 24: Best makespan = 11
Iteration 25: Best makespan = 11
Iteration 26: Best makespan = 11
Iteration 27: Best makespan = 11
Iteration 28: Best makespan = 11
Iteration 29: Best makespan = 11
Iteration 30: Best makespan = 11
Iteration 31: Best makespan = 11
Iteration 32: Best makespan = 11
Iteration 33: Best makespan = 11
Iteration 34: Best makespan = 11
Iteration 35: Best makespan = 11
Iteration 36: Best makespan = 11
Iteration 37: Best makespan = 11
Iteration 38: Best makespan = 11
Iteration 39: Best makespan = 11
Iteration 40: Best makespan = 11
Iteration 41: Best makespan = 11
Iteration 42: Best makespan = 11
Iteration 43: Best makespan = 11
Iteration 44: Best makespan = 11
Iteration 45: Best makespan = 11
Iteration 46: Best makespan = 11
Iteration 47: Best makespan = 11
Iteration 48: Best makespan = 11
Iteration 49: Best makespan = 11
Iteration 50: Best makespan = 11

Final Best Makespan: 11
Task assignment to machines:
Machine 1: Tasks [3, 4], Total time 11
Machine 2: Tasks [2, 5], Total time 11
Machine 3: Tasks [0, 1], Total time 7

"""
