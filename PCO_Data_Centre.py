import numpy as np
import matplotlib.pyplot as plt

# Energy function: combines load balance + temperature penalty
def energy(load, temp):
    # Ideal load = 50%, ideal temp = 40°C
    return (load - 50)**2 + (temp - 40)**2

# Initialize parameters
grid_size = (10, 10)
iterations = 100
np.random.seed(42)

# Each cell: server load (%) and temperature (°C)
loads = np.random.uniform(20, 90, grid_size)
temps = np.random.uniform(30, 70, grid_size)
neighborhood_radius = np.ones(grid_size, dtype=int)

# For visualization
avg_energy_over_time = []

for it in range(iterations):
    new_loads = np.copy(loads)
    new_temps = np.copy(temps)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            r = neighborhood_radius[i, j]
            
            # Collect neighbors (with wrap-around)
            neighbors = []
            for x in range(-r, r + 1):
                for y in range(-r, r + 1):
                    ni, nj = (i + x) % grid_size[0], (j + y) % grid_size[1]
                    neighbors.append((loads[ni, nj], temps[ni, nj]))
            
            # Compute energy of neighbors
            neighbor_energy = np.array([energy(l, t) for l, t in neighbors])
            best_neighbor = neighbors[np.argmin(neighbor_energy)]
            
            # Update load and temperature toward best neighbor
            new_loads[i, j] = (loads[i, j] + best_neighbor[0]) / 2
            new_temps[i, j] = (temps[i, j] + best_neighbor[1]) / 2
            
            # Adapt neighborhood based on improvement
            old_e = energy(loads[i, j], temps[i, j])
            new_e = energy(new_loads[i, j], new_temps[i, j])
            
            if new_e < old_e:  # improvement
                neighborhood_radius[i, j] = max(1, neighborhood_radius[i, j] - 1)
            else:
                neighborhood_radius[i, j] = min(3, neighborhood_radius[i, j] + 1)
    
    # Update grids
    loads, temps = new_loads, new_temps
    
    # Track average energy
    avg_energy_over_time.append(np.mean(energy(loads, temps)))

# --- Results ---
print("Final Average Energy:", avg_energy_over_time[-1])
print("Average Load:", np.mean(loads))
print("Average Temperature:", np.mean(temps))

# Plot convergence
plt.plot(avg_energy_over_time, color='teal')
plt.title("Energy Efficiency Improvement Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Average Energy (Lower is Better)")
plt.grid(True)
plt.show()

# Show final load distribution
plt.imshow(loads, cmap='coolwarm')
plt.title("Final Server Load Distribution")
plt.colorbar(label='CPU Load (%)')
plt.show()

explain this in detail 
