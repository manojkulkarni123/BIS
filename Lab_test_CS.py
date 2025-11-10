import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------
# Plant + PID simulation
# ------------------------
def simulate_pid(Kp, Ki, Kd, plant_a=1.0, plant_b=1.0,
                 t_final=10.0, dt=0.01, setpoint=1.0, u_limits=None):
    steps = int(t_final / dt) + 1
    t = np.linspace(0, t_final, steps)
    y = np.zeros(steps)
    u = np.zeros(steps)
    e = np.zeros(steps)
    integral = 0.0
    e_prev = 0.0

    for i in range(1, steps):
        e[i] = setpoint - y[i-1]
        integral += e[i] * dt
        derivative = (e[i] - e_prev) / dt
        e_prev = e[i]

        u_unclamped = Kp * e[i] + Ki * integral + Kd * derivative
        if u_limits is not None:
            u[i] = np.clip(u_unclamped, u_limits[0], u_limits[1])
        else:
            u[i] = u_unclamped

        # forward Euler for dy/dt = -a*y + b*u
        y[i] = y[i-1] + dt * (-plant_a * y[i-1] + plant_b * u[i])

    # metrics
    ise = np.sum((setpoint - y)**2) * dt
    overshoot = max(0.0, np.max(y) - setpoint)
    tol = 0.02 * abs(setpoint)
    settling_time = t[-1]
    for i in range(len(t)):
        if np.all(np.abs(setpoint - y[i:]) <= tol):
            settling_time = t[i]
            break

    return {"t": t, "y": y, "u": u, "e": e,
            "ise": ise, "overshoot": overshoot, "settling_time": settling_time}

# ------------------------
# Objective function
# ------------------------
def pid_cost(gains):
    Kp, Ki, Kd = gains
    if np.any(np.array(gains) < 0):
        # heavy penalty for negative gains
        return 1e6 + np.sum(np.maximum(0, -np.array(gains))) * 1e6

    sim = simulate_pid(Kp, Ki, Kd, plant_a=1.0, plant_b=1.0,
                       t_final=8.0, dt=0.01, setpoint=1.0, u_limits=(-10, 10))
    cost = sim["ise"] + 50.0 * sim["overshoot"] + 10.0 * sim["settling_time"]
    return cost

# ------------------------
# Cuckoo Search with Lévy flights (Mantegna)
# ------------------------
def cuckoo_search_levy(objective, bounds, n_nests=25, n_iter=150, pa=0.25,
                       alpha=0.5, beta=1.5, seed=None):
    """
    Cuckoo Search using Lévy flights (Mantegna's algorithm).
    - bounds: list of (low, high) for each dimension
    - pa: fraction of nests to abandon each iteration
    - alpha: global step scale
    - beta: Lévy exponent (1 < beta <= 2 commonly)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = len(bounds)
    nests = np.zeros((n_nests, dim))
    for d in range(dim):
        low, high = bounds[d]
        nests[:, d] = np.random.uniform(low, high, size=n_nests)

    costs = np.array([objective(nests[i]) for i in range(n_nests)])
    best_idx = np.argmin(costs)
    best_nest = nests[best_idx].copy()
    best_cost = costs[best_idx]

    history = {"best_cost": [], "best_nest": []}

    # Mantegna's sigma for numerator u (std, not variance)
    # sigma_u formula (std) derived from Mantegna's algorithm:
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1.0 / beta)
    sigma_v = 1.0  # v is standard normal (std=1)

    for it in range(n_iter):
        # Levy flight proposals
        for i in range(n_nests):
            # Mantegna: u ~ N(0, sigma_u^2) but numpy takes std -> pass sigma_u
            u_levy = np.random.normal(0.0, sigma_u, size=dim)
            v_levy = np.random.normal(0.0, sigma_v, size=dim)
            # avoid div by zero by adding a tiny epsilon to v
            levy_step = u_levy / (np.abs(v_levy) ** (1.0 / beta) + 1e-16)

            # scale the Levy step and optionally direct it relative to best
            # This hybrid (levy * (current - best)) helps exploitation around good regions
            step = alpha * levy_step * (nests[i] - best_nest)

            candidate = nests[i] + step

            # enforce bounds
            for d in range(dim):
                low, high = bounds[d]
                candidate[d] = np.clip(candidate[d], low, high)

            cand_cost = objective(candidate)
            if cand_cost < costs[i]:
                nests[i] = candidate
                costs[i] = cand_cost

        # discovery: abandon worst nests and replace near best
        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            worst_idx = np.argsort(costs)[-n_abandon:]
            for idx in worst_idx:
                new_nest = np.zeros(dim)
                for d in range(dim):
                    low, high = bounds[d]
                    # small random around best_nest plus some global jitter
                    new_nest[d] = best_nest[d] + 0.1 * (high - low) * np.random.randn()
                    new_nest[d] = np.clip(new_nest[d], low, high)
                nests[idx] = new_nest
                costs[idx] = objective(new_nest)

        # update global best
        cur_best_idx = np.argmin(costs)
        if costs[cur_best_idx] < best_cost:
            best_cost = costs[cur_best_idx]
            best_nest = nests[cur_best_idx].copy()

        history["best_cost"].append(best_cost)
        history["best_nest"].append(best_nest.copy())

        # progress print
        if (it + 1) % max(1, (n_iter // 10)) == 0 or it < 5:
            print(f"Iter {it+1}/{n_iter} - best_cost: {best_cost:.6f} - best_gains: {best_nest}")

    return best_nest, best_cost, history

# ------------------------
# Run optimizer
# ------------------------
if __name__ == "__main__":
    bounds = [(0.0, 50.0), (0.0, 50.0), (0.0, 10.0)]  # Kp, Ki, Kd
    best_gains, best_cost, history = cuckoo_search_levy(pid_cost, bounds,
                                                        n_nests=30, n_iter=120,
                                                        pa=0.25, alpha=0.6, beta=1.5, seed=42)

    print("\n=== Best found ===")
    print("Kp = {:.4f}, Ki = {:.4f}, Kd = {:.4f}".format(*best_gains))
    print("Cost =", best_cost)

    sim = simulate_pid(*best_gains, plant_a=1.0, plant_b=1.0,
                       t_final=8.0, dt=0.01, setpoint=1.0, u_limits=(-10, 10))
    t, y, u = sim["t"], sim["y"], sim["u"]
    print("ISE = {:.6f}, Overshoot = {:.6f}, Settling time = {:.3f}s".format(
        sim["ise"], sim["overshoot"], sim["settling_time"]))

"""
Iter 1/120 - best_cost: 1.849887 - best_gains: [34.3998781  18.85269417  0.30040203]
Iter 2/120 - best_cost: 1.439312 - best_gains: [38.18741665 16.20018843  0.        ]
Iter 3/120 - best_cost: 1.407964 - best_gains: [37.11288756 18.04946665  0.        ]
Iter 4/120 - best_cost: 1.239247 - best_gains: [47.16391026 17.16473051  0.        ]
Iter 5/120 - best_cost: 1.239247 - best_gains: [47.16391026 17.16473051  0.        ]
Iter 12/120 - best_cost: 1.239237 - best_gains: [50.        17.2793083  0.       ]
Iter 24/120 - best_cost: 1.239237 - best_gains: [50.         17.38329886  0.        ]
Iter 36/120 - best_cost: 1.239237 - best_gains: [50.         17.39438218  0.        ]
Iter 48/120 - best_cost: 1.239237 - best_gains: [50.         17.39530956  0.        ]
Iter 60/120 - best_cost: 1.239237 - best_gains: [50.         17.39531427  0.        ]
Iter 72/120 - best_cost: 1.239237 - best_gains: [50.         17.39531576  0.        ]
Iter 84/120 - best_cost: 1.239237 - best_gains: [50.         17.39531609  0.        ]
Iter 96/120 - best_cost: 1.239237 - best_gains: [50.        17.3953161  0.       ]
Iter 108/120 - best_cost: 1.239237 - best_gains: [50.        17.3953161  0.       ]
Iter 120/120 - best_cost: 1.239237 - best_gains: [50.        17.3953161  0.       ]

=== Best found ===
Kp = 50.0000, Ki = 17.3953, Kd = 0.0000
Cost = 1.2392369026691938
ISE = 0.039237, Overshoot = 0.000000, Settling time = 0.120s

"""
