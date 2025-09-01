import numpy as np
import yfinance as yf
import random

# ----------------------------
# 1. Fetch Stock Data
# ----------------------------
tickers = ["HAL.NS", "SUZLON.NS", "SBIN.NS", "HDFCBANK.NS"]

data = yf.download(tickers, start="2024-01-01", end="2025-01-01")["Close"]

# Daily returns
returns = data.pct_change().dropna()

# Expected annual returns & covariance
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
risk_free_rate = 0.06

n_assets = len(tickers)
budget = 100000

# ----------------------------
# 2. Fitness Functions
# ----------------------------
def portfolio_return(weights):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def sharpe_ratio(weights):
    ret = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    return (ret - risk_free_rate) / vol if vol > 0 else 0

# ----------------------------
# 3. GA Parameters
# ----------------------------
POP_SIZE = 200
GENS = 500
MUTATION_RATE = 0.2
ELITE_FRAC = 0.1

def init_population():
    return [np.random.dirichlet(np.ones(n_assets)) for _ in range(POP_SIZE)]

def select(pop):
    contenders = random.sample(pop, 5)
    return max(contenders, key=sharpe_ratio)

def crossover(p1, p2):
    alpha = random.random()
    child = alpha * p1 + (1 - alpha) * p2
    return child / np.sum(child)

def mutate(weights):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(n_assets), 2)
        change = random.uniform(-0.05, 0.05)
        weights[i] = max(0, weights[i] + change)
        weights[j] = max(0, weights[j] - change)
        weights /= np.sum(weights)
    return weights

# ----------------------------
# 4. Run GA
# ----------------------------
pop = init_population()
best = max(pop, key=sharpe_ratio)

for g in range(GENS):
    new_pop = []
    elite_count = int(ELITE_FRAC * POP_SIZE)
    pop.sort(key=sharpe_ratio, reverse=True)
    new_pop.extend(pop[:elite_count])

    while len(new_pop) < POP_SIZE:
        p1, p2 = select(pop), select(pop)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)

    pop = new_pop
    gen_best = max(pop, key=sharpe_ratio)
    if sharpe_ratio(gen_best) > sharpe_ratio(best):
        best = gen_best

# ----------------------------
# 5. Results
# ----------------------------
allocation = best * budget

print("\n--- Optimal Portfolio Allocation ---")
for t, w, amt in zip(tickers, best, allocation):
    print(f"{t}: {w:.2%} → ₹{amt:.0f}")

print("\nExpected Annual Return:", portfolio_return(best))
print("Volatility (Risk):", portfolio_volatility(best))
print("Sharpe Ratio:", sharpe_ratio(best))

//Sample Output:
/*
--- Optimal Portfolio Allocation ---
HAL.NS: 39.07% → ₹39075
SUZLON.NS: 0.00% → ₹0
SBIN.NS: 13.20% → ₹13195
HDFCBANK.NS: 47.73% → ₹47730

Expected Annual Return: 0.5161504476791288
Volatility (Risk): 0.32761040524078033
Sharpe Ratio: 1.392356409876166
*/
