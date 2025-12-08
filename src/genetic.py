import numpy as np

def fitness(threshold, y_true, y_proba):
    preds = (y_proba >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    return tp / (tp + fp + 1e-5)

def evolve(y_true, y_proba, generations=20, pop_size=10):
    population = np.random.uniform(0.1, 0.9, pop_size)

    for gen in range(generations):
        scores = [fitness(t, y_true, y_proba) for t in population]
        parents = np.argsort(scores)[-2:]

        child = (population[parents[0]] + population[parents[1]]) / 2
        child = child + np.random.uniform(-0.05, 0.05)
        child = np.clip(child, 0.1, 0.9)

        worst = np.argmin(scores)
        population[worst] = child

    best = population[np.argmax(scores)]
    return best
