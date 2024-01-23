import numpy as np

T = np.array([[0.7, 0.3], [0.4, 0.6]])
Z = np.array([[0.1, 0.9], [0.7, 0.3]])

measurement = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
initial_state = np.array([0.5, 0.5])
def bayes_filter(T, Z, measurement, initial_state):
    current_state = initial_state
    results = []

    for z in measurement:
        current_state = np.dot(T.T, current_state)
        current_state = current_state*Z[:,z]
        current_state = current_state / sum(current_state)
        results.append(current_state)

    return results

results = bayes_filter(T, Z, measurement, initial_state)

for t, state in enumerate(results, 1):
    print(f"Time {t}: P(X_t=0) = {state[1]:.4f}, P(X_t=1) = {state[0]:.4f}")
