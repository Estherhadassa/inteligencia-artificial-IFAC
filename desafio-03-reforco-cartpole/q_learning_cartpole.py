import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Ambiente (sem render para treinar rápido)
env = gym.make("CartPole-v1")

# Parâmetros de discretização
NUM_BINS = (6, 12, 6, 12)
OBS_SPACE = env.observation_space.high

# Limites manuais (evita infinito)
OBS_SPACE[1] = 5
OBS_SPACE[3] = 5

bins = [
    np.linspace(-OBS_SPACE[i], OBS_SPACE[i], NUM_BINS[i] - 1)
    for i in range(4)
]

def discretize(obs):
    return tuple(
        np.digitize(obs[i], bins[i]) for i in range(4)
    )

# Q-table
q_table = np.zeros(NUM_BINS + (env.action_space.n,))

# Hiperparâmetros
alpha = 0.1        # taxa de aprendizado
gamma = 0.99       # desconto
epsilon = 1.0      # exploração inicial
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 1000
rewards = []

# Treinamento
for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Atualização Q-learning
        best_next = np.max(q_table[next_state])
        q_table[state][action] += alpha * (
            reward + gamma * best_next - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 100 == 0:
        print(f"Episódio {ep} | Recompensa: {total_reward}")

env.close()
plt.plot(rewards)
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.title("Aprendizado por Reforço — CartPole (Q-learning)")
plt.show()
