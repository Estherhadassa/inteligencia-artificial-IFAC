import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

NUM_BUCKETS = (6, 12)
NUM_ACTIONS = env.action_space.n

STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-50, 50)

q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 500
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


def discretize_state(state):
    ratios = []
    for i in range(len(state)):
        low, high = STATE_BOUNDS[i]
        ratios.append((state[i] - low) / (high - low))

    new_state = []
    for i in range(len(NUM_BUCKETS)):
        new_state.append(int(round((NUM_BUCKETS[i] - 1) * ratios[i])))

    return tuple(
        min(NUM_BUCKETS[i] - 1, max(0, new_state[i]))
        for i in range(len(NUM_BUCKETS))
    )


for episode in range(EPISODES):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state)

        q_table[state][action] += LEARNING_RATE * (
            reward + DISCOUNT * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % 50 == 0:
        print(f"Episódio {episode} | Recompensa: {total_reward}")

env.close()
print("Treinamento concluído!")
