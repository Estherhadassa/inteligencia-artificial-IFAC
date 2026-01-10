import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions=4, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: [0.0] * actions)
        self.alpha = alpha      # taxa de aprendizado
        self.gamma = gamma      # desconto futuro
        self.epsilon = epsilon  # exploração
        self.actions = actions

    def choose_action(self, state):
        # Exploração
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        # Exploração do melhor valor
        return self.q_table[state].index(max(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])

        # Fórmula do Q-Learning
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
