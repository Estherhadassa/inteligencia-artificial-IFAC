from environment import Xadrez4JogadoresEnv
from agent import QLearningAgent
import random

EPISODES = 500

env = Xadrez4JogadoresEnv()
agent = QLearningAgent()

for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        # Jogador A é o agente que aprende
        action = agent.choose_action(state)
        next_state, reward, done = env.step("♔", action)


        agent.learn(state, action, reward, next_state)
        state = next_state

        # Jogadores B, C e D fazem movimentos aleatórios
        for other in ["♕", "♖", "♜"]:
            if done:
                break
            random_action = random.randint(0, 3)
            _, _, done = env.step(other, random_action)

    if episode % 50 == 0:
        print(f"Treinando... episódio {episode}")

print("✅ Treinamento finalizado!")
