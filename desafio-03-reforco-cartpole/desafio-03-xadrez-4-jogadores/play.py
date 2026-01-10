from environment import Xadrez4JogadoresEnv
from agent import QLearningAgent
import random

env = Xadrez4JogadoresEnv()
agent = QLearningAgent()

# 🔁 Treinamento rápido
for _ in range(300):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step("♔", action)
        agent.learn(state, action, reward, state)

        for other in ["♕", "♖", "♜"]:
            if done:
                break
            env.step(other, random.randint(0, 3))


# 🎮 Jogo contra o humano
state = env.reset()
env.print_board()

print("🎮 Você é o jogador ♕")
print("Movimentos: 0=cima | 1=baixo | 2=esquerda | 3=direita")

done = False
while not done:
    # Turno da IA
    action = agent.choose_action(state)
    state, _, done = env.step("♔", action)
    env.print_board()

    if done:
        break

    # Turno do humano
    move = int(input("Sua jogada: "))
    state, _, done = env.step("♕", move)
    env.print_board()

print("🏁 Jogo encerrado!")
