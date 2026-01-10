import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

episodes = 5

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # ação aleatória
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episódio {ep + 1} | Recompensa: {total_reward}")

env.close()
