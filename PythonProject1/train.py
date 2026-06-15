from env_jobshop import JobShopEnv
from dqn_agent import DQNAgent
from gantt_chart import plot_gantt

env = JobShopEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

EPISODES = 200

for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(200):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

# SHOW FINAL GANTT
plot_gantt(env.schedule_log, env.machines)