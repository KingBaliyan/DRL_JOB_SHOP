import numpy as np
from env_jobshop import JobShopEnv
from dqn_agent import DQNAgent
from gantt_chart import plot_gantt
import torch


# -------------------------------
# CONFIGURATION
# -------------------------------
NUM_JOBS = 3
NUM_MACHINES = 3
EPISODES = 300
MAX_STEPS = 50


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    print("\n=== DRL JOB SHOP SCHEDULING STARTED ===\n")

    # Initialize environment
    env = JobShopEnv(jobs=NUM_JOBS, machines=NUM_MACHINES)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize DQN Agent
    agent = DQNAgent(state_size, action_size)

    all_rewards = []

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        all_rewards.append(total_reward)

        if episode % 20 == 0:
            print(f"Episode {episode}/{EPISODES} | Reward: {total_reward}")

    print("\n=== TRAINING COMPLETED ===\n")

    # -------------------------------
    # TESTING PHASE (FOR GANTT CHART)
    # -------------------------------
    print("Generating final schedule for visualization...\n")

    state, _ = env.reset()
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        state_tensor = torch.FloatTensor(state)
        action = torch.argmax(agent.model(state_tensor)).item()
        state, _, done, _, _ = env.step(action)
        step_count += 1

    # -------------------------------
    # GANTT CHART VISUALIZATION
    # -------------------------------
    print("Displaying Gantt Chart...\n")
    plot_gantt(env.schedule_log, NUM_MACHINES)


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()