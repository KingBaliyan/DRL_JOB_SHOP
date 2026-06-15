import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import csv
import gym
from gym import spaces

# ==========================================
# 1. ENVIRONMENT (Job Shop Scheduling)
# ==========================================
class JobShopEnv(gym.Env):
    def __init__(self, data_file=None, num_jobs=3, num_machines=3, breakdown_prob=0.0):
        super(JobShopEnv, self).__init__()
        self.breakdown_prob = breakdown_prob
        
        if data_file is not None and os.path.exists(data_file):
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                next(reader) # skip header
                data = [[int(x) for x in row] for row in reader if row]
            self.processing_times = np.array(data)
            self.num_jobs, self.num_machines = self.processing_times.shape
        else:
            self.num_jobs = num_jobs
            self.num_machines = num_machines
            self.processing_times = np.random.randint(1, 10, (self.num_jobs, self.num_machines))
            
        # Action: Select a job to schedule next (0 to num_jobs-1)
        self.action_space = spaces.Discrete(self.num_jobs)
        
        # State: Current operation index of each job + machine completion times
        # Shape: (num_jobs + num_machines, )
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(self.num_jobs + self.num_machines,), 
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.job_steps = np.zeros(self.num_jobs, dtype=int)
        self.machine_times = np.zeros(self.num_machines, dtype=np.float32)
        return self._get_state(), {}

    def _get_state(self):
        # State representation: [job_steps, machine_times]
        state = np.concatenate((self.job_steps, self.machine_times))
        return state.astype(np.float32)

    def step(self, action):
        # If the chosen job is already completely finished, invalid action
        if self.job_steps[action] >= self.num_machines:
            # Large penalty for invalid action to discourage it
            return self._get_state(), -100.0, False, False, {}

        # The machine required for this job's current step
        machine_idx = self.job_steps[action]
        
        # Calculate start and end time
        # The job can only start when the machine is free
        start_time = self.machine_times[machine_idx]
        duration = self.processing_times[action, machine_idx]
        
        # Breakdown Simulation (random penalty delay if enabled)
        if self.breakdown_prob > 0.0 and random.random() < self.breakdown_prob:
            duration += random.randint(5, 20)
            
        end_time = start_time + duration
        
        # Update environment state
        self.machine_times[machine_idx] = end_time
        self.job_steps[action] += 1
        
        # Reward: Negative operation completion time (as per README)
        reward = -float(end_time)
        
        # Check if all jobs are fully completed
        terminated = bool(np.all(self.job_steps >= self.num_machines))
        
        return self._get_state(), reward, terminated, False, {}


# ==========================================
# 2. DQN NEURAL NETWORK
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. DQN AGENT
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Networks (Policy and Target)
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay Buffer
        self.memory = []
        self.memory_capacity = 5000
        self.batch_size = 64
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def save(self, filename="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        
    def load(self, filename="dqn_model.pth"):
        if os.path.exists(filename):
            self.policy_net.load_state_dict(torch.load(filename))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return True
        return False

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        # Compute loss and update
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ==========================================
# 4. EXECUTION WRAPPERS FOR DASHBOARD
# ==========================================
def train_agent(env, agent, episodes=400, max_steps=50, target_update_freq=10, progress_callback=None):
    rewards_history = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        agent.decay_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        
        # Callback for real-time dashboard updates
        if progress_callback:
            progress_callback(episode, total_reward)
            
        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode} | Total Reward: {total_reward:.2f}")
            
    return rewards_history

def test_agent(env, agent, max_steps=100, random_baseline=False):
    state, _ = env.reset()
    done = False
    step_count = 0
    if not random_baseline:
        agent.epsilon = 0.0  # Turn off exploration
    
    schedule = []
    action_log = []
    
    while not done and step_count < max_steps:
        if random_baseline:
            valid_actions = [a for a in range(env.num_jobs) if env.job_steps[a] < env.num_machines]
            action = random.choice(valid_actions) if valid_actions else 0
        else:
            action = agent.act(state)
        # Capture the machine index and start time before taking the step
        machine_idx = env.job_steps[action] if env.job_steps[action] < env.num_machines else -1
        start_time = env.machine_times[machine_idx] if machine_idx != -1 else 0
        
        state, reward, done, _, _ = env.step(action)
        
        if reward > -100.0 and machine_idx != -1:
            duration = env.processing_times[action, machine_idx]
            end_time = start_time + duration
            schedule.append({
                "job": int(action),
                "machine": int(machine_idx),
                "start": float(start_time),
                "end": float(end_time)
            })
            prefix = "[DUMB AI]" if random_baseline else "✅ [SMART AI]"
            action_log.append(f"{prefix} Scheduled Job {action} on Machine {machine_idx} (Duration: {duration})")
        else:
            prefix = "[DUMB AI]" if random_baseline else "❌ [SMART AI]"
            action_log.append(f"{prefix} Invalid Move: Attempted Job {action} (Penalized)")
            
        step_count += 1
        
    return schedule, action_log

def main():
    print("=== DRL JOB SHOP SCHEDULING ===")
    env = JobShopEnv(data_file="factory_data.csv")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    print("\n--- Starting Training Phase ---")
    train_agent(env, agent)
    
    print("\n--- Testing Trained Policy ---")
    schedule = test_agent(env, agent)
    for task in schedule:
        print(f"Scheduled Job: {task['job']} on Machine {task['machine']} (Time: {task['start']} to {task['end']})")
    
    print("\n✅ Run Completed Successfully.")

if __name__ == "__main__":
    main()
