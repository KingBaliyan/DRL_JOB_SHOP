🧠 Job Shop Scheduling using Deep Reinforcement Learning

Final Year Project

📌 Overview

This project implements a Job Shop Scheduling Problem (JSSP) solver using Deep Reinforcement Learning (DRL).
The goal is to minimize the makespan (total completion time) by learning an optimal job scheduling policy through interaction with a simulated job shop environment.

Traditional scheduling approaches rely on static heuristics, whereas this project demonstrates how Deep Q-Learning (DQN) can automatically learn efficient scheduling strategies.

🎓 Academic Context

Project Type: Final Year Project

Domain: Artificial Intelligence / Machine Learning

Technique Used: Deep Reinforcement Learning (DQN)

Application Area: Manufacturing & Production Scheduling

🏗️ Problem Statement

The Job Shop Scheduling Problem is an NP-Hard optimization problem where multiple jobs, each consisting of ordered operations, must be processed on specific machines.
The objective is to schedule these operations such that the overall completion time (makespan) is minimized while respecting machine and job constraints.

🚀 Solution Approach

The scheduling problem is formulated as a Reinforcement Learning environment:

RL Component	Description
Agent	Scheduler
State	Current operation index of each job
Action	Select next job to schedule
Reward	Negative operation completion time
Policy	Learned scheduling strategy

A Deep Q-Network (DQN) is used to approximate the optimal action-value function.

🧩 System Architecture
+-------------------+
| Job Shop Env      |
|-------------------|
| State Generator   |
| Reward Function   |
| Constraint Logic  |
+--------▲----------+
         |
         |
+--------+----------+
| DRL Agent (DQN)   |
|------------------|
| Policy Network   |
| Target Network   |
| Replay Buffer    |
+------------------+
🛠️ Technologies Used

Python 3

PyTorch

OpenAI Gym

NumPy

Matplotlib (optional)

📂 Project Structure
drl_jobshop/
│
├── jobshop_drl.py        # Main implementation
├── README.md             # Project documentation
└── requirements.txt      # Dependencies (optional)
▶️ How to Run
1️⃣ Clone Repository
git clone https://github.com/your-username/drl-jobshop-scheduling.git
cd drl-jobshop-scheduling
2️⃣ Install Dependencies
pip install numpy torch gym matplotlib
3️⃣ Run the Project
python jobshop_drl.py
📈 Output Explanation

Training Phase:
Displays episode-wise total reward.
Less negative reward indicates improved scheduling performance.

Testing Phase:
Prints the sequence of scheduled jobs representing the learned policy.

📊 Example Output
Episode 0 | Total Reward: -130
Episode 100 | Total Reward: -85
Episode 300 | Total Reward: -60

--- Testing Trained Policy ---
Scheduled Job: 1
Scheduled Job: 0
Scheduled Job: 2
...
✅ Results

Successful reduction in makespan over training episodes

Learned scheduling strategy outperforms random scheduling

Demonstrates feasibility of DRL for NP-Hard optimization problems

⚠️ Limitations

Fixed processing times

No machine breakdowns or job priorities

Scalability limited for large job sets

🔮 Future Scope

PPO / A2C implementation

Graph Neural Networks for state representation

Real benchmark datasets

Gantt chart visualization

Multi-objective optimization

📚 References

Mnih et al., Human-level control through deep reinforcement learning

Pinedo, Scheduling: Theory, Algorithms, and Systems

OpenAI Gym Documentation

👨‍🎓 Author

Satya Baliyan 
Final year student - Computer science/ AI & ML
Sunny Choubey
Final Year Student – Computer Science / AI & ML

📜 License

This project is developed for academic purposes and is free to use for learning and research.


