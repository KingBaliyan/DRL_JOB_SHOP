import unittest
import numpy as np
import torch
import os
from jobshop_drl import JobShopEnv, DQNAgent

class TestJobShopDRL(unittest.TestCase):

    def setUp(self):
        # Create a tiny dummy CSV for deterministic testing
        self.test_csv = "test_factory_data.csv"
        with open(self.test_csv, "w") as f:
            f.write("Machine_0,Machine_1\n")
            f.write("5,2\n")   # Job 0 processing times
            f.write("1,4\n")   # Job 1 processing times
            
        self.env = JobShopEnv(data_file=self.test_csv, breakdown_prob=0.0)

    def tearDown(self):
        # Clean up the dummy file after tests
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
            
    def test_env_initialization(self):
        """Test if the environment dynamically sizes itself correctly based on CSV."""
        self.assertEqual(self.env.num_jobs, 2, "Should detect 2 jobs")
        self.assertEqual(self.env.num_machines, 2, "Should detect 2 machines")
        self.assertEqual(self.env.action_space.n, 2, "Action space should match job count")
        self.assertEqual(self.env.observation_space.shape[0], 4, "State space should be jobs + machines")

    def test_env_step_valid_action(self):
        """Test that a valid move advances the machine time and job step."""
        self.env.reset()
        
        # Schedule Job 0. It should go to Machine 0 and take 5 time units.
        state, reward, done, _, _ = self.env.step(0)
        
        self.assertEqual(reward, -5, "Reward should be the negative duration (-5)")
        self.assertFalse(done, "Environment should not be done yet")
        self.assertEqual(self.env.job_steps[0], 1, "Job 0 should move to step 1")
        self.assertEqual(self.env.machine_times[0], 5, "Machine 0 time should be 5")

    def test_env_step_invalid_action(self):
        """Test that the environment severely penalizes invalid moves."""
        self.env.reset()
        
        # Finish job 0 entirely
        self.env.step(0) # M0
        self.env.step(0) # M1
        
        # Now try to schedule Job 0 again (which is already fully complete)
        state, reward, done, _, _ = self.env.step(0)
        
        self.assertEqual(reward, -100, "Should apply a -100 penalty for an invalid move")
        
    def test_dqn_agent_initialization(self):
        """Test that the Deep Q-Network initializes perfectly."""
        agent = DQNAgent(state_size=4, action_size=2)
        self.assertIsNotNone(agent.policy_net, "Neural Network failed to initialize")
        self.assertEqual(agent.epsilon, 1.0, "Exploration rate should start at 1.0")
        
    def test_dqn_agent_memory(self):
        """Test the Replay Buffer memory system of the Agent."""
        agent = DQNAgent(state_size=4, action_size=2)
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        next_state = np.array([1, 0, 5, 0], dtype=np.float32)
        
        agent.remember(state, action=0, reward=-5, next_state=next_state, done=False)
        self.assertEqual(len(agent.memory), 1, "Memory buffer should store the transition")

if __name__ == "__main__":
    unittest.main()
