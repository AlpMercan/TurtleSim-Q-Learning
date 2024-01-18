# Q_learning.py
import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.5,
        exploration_rate=1.0,
        max_exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.005,
        target_state=(70, 60),
    ):
        self.states = [
            (x, y) for x in range(-200, 201) for y in range(-200, 201)
        ]  # Example grid
        self.actions = ["up", "down", "left", "right"]
        self.target_state = target_state

        # Initialize Q-table
        self.q_table = {
            state: {action: 0 for action in self.actions} for state in self.states
        }

        # Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

    def get_reward(self, state):
        """Returns the reward for the current state"""
        return 100 if state == self.target_state else -1

    def update_q_table(self, state, new_state, action, reward):
        """Updates the Q-table based on the action taken and reward received"""
        max_future_q = max(self.q_table[new_state].values())
        current_q = self.q_table[state][action]

        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q
        )
        self.q_table[state][action] = new_q

    def choose_action(self, state):
        """Choose an action based on the current state"""
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        return action

    def simulate_environment(self, state, action):
        """Simulate the environment (simplified example)"""
        x, y = state
        if action == "up":
            y += 10
        elif action == "down":
            y -= 10
        elif action == "left":
            x -= 10
        elif action == "right":
            x += 10
        new_state = (x, y)
        reward = self.get_reward(new_state)
        return new_state, reward

    def train(self):
        """Training loop"""
        for episode in range(1000):
            state = (0, 0)  # Starting position

            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward = self.simulate_environment(state, action)
                self.update_q_table(state, new_state, action, reward)
                state = new_state

                if state == self.target_state:
                    done = True

            # Update exploration rate
            self.exploration_rate = self.min_exploration_rate + (
                self.max_exploration_rate - self.min_exploration_rate
            ) * np.exp(-self.exploration_decay_rate * episode)

        print("Training complete")


# Example usage
if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train()
