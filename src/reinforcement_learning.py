"""Reinforcement Learning for maintenance scheduling."""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import RL_CONFIG

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """Q-Learning based maintenance scheduler."""

    def __init__(
        self,
        states: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
    ):
        """
        Initialize the Q-Learning maintenance scheduler.

        Args:
            states: List of possible states
            actions: List of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.states = states or RL_CONFIG["states"]
        self.actions = actions or RL_CONFIG["actions"]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table
        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in self.states
        }

        logger.info(
            f"Initialized MaintenanceScheduler with {len(self.states)} states "
            f"and {len(self.actions)} actions"
        )

    def get_state_from_rul(self, rul: float) -> str:
        """
        Map RUL value to a discrete state.

        Args:
            rul: Remaining Useful Life value

        Returns:
            Corresponding state string
        """
        if rul > 100:
            return "healthy"
        elif rul > 50:
            return "moderate_wear"
        elif rul > 20:
            return "severe_wear"
        else:
            return "failed"

    def choose_action(self, state: str, greedy: bool = False) -> str:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state
            greedy: If True, always choose best action (no exploration)

        Returns:
            Selected action
        """
        if not greedy and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(self.actions)
        else:
            # Exploit: best action from Q-table
            action = max(self.q_table[state], key=lambda a: self.q_table[state][a])

        return action

    def calculate_reward(self, state: str, action: str, next_state: str) -> float:
        """
        Calculate reward for taking an action in a state.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # Define cost structure
        maintenance_cost = -10
        failure_cost = -100
        no_action_cost = -1  # Small cost for each time step

        # Reward logic
        if next_state == "failed":
            return failure_cost

        if action == "immediate_maintenance":
            if state in ["severe_wear", "failed"]:
                return maintenance_cost + 20  # Good decision
            else:
                return maintenance_cost - 10  # Premature maintenance

        if action == "schedule_maintenance":
            if state == "moderate_wear":
                return maintenance_cost + 10  # Optimal timing
            elif state == "severe_wear":
                return maintenance_cost  # Acceptable
            else:
                return maintenance_cost - 5  # Too early

        if action == "no_maintenance":
            if state == "healthy":
                return 5  # Good, no unnecessary maintenance
            elif state == "moderate_wear":
                return no_action_cost  # Acceptable risk
            else:
                return failure_cost / 2  # Risky decision

        return no_action_cost

    def update_q_table(
        self, state: str, action: str, reward: float, next_state: str
    ) -> None:
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def train(
        self,
        training_data: pd.DataFrame,
        episodes: int = 1000,
        max_steps_per_episode: int = 100,
    ) -> Dict[str, Union[List[float], Dict[str, int]]]:
        """
        Train the Q-learning agent.

        Args:
            training_data: DataFrame with RUL values
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training Q-Learning agent for {episodes} episodes")

        rewards_per_episode = []
        actions_taken = {action: 0 for action in self.actions}

        for episode in range(episodes):
            episode_reward = 0

            # Sample random starting RUL values
            for step in range(max_steps_per_episode):
                # Sample a RUL value
                if len(training_data) > 0:
                    rul = np.random.choice(training_data["RUL"].to_numpy())
                else:
                    rul = np.random.uniform(0, 200)

                state = self.get_state_from_rul(rul)

                # Choose action
                action = self.choose_action(state)
                actions_taken[action] += 1

                # Simulate next state (degradation)
                next_rul = max(0, rul - np.random.uniform(1, 10))
                if action == "immediate_maintenance":
                    next_rul = rul + np.random.uniform(20, 50)  # Restored
                elif action == "schedule_maintenance":
                    next_rul = rul + np.random.uniform(10, 30)  # Improved

                next_state = self.get_state_from_rul(next_rul)

                # Calculate reward and update Q-table
                reward = self.calculate_reward(state, action, next_state)
                self.update_q_table(state, action, reward, next_state)

                episode_reward += reward

            rewards_per_episode.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                logger.info(f"Episode {episode + 1}: Average reward = {avg_reward:.2f}")

        logger.info("Q-Learning training complete")
        logger.info(f"Actions distribution: {actions_taken}")

        return {
            "rewards_per_episode": rewards_per_episode,
            "actions_taken": actions_taken,
        }

    def get_maintenance_schedule(
        self, predictions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate maintenance schedule based on learned policy.

        Args:
            predictions_df: DataFrame with unit_number, time_in_cycles, and RUL predictions

        Returns:
            DataFrame with recommended maintenance actions
        """
        logger.info("Generating maintenance schedule")

        schedule = []

        for _, row in predictions_df.iterrows():
            state = self.get_state_from_rul(row["RUL"])
            recommended_action = self.choose_action(state, greedy=True)

            schedule.append(
                {
                    "unit_number": row["unit_number"],
                    "time_in_cycles": row["time_in_cycles"],
                    "RUL": row["RUL"],
                    "state": state,
                    "recommended_action": recommended_action,
                }
            )

        schedule_df = pd.DataFrame(schedule)
        logger.info(f"Generated schedule for {len(schedule_df)} records")

        return schedule_df

    def get_q_table(self) -> pd.DataFrame:
        """
        Get Q-table as DataFrame for visualization.

        Returns:
            Q-table as DataFrame
        """
        return pd.DataFrame(self.q_table).T

    def save_policy(self, filepath: str) -> None:
        """
        Save the learned policy (Q-table).

        Args:
            filepath: Path to save the policy
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.q_table, f, indent=2)

        logger.info(f"Policy saved to {filepath}")

    @classmethod
    def load_policy(cls, filepath: str) -> "MaintenanceScheduler":
        """
        Load a saved policy.

        Args:
            filepath: Path to the saved policy

        Returns:
            MaintenanceScheduler with loaded policy
        """
        import json

        with open(filepath, "r") as f:
            q_table = json.load(f)

        instance = cls()
        instance.q_table = q_table
        logger.info(f"Policy loaded from {filepath}")

        return instance
