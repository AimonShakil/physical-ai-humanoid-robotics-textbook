---
sidebar_position: 6
title: "Chapter 5: Reinforcement Learning for Robot Control"
---

# Reinforcement Learning for Robot Control

## The Reinforcement Learning Framework

Reinforcement Learning (RL) trains agents to make sequences of decisions that maximize cumulative reward. A robot learns by trial-and-error: take action → observe result → update policy → repeat.

### The MDP Formulation

```
State (s) --[Action a]--> Environment --> Reward (r), Next State (s')
Policy π(a|s): "Given state s, what action should I take?"
```

The goal: find a policy π that maximizes expected return:

```
J(π) = E[∑ γ^t r_t]  where γ is discount factor (usually 0.99)
```

## Policy Gradient Methods

Rather than learning the value of each state (value-based RL), policy gradient methods directly optimize the policy parameters.

### REINFORCE: Vanilla Policy Gradient

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RobotPolicyNetwork(nn.Module):
    """Simple policy network for robot control."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """
        Args:
            state: (B, state_dim)
        Returns:
            action_logits: (B, action_dim)
        """
        return self.net(state)

    def sample_action(self, state):
        """Sample action from policy."""
        logits = self.forward(state)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1)
        return action.squeeze()

class REINFORCEAgent:
    """Basic REINFORCE (vanilla policy gradient) agent."""

    def __init__(self, state_dim, action_dim, lr=0.01):
        self.policy = RobotPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(self, states, actions, rewards):
        """
        Args:
            states: List of states from episode
            actions: List of actions taken
            rewards: List of rewards received
        """
        # Compute returns (cumulative discounted rewards)
        returns = []
        cumsum = 0
        for r in reversed(rewards):
            cumsum = r + 0.99 * cumsum
            returns.insert(0, cumsum)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # Compute policy loss
        logits = self.policy(states)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]

        # Loss: negative expected return
        loss = -(selected_log_probs * returns).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Usage
agent = REINFORCEAgent(state_dim=4, action_dim=2)
# In training loop:
# episode_states, episode_actions, episode_rewards = collect_episode()
# loss = agent.train_step(episode_states, episode_actions, episode_rewards)
```

## Actor-Critic Methods

Actor-Critic combines two networks:
- **Actor**: Policy π(a|s) that chooses actions
- **Critic**: Value function V(s) that estimates expected return

The critic helps reduce variance in gradient estimates, making learning more stable.

### A3C (Asynchronous Advantage Actor-Critic)

```python
class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared features."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

class ActorCriticAgent:
    """A2C agent for robot control."""

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

    def train_step(self, states, actions, rewards):
        """
        Args:
            states: Tensor (T, state_dim)
            actions: Tensor (T,)
            rewards: Tensor (T,)
        """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Get policy logits and values
        logits, values = self.network(states)

        # Compute returns and advantages
        returns = []
        cumsum = 0
        for r in reversed(rewards):
            cumsum = r + self.gamma * cumsum
            returns.insert(0, cumsum)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - values.squeeze().detach()

        # Actor loss (policy gradient)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        actor_loss = -(selected_log_probs * advantages).mean()

        # Critic loss (value function)
        critic_loss = (values.squeeze() - returns).pow(2).mean()

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

## PPO (Proximal Policy Optimization)

**PPO** is state-of-the-art for robotics. It uses a clipped objective to prevent destructively large policy updates:

```python
class PPOAgent:
    """PPO agent with clipped surrogate objective."""

    def __init__(self, state_dim, action_dim, lr=3e-4, clip_ratio=0.2):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.clip_ratio = clip_ratio

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE).
        Smooth estimate of advantages with reduced variance.
        """
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def train_step(self, states, actions, rewards, old_log_probs, num_epochs=5):
        """
        Args:
            states, actions, rewards: Experience from rollout
            old_log_probs: Log probabilities from old policy
            num_epochs: Number of PPO update iterations
        """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

        _, values = self.network(states)
        advantages = self.compute_gae(rewards, values.squeeze().detach())

        # Returns for critic training
        returns = advantages + values.squeeze().detach()

        for epoch in range(num_epochs):
            logits, new_values = self.network(states)
            new_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs[range(len(actions)), actions]

            # Probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = (new_values.squeeze() - returns).pow(2).mean()

            total_loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        return total_loss.item()
```

## RL for Visual Control

Robots often learn policies directly from high-dimensional visual observations. This requires careful architecture design:

```python
class VisualPolicyNetwork(nn.Module):
    """End-to-end visuomotor policy."""

    def __init__(self, image_channels=3, action_dim=4, hidden_dim=256):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # MLP policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            action_logits, value
        """
        features = self.cnn(image)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value
```

## The Challenge of Exploration

In robotics, random exploration is risky (robot might break itself). Smart exploration strategies are essential:

- **Curiosity-Driven Exploration**: Reward the agent for visiting novel states
- **Uncertainty Estimation**: Use ensemble methods or dropout to estimate state visitation uncertainty
- **Demonstrations**: Warm-start with imitation learning before RL fine-tuning

## Sample Efficiency and Reality Gap

Key challenges in applying RL to real robots:

1. **Sample Efficiency**: Real robot experiments are expensive. RL typically needs millions of interactions.
2. **Safety**: During exploration, robots may collide or damage equipment.
3. **Sim-to-Real Gap**: Policies trained in simulation often fail on real hardware.

**Solutions**:
- Use demonstrations to initialize policy (imitation learning)
- Train extensively in simulation, then fine-tune on real robots
- Learn task representations that transfer across environments

## Exercises

**Exercise 5.1**: Implement a PPO agent for a simulated robot arm task (e.g., reaching, pushing). Compare learning curves with REINFORCE and A2C.

**Exercise 5.2**: Train a visuomotor policy (CNN + policy head) on a simulated grasping task. Visualize learned attention maps to understand what the policy focuses on.

**Exercise 5.3**: Design a curiosity-driven exploration bonus and incorporate it into your PPO agent. Does it improve sample efficiency compared to standard reward-only training?

---

**Next Chapter**: [Chapter 6: Imitation Learning and Behavior Cloning](./chapter6-imitation-learning.md)
