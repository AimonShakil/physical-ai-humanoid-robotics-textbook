---
sidebar_position: 7
title: "Chapter 6: Imitation Learning and Behavior Cloning"
---

# Imitation Learning and Behavior Cloning

## Why Learn from Demonstrations?

Reinforcement learning is sample-inefficient: robots must explore and fail to learn. In contrast, **imitation learning** leverages expert demonstrations to jumpstart learning. A robot watches skilled human operators (or other robots) and learns to replicate their behavior.

Applications:
- **Manipulation**: Learning complex multi-step tasks (assembly, surgery, cooking)
- **Navigation**: Learning to navigate from human demonstrations
- **Humanoid Control**: Imitating human movement patterns
- **Few-Shot Learning**: One or few demonstrations can bootstrap substantial capability

## Behavior Cloning: The Simplest Approach

**Behavior Cloning** frames imitation as a supervised learning problem:

```
Expert demonstration: (state, action) pairs
Task: Learn a policy π(a|s) to predict expert actions from states
```

It's simple but powerful. A neural network trained on expert trajectories can generalize to new states.

### Basic Behavior Cloning

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BehaviorCloningPolicy(nn.Module):
    """Imitate expert behavior via supervised learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """Predict action for given state."""
        return self.net(state)

class BehaviorCloningAgent:
    """Train policy via imitation learning."""

    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy = BehaviorCloningPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, expert_states, expert_actions, num_epochs=100, batch_size=32):
        """
        Args:
            expert_states: (N, state_dim) - expert observations
            expert_actions: (N, action_dim) - expert actions
            num_epochs: Training iterations
            batch_size: Batch size for training
        """
        # Create dataset
        dataset = TensorDataset(expert_states, expert_actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_states, batch_actions in loader:
                # Forward pass
                predicted_actions = self.policy(batch_states)

                # Loss: MSE between predicted and expert actions
                loss = self.loss_fn(predicted_actions, batch_actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def act(self, state):
        """Take action using learned policy."""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.policy(state)
        return action.squeeze().numpy()

# Usage
agent = BehaviorCloningAgent(state_dim=10, action_dim=3)

# Simulate expert data (in practice, collect from human demonstrations)
import numpy as np
expert_states = torch.randn(1000, 10)
expert_actions = torch.randn(1000, 3)

# Train
agent.train(expert_states, expert_actions, num_epochs=50)

# Test
test_state = np.random.randn(10)
predicted_action = agent.act(test_state)
print(f"Predicted action: {predicted_action}")
```

## The Distribution Shift Problem

A critical issue with behavior cloning: the **covariate shift**. The policy is trained on states it sees in expert demonstrations. But during rollout, the policy makes slightly different decisions, encountering different states than the expert saw.

Over time, this drift accumulates:
```
Expert sees state s0 --[expert action a0]--> s1
Clone sees state s0 --[slightly different a0']--> s1' (different from s1)
Clone has never trained on s1', so prediction is poor
Errors compound: s1' --> s2' --> s3' --> ... divergence grows
```

### DAgger (Dataset Aggregation): Addressing Distribution Shift

```python
class DAggerAgent:
    """DAgger: Interactively improve imitation learning."""

    def __init__(self, state_dim, action_dim, expert_fn, lr=1e-3):
        """
        Args:
            expert_fn: Callable that returns expert action for state
        """
        self.policy = BehaviorCloningPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.expert_fn = expert_fn
        self.loss_fn = nn.MSELoss()

    def dagger_iteration(self, num_rollouts=10, rollout_length=100):
        """
        One iteration of DAgger: aggregate expert labels on policy-visited states.
        """
        collected_states = []
        collected_actions = []

        # Step 1: Rollout learned policy, collect states
        for _ in range(num_rollouts):
            state = torch.randn(1, 10)  # Random initial state (mock)
            for _ in range(rollout_length):
                with torch.no_grad():
                    action = self.policy(state)

                collected_states.append(state.squeeze().numpy())

                # Get expert label for this state
                expert_action = self.expert_fn(state.squeeze().numpy())
                collected_actions.append(expert_action)

                # Mock environment step (normally would step real environment)
                state = state + 0.01 * torch.randn_like(state)  # Add noise

        # Step 2: Train on collected (state, expert_action) pairs
        collected_states = torch.tensor(collected_states, dtype=torch.float32)
        collected_actions = torch.tensor(collected_actions, dtype=torch.float32)

        dataset = TensorDataset(collected_states, collected_actions)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for batch_states, batch_actions in loader:
            predicted = self.policy(batch_states)
            loss = self.loss_fn(predicted, batch_actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def train(self, initial_expert_data, num_dagger_iters=10):
        """
        Args:
            initial_expert_data: (states, actions) tuple of initial expert trajectories
        """
        # Initial BC training
        expert_states, expert_actions = initial_expert_data
        dataset = TensorDataset(expert_states, expert_actions)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for _ in range(20):  # Initial BC warmup
            for batch_states, batch_actions in loader:
                predicted = self.policy(batch_states)
                loss = self.loss_fn(predicted, batch_actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # DAgger iterations
        for i in range(num_dagger_iters):
            loss = self.dagger_iteration()
            print(f"DAgger iteration {i+1}/{num_dagger_iters}, Loss: {loss:.4f}")
```

**Key insight**: By iteratively collecting states where the learned policy diverges from the expert, and asking the expert to label those states, DAgger avoids catastrophic compounding of errors.

## Multi-Modal Behavior Learning

Humans often demonstrate multiple ways to solve a task. A robot should learn this diversity, not collapse to a single averaged behavior.

### Mixture of Experts for Multi-Modal Imitation

```python
class MultiModalImitationPolicy(nn.Module):
    """Learn multiple behavior modes from demonstrations."""

    def __init__(self, state_dim, action_dim, num_modes=5, hidden_dim=256):
        super().__init__()
        self.num_modes = num_modes

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Mode selection head (predicts which behavior mode)
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes)
        )

        # Per-mode action prediction heads
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            for _ in range(num_modes)
        ])

    def forward(self, state, mode=None):
        """
        Args:
            state: (B, state_dim)
            mode: If None, sample mode; else use specified mode
        Returns:
            action: (B, action_dim)
            mode_logits: (B, num_modes)
        """
        encoded = self.encoder(state)
        mode_logits = self.mode_selector(encoded)

        if mode is None:
            mode = torch.argmax(mode_logits, dim=1)  # Greedy selection

        # Collect actions from selected modes
        actions = []
        for b in range(state.size(0)):
            head = self.action_heads[mode[b]]
            action = head(encoded[b:b+1])
            actions.append(action)

        actions = torch.cat(actions, dim=0)
        return actions, mode_logits

    def get_mode_distribution(self, state):
        """Get softmax over modes (interpretability)."""
        encoded = self.encoder(state)
        mode_logits = self.mode_selector(encoded)
        return torch.softmax(mode_logits, dim=1)

# Usage
policy = MultiModalImitationPolicy(state_dim=10, action_dim=3, num_modes=5)
state = torch.randn(4, 10)
action, mode_logits = policy(state)
print(f"Action shape: {action.shape}, Mode logits shape: {mode_logits.shape}")
```

## Imitation Learning with Vision

For vision-based tasks, behavior cloning directly learns visuomotor policies:

```python
class VisualBehaviorCloningPolicy(nn.Module):
    """Imitate expert from images."""

    def __init__(self, action_dim=4, hidden_dim=256):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Action prediction
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W)
        Returns:
            action: (B, action_dim)
        """
        features = self.cnn(image)
        action = self.mlp(features)
        return action
```

## Inverse Models and Self-Supervised Learning

An **inverse model** predicts actions from state transitions: given `(s_t, s_{t+1})`, predict `a_t`. This can be trained without expert labels, using any collected experience.

```python
class InverseModel(nn.Module):
    """Learn action prediction from state transitions."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # Concat states
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, next_state):
        """
        Args:
            state, next_state: (B, state_dim)
        Returns:
            predicted_action: (B, action_dim)
        """
        combined = torch.cat([state, next_state], dim=1)
        return self.net(combined)
```

Using inverse models, robots can learn action representations without explicit labels—useful for self-supervised pretraining.

## Exercises

**Exercise 6.1**: Implement behavior cloning on a classic control task (CartPole, Reacher). Train on expert demonstrations collected via a pre-trained RL policy. How many expert trajectories are needed for good performance?

**Exercise 6.2**: Implement DAgger and compare against standard behavior cloning. Plot learning curves and discuss how DAgger prevents distribution shift.

**Exercise 6.3**: Collect a multi-modal demonstration set (human performing same task in 3+ different ways). Train a multi-modal policy and verify it learns mode diversity.

---

**Next Chapter**: [Chapter 7: Language Models for Robotics](./chapter7-llms-robotics.md)
