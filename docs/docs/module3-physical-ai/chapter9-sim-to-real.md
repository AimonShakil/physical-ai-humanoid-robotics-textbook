---
sidebar_position: 10
title: "Chapter 9: Sim-to-Real Transfer"
---

# Sim-to-Real Transfer

## The Simulation Advantage

Training robots in the physical world is slow, expensive, and risky. A single task might require hundreds of real-world trials. Meanwhile, simulation is:

- **Fast**: Run 1000s of parallel simulations
- **Safe**: Robots fail without consequences
- **Cheap**: No hardware replacement costs
- **Repeatable**: Exactly reproduce scenarios for debugging

But simulation has a fatal flaw: **the reality gap**. A policy that dominates in PyBullet often fails completely on real hardware.

### Sources of Reality Gap

1. **Physics Simulation Errors**: Friction models, contact dynamics, softness of materials
2. **Sensor Noise**: Real cameras have blur, noise, rolling shutter; simulated observations are perfect
3. **Control Delays**: Actuators have latency; simulation assumes instant response
4. **Unmodeled Dynamics**: Wind, vibration, mechanical slop, cable stretching
5. **Domain Shift in Perception**: Textures, lighting, backgrounds differ between sim and real

## Domain Randomization

The key insight: **if you randomize enough parameters in simulation, the learned policy becomes invariant to those parameters**. A policy trained on objects of random colors, sizes, and materials generalizes better to new real objects.

### Code Example: Domain Randomization

```python
import random
import numpy as np
from pybullet_utils import BulletEnv

class DomainRandomizationEnv:
    """Wrapper that randomizes environment parameters."""

    def __init__(self, base_env):
        self.env = base_env

    def randomize_object_appearance(self):
        """Randomize object colors, textures, sizes."""
        for obj_id in self.env.object_ids:
            # Random color (RGB)
            color = np.random.uniform(0, 1, 3).tolist() + [1.0]  # Add alpha
            self.env.set_object_color(obj_id, color)

            # Random size (within reasonable bounds)
            scale = np.random.uniform(0.8, 1.2)
            self.env.set_object_scale(obj_id, scale)

    def randomize_physics(self):
        """Randomize physical properties."""
        # Friction coefficient
        friction = np.random.uniform(0.3, 0.9)
        for obj_id in self.env.object_ids:
            self.env.set_friction(obj_id, friction)

        # Gravity magnitude
        gravity = np.random.uniform(8.0, 10.0)
        self.env.set_gravity(0, 0, -gravity)

        # Robot joint damping
        damping = np.random.uniform(0.1, 1.0)
        for joint_id in self.env.robot_joints:
            self.env.set_joint_damping(joint_id, damping)

    def randomize_camera(self):
        """Randomize camera pose and intrinsics."""
        # Camera position
        x = np.random.uniform(-0.1, 0.1)
        y = np.random.uniform(-0.1, 0.1)
        z = np.random.uniform(0.5, 0.7)

        # Camera orientation (small rotations)
        roll = np.random.uniform(-0.1, 0.1)
        pitch = np.random.uniform(-0.1, 0.1)

        self.env.set_camera_pose([x, y, z], [roll, pitch, 0])

    def reset(self):
        """Reset environment and apply randomization."""
        self.env.reset()

        # Apply all randomizations
        self.randomize_object_appearance()
        self.randomize_physics()
        self.randomize_camera()

        return self.env.get_observation()

    def step(self, action):
        return self.env.step(action)

# Usage
base_env = BulletEnv()
randomized_env = DomainRandomizationEnv(base_env)

# Training loop
for episode in range(1000):
    obs = randomized_env.reset()  # Randomize at each reset
    for t in range(100):
        action = policy(obs)  # Trained policy
        obs, reward, done, info = randomized_env.step(action)
        if done:
            break
```

**Result**: A policy trained on 1000 randomized simulations often transfers to real robots with minimal fine-tuning.

## Adversarial Domain Randomization

Rather than uniform random sampling, **adversarial randomization** searches for the most challenging parameter values—exposing the policy's weaknesses:

```python
class AdversarialDomainRandomization:
    """Learn randomization parameters that maximize training difficulty."""

    def __init__(self, base_env, policy):
        self.env = base_env
        self.policy = policy
        self.randomizer_params = {
            'friction': [0.5],
            'gravity': [9.81],
            'object_size': [1.0]
        }

    def compute_failure_rate(self, params):
        """Evaluate policy under specific randomization."""
        success_count = 0
        num_trials = 10

        for _ in range(num_trials):
            self.set_randomization_params(params)
            obs = self.env.reset()
            for t in range(100):
                action = self.policy(obs)
                obs, reward, done, info = self.env.step(action)
                if reward > 0.9:  # Success threshold
                    success_count += 1
                    break

        return (num_trials - success_count) / num_trials  # Failure rate

    def set_randomization_params(self, params):
        """Apply parameters to environment."""
        self.env.set_friction(params['friction'][0])
        self.env.set_gravity(0, 0, -params['gravity'][0])

    def optimize_adversarial_params(self, num_iterations=50):
        """Find hardest randomization parameters."""
        from scipy.optimize import minimize

        def objective(param_vector):
            params = {
                'friction': [param_vector[0]],
                'gravity': [param_vector[1]],
                'object_size': [param_vector[2]]
            }
            return -self.compute_failure_rate(params)  # Negative for minimization

        result = minimize(
            objective,
            x0=[0.5, 9.81, 1.0],
            bounds=[(0.1, 1.0), (5.0, 15.0), (0.5, 2.0)],
            method='L-BFGS-B'
        )

        self.randomizer_params = {
            'friction': [result.x[0]],
            'gravity': [result.x[1]],
            'object_size': [result.x[2]]
        }

        return result
```

## Progressive Domain Randomization

Start with small randomization and gradually increase it:

```python
class ProgressiveDomainRandomization:
    """Gradually increase randomization as policy improves."""

    def __init__(self, base_env, policy):
        self.env = base_env
        self.policy = policy
        self.randomization_scale = 0.1  # Start small

    def step_training(self, num_episodes=100):
        """Train for num_episodes, then increase randomization."""
        for episode in range(num_episodes):
            obs = self.env.reset()

            # Apply scaled randomization
            friction = 0.5 * (1.0 + np.random.uniform(-1, 1) * self.randomization_scale)
            self.env.set_friction(friction)

            for t in range(100):
                action = self.policy(obs)
                obs, reward, done, info = self.env.step(action)
                if done:
                    break

        # Increase randomization after episodes
        self.randomization_scale = min(1.0, self.randomization_scale * 1.1)
        print(f"Randomization scale: {self.randomization_scale:.3f}")
```

## Physics Simulation Accuracy

More accurate simulation reduces the reality gap:

- **MuJoCo**: Industry standard, fast and stable
- **PyBullet**: Free, good for quick prototyping
- **Gazebo**: ROS integration, complex but configurable
- **IsaacGym/Sim**: NVIDIA physics with GPU acceleration, millions of parallel environments

### Example: Using MuJoCo for High-Fidelity Simulation

```python
import mujoco
import mujoco.viewer

class MuJoCoEnv:
    """Environment using MuJoCo physics engine."""

    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

    def step(self, action):
        """Execute action for one timestep."""
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self.data.qpos.copy()  # Joint positions
        return obs

    def render(self):
        self.renderer.update_scene(self.data)
        return self.renderer.render()

# Usage
env = MuJoCoEnv("robot.xml")
for t in range(1000):
    action = policy(obs)
    obs = env.step(action)
```

## Transfer Learning Strategies

### 1. Fine-Tuning on Real Data

Train in simulation, then fine-tune on limited real robot data:

```python
class TransferLearningAgent:
    """Fine-tune sim-trained policy on real data."""

    def __init__(self, pretrained_policy):
        self.policy = pretrained_policy
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-5  # Small learning rate to avoid forgetting
        )

    def fine_tune_on_real_data(self, real_demonstrations, num_epochs=5):
        """
        Args:
            real_demonstrations: List of (observation, action) pairs from real robot
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for obs, action in real_demonstrations:
                pred_action = self.policy(obs)
                loss = torch.nn.functional.mse_loss(pred_action, action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(real_demonstrations):.4f}")
```

### 2. Domain Adaptation

Use adversarial training to learn a domain-invariant representation:

```python
class DomainAdaptivePolicy(nn.Module):
    """Learn representations that work in both sim and real."""

    def __init__(self, feature_dim=64):
        super().__init__()

        # Feature extractor (domain-invariant)
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Action predictor
        self.action_head = nn.Linear(feature_dim, 4)

        # Domain classifier (adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, observation):
        features = self.feature_extractor(observation)
        action = self.action_head(features)
        return action, features

    def get_domain_logit(self, features):
        """Domain prediction (0=sim, 1=real)."""
        return self.domain_classifier(features)

# Training: alternately minimize action loss and domain classification loss
def train_domain_adaptive_policy(policy, sim_data, real_data, num_epochs=100):
    action_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        # Sim data
        sim_obs, sim_actions = sim_data
        pred_actions, sim_features = policy(sim_obs)
        action_loss = action_loss_fn(pred_actions, sim_actions)

        # Domain loss: make features indistinguishable between sim and real
        real_obs, _ = real_data
        _, real_features = policy(real_obs)

        sim_domain_logits = policy.get_domain_logit(sim_features)
        real_domain_logits = policy.get_domain_logit(real_features)

        domain_loss = (
            domain_loss_fn(sim_domain_logits, torch.zeros_like(sim_domain_logits)) +
            domain_loss_fn(real_domain_logits, torch.ones_like(real_domain_logits))
        ) / 2

        # Adversarial: minimize domain classification while maximizing for features
        total_loss = action_loss - 0.1 * domain_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Action loss={action_loss:.4f}, Domain loss={domain_loss:.4f}")
```

### 3. Randomized-to-Real (RtR)

A middle ground: train on highly randomized sim, then fine-tune on real:

```
Phase 1: Train extensively in randomized simulation (1000s of episodes)
         ↓
Phase 2: Collect small amount of real data (10-100 episodes)
         ↓
Phase 3: Fine-tune on real data with low learning rate
         ↓
Phase 4: Deploy and monitor performance
```

## Safety During Transfer

Real robot experiments are risky. Mitigation strategies:

```python
class SafeRealRobotTransfer:
    """Execute sim-trained policy on real robot with safety checks."""

    def __init__(self, policy, robot, safety_monitor):
        self.policy = policy
        self.robot = robot
        self.safety = safety_monitor

    def safe_execute(self, observation, action_limit=1.0):
        """
        Execute action with safety constraints.
        """
        # Get action from policy
        action = self.policy(observation)

        # Clip to safe bounds
        action = torch.clamp(action, -action_limit, action_limit)

        # Check for dangerous conditions
        if self.safety.is_unsafe(observation, action):
            print("Unsafe action detected, stopping robot")
            return self.robot.stop()

        # Execute with monitoring
        try:
            self.robot.execute_action(action)
        except Exception as e:
            print(f"Execution failed: {e}")
            self.robot.emergency_stop()

    def collect_safe_trajectories(self, num_episodes=10):
        """Collect real data while maintaining safety."""
        real_data = []

        for episode in range(num_episodes):
            obs = self.robot.reset()
            for t in range(100):
                if not self.safety.is_observation_normal(obs):
                    self.robot.emergency_stop()
                    break

                self.safe_execute(obs)
                obs = self.robot.get_observation()
                real_data.append(obs)

        return real_data
```

## Exercises

**Exercise 9.1**: Implement domain randomization for a simulated grasping task. Train a policy with and without randomization. Evaluate transfer by testing on held-out randomization parameters.

**Exercise 9.2**: Implement adversarial domain randomization. Identify which parameters are most critical for sim-to-real transfer in your task.

**Exercise 9.3**: Simulate a sim-to-real workflow: train extensively in simulation with domain randomization, then fine-tune on 20 real demonstrations. How much real data is needed for good performance?

---

**Next Chapter**: [Chapter 10: Building Production Physical AI Systems](./chapter10-production-systems.md)
