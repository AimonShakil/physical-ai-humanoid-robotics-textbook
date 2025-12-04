---
sidebar_position: 9
title: "Chapter 8: Multi-Modal Robot AI Systems"
---

# Multi-Modal Robot AI Systems

## Beyond Single Modalities

A complete robot system integrates multiple AI paradigms simultaneously:

- **Vision**: CNN-based perception and scene understanding
- **Language**: LLM-based planning and reasoning
- **Touch**: Tactile sensing for compliance and force feedback
- **Proprioception**: Joint angles and end-effector state
- **Audio**: Environmental sounds and human speech
- **Reasoning**: Symbolic planning combined with neural networks

The challenge: coordinate these systems in real-time, with consistent semantics and safety guarantees.

## Hierarchical Architecture: Bridging Levels of Abstraction

Modern robot systems use hierarchical control:

```
High Level: Language understanding, task planning (LLMs)
            ↓
Mid Level:  Skill composition, state machine logic
            ↓
Low Level:  Motor control, proprioceptive feedback
```

Example: "Pick up the red cube"
1. **LLM Planning**: "Move to cube location → open gripper → close gripper → move to target location"
2. **Skill Layer**: Each step maps to a learned skill (e.g., `pick_up`, `place`)
3. **Control Layer**: Joint controllers and impedance control track desired end-effector trajectories

### Code Example: Hierarchical Robot Control

```python
import torch
import torch.nn as nn
from enum import Enum

class RobotSkill:
    """Base class for reusable robot skills."""

    def __init__(self, name):
        self.name = name
        self.is_done = False

    def reset(self):
        self.is_done = False

    def step(self, observation):
        """Execute one step of skill. Return action."""
        raise NotImplementedError

    def is_complete(self):
        return self.is_done


class MoveToSkill(RobotSkill):
    """Skill: Move end-effector to target position."""

    def __init__(self, target_position, tolerance=0.01):
        super().__init__("move_to")
        self.target = torch.tensor(target_position, dtype=torch.float32)
        self.tolerance = tolerance
        self.steps = 0

    def step(self, observation):
        """
        Args:
            observation: dict with 'ee_position' (end-effector position)
        Returns:
            action: desired joint velocities or positions
        """
        ee_pos = torch.tensor(observation['ee_position'], dtype=torch.float32)
        error = self.target - ee_pos
        distance = torch.norm(error).item()

        if distance < self.tolerance:
            self.is_done = True
            return torch.zeros(7)  # 7-DOF robot

        # Simple proportional control
        action = 0.5 * error  # Scale down for safety
        self.steps += 1

        return action.numpy()


class GraspSkill(RobotSkill):
    """Skill: Close gripper to grasp object."""

    def __init__(self, gripper_force=50):
        super().__init__("grasp")
        self.target_force = gripper_force
        self.steps = 0

    def step(self, observation):
        """Close gripper until object is grasped."""
        current_force = observation.get('gripper_force', 0)

        if current_force >= self.target_force:
            self.is_done = True

        # Gripper action: closing speed
        gripper_action = 1.0  # Close
        joint_action = torch.zeros(7)

        self.steps += 1
        return torch.cat([joint_action, torch.tensor([gripper_action])])


class ReleaseSkill(RobotSkill):
    """Skill: Open gripper."""

    def step(self, observation):
        self.is_done = True
        joint_action = torch.zeros(7)
        gripper_action = -1.0  # Open
        return torch.cat([joint_action, torch.tensor([gripper_action])])


class SkillComposer:
    """Composes high-level skills into behaviors."""

    def __init__(self):
        self.skills = {}
        self.current_skill = None
        self.skill_queue = []

    def register_skill(self, skill):
        """Register a skill."""
        self.skills[skill.name] = skill

    def queue_skills(self, skill_names):
        """Queue a sequence of skills."""
        self.skill_queue = [self.skills[name] for name in skill_names]
        self.advance_skill()

    def advance_skill(self):
        """Move to next skill in queue."""
        if self.skill_queue:
            self.current_skill = self.skill_queue.pop(0)
            self.current_skill.reset()
        else:
            self.current_skill = None

    def step(self, observation):
        """Execute current skill, advance when done."""
        if self.current_skill is None:
            return torch.zeros(8)  # 7 joints + 1 gripper

        action = self.current_skill.step(observation)

        if self.current_skill.is_complete():
            self.advance_skill()

        return action

# Usage
composer = SkillComposer()
composer.register_skill(MoveToSkill(target_position=[0.5, 0.2, 0.3]))
composer.register_skill(GraspSkill(gripper_force=50))
composer.register_skill(MoveToSkill(target_position=[0.4, 0.1, 0.5]))
composer.register_skill(ReleaseSkill())

# Queue: move to object → grasp → move to destination → release
composer.queue_skills(["move_to", "grasp", "move_to", "release"])

# Simulate execution
for t in range(100):
    observation = {
        'ee_position': [0.5, 0.2, 0.3],
        'gripper_force': 0
    }
    action = composer.step(observation)
    print(f"Step {t}: Action = {action}")
    if composer.current_skill is None:
        print("All skills completed!")
        break
```

## Multimodal Fusion at Inference Time

At runtime, a robot continuously integrates heterogeneous observations:

```python
class MultiModalRobotObserver:
    """Fuse observations from all modalities."""

    def __init__(self):
        self.vision_model = VisionModule()
        self.language_encoder = LanguageModule()
        self.proprioception_buffer = []

    def fuse_observations(self, rgb_image, depth_image, proprioception, audio_features, language_instruction):
        """
        Integrate all available information.

        Returns:
            fused_state: Unified representation for decision-making
        """
        # Visual understanding
        visual_features = self.vision_model.encode(rgb_image, depth_image)

        # Language grounding
        language_embedding = self.language_encoder.encode(language_instruction)

        # Proprioceptive state (with history for temporal context)
        self.proprioception_buffer.append(proprioception)
        if len(self.proprioception_buffer) > 10:
            self.proprioception_buffer.pop(0)
        proprioceptive_features = torch.tensor(self.proprioception_buffer).mean(dim=0)

        # Audio (e.g., detect impact or motor strain)
        audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32)

        # Combine into unified state
        fused_state = {
            'visual': visual_features,
            'language': language_embedding,
            'proprioceptive': proprioceptive_features,
            'audio': audio_features_tensor
        }

        return fused_state

class VisionModule:
    def encode(self, rgb, depth):
        # Return high-level visual understanding
        return torch.randn(256)  # Mock

class LanguageModule:
    def encode(self, text):
        # Return language embedding
        return torch.randn(256)  # Mock
```

## End-to-End Multimodal Learning

Rather than modular composition, some modern systems train **end-to-end multimodal policies** that directly map all inputs to actions:

```python
class EndToEndMultiModalPolicy(nn.Module):
    """Train single policy on all modalities jointly."""

    def __init__(self, action_dim=8, d_hidden=512):
        super().__init__()

        # Modality encoders
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_hidden)
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, d_hidden)
        )

        self.proprioception_encoder = nn.Sequential(
            nn.Linear(7, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, d_hidden)
        )

        self.language_encoder = nn.Embedding(10000, d_hidden)

        # Multimodal fusion via transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, action_dim)
        )

    def forward(self, rgb_image, depth_image, proprioception, language_token_ids):
        """
        Args:
            rgb_image: (B, 3, H, W)
            depth_image: (B, 1, H, W)
            proprioception: (B, 7)
            language_token_ids: (B, T) - language instruction tokens
        Returns:
            action: (B, action_dim)
        """
        # Encode each modality
        vision = self.vision_encoder(rgb_image).unsqueeze(1)  # (B, 1, d_hidden)
        depth = self.depth_encoder(depth_image).unsqueeze(1)  # (B, 1, d_hidden)
        prop = self.proprioception_encoder(proprioception).unsqueeze(1)  # (B, 1, d_hidden)
        lang = self.language_encoder(language_token_ids).mean(dim=1, keepdim=True)  # (B, 1, d_hidden)

        # Stack into sequence and apply transformer
        multimodal = torch.cat([vision, depth, prop, lang], dim=1)  # (B, 4, d_hidden)
        fused = self.transformer(multimodal)

        # Average pool and predict action
        pooled = fused.mean(dim=1)  # (B, d_hidden)
        action = self.action_head(pooled)  # (B, action_dim)

        return action

# Usage
policy = EndToEndMultiModalPolicy(action_dim=8)
rgb = torch.randn(4, 3, 224, 224)
depth = torch.randn(4, 1, 224, 224)
prop = torch.randn(4, 7)
lang_ids = torch.randint(0, 10000, (4, 20))
action = policy(rgb, depth, prop, lang_ids)
print(f"Action shape: {action.shape}")  # (4, 8)
```

## Attention and Interpretability

Multi-modal systems benefit from attention visualization. Understanding which modality the robot "focuses on" is crucial for safety and debugging:

```python
class InterpretableMultiModalPolicy(nn.Module):
    """Multimodal policy with attention visualization."""

    def __init__(self, action_dim=8, d_hidden=256):
        super().__init__()
        self.d_hidden = d_hidden

        # Simplified setup
        self.modality_encoders = nn.ModuleDict({
            'vision': nn.Linear(64, d_hidden),
            'proprioception': nn.Linear(7, d_hidden),
            'language': nn.Linear(512, d_hidden)
        })

        # Cross-modal attention
        self.attention = nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True)

        self.action_head = nn.Linear(d_hidden, action_dim)

    def forward(self, vision_feat, proprioception, language_feat):
        """
        Args:
            vision_feat: (B, 64)
            proprioception: (B, 7)
            language_feat: (B, 512)
        Returns:
            action: (B, action_dim)
            attention_weights: For visualization
        """
        # Encode modalities
        vision_emb = self.modality_encoders['vision'](vision_feat).unsqueeze(1)
        prop_emb = self.modality_encoders['proprioception'](proprioception).unsqueeze(1)
        lang_emb = self.modality_encoders['language'](language_feat).unsqueeze(1)

        # Concatenate into sequence
        modalities = torch.cat([vision_emb, prop_emb, lang_emb], dim=1)  # (B, 3, d_hidden)

        # Self-attention: learn which modalities to focus on
        fused, attention_weights = self.attention(modalities, modalities, modalities)

        # Use first output (could use pooling instead)
        output = fused[:, 0, :]  # (B, d_hidden)
        action = self.action_head(output)

        return action, attention_weights

    def visualize_attention(self, attention_weights, modality_names=['vision', 'proprioception', 'language']):
        """Visualize which modalities the policy attends to."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(modality_names), figsize=(12, 3))

        for i, name in enumerate(modality_names):
            axes[i].bar(modality_names, attention_weights[0, i, :].detach().cpu().numpy())
            axes[i].set_title(f"Attention to {name}")
            axes[i].set_ylim([0, 1])

        plt.tight_layout()
        return fig
```

## Real-Time Coordination

A production system must manage latency across modalities. Some observations arrive faster than others:

- Vision: 30 Hz
- Proprioception: 100+ Hz
- Language: Episodic (only updated when human gives instruction)
- Audio: 16 kHz sampling

**Asynchronous Fusion**:
```python
class AsynchronousMultiModalFusion:
    """Handle different sensor frequencies."""

    def __init__(self):
        self.latest_observations = {}
        self.last_action_time = 0

    def update_observation(self, modality, data, timestamp):
        """Asynchronously receive updates from different sensors."""
        self.latest_observations[modality] = (data, timestamp)

    def get_decision(self, current_time):
        """Make decision using latest available data."""
        # Only use observations from last 50ms (avoid stale data)
        freshness_threshold = current_time - 0.05

        observations = {}
        for modality, (data, timestamp) in self.latest_observations.items():
            if timestamp > freshness_threshold:
                observations[modality] = data

        # Make decision with available data
        # (Missing modalities use defaults or previous values)
        return observations
```

## Exercises

**Exercise 8.1**: Implement a simple hierarchical controller for a 7-DOF robot. Define 3-4 reusable skills (move, grasp, place) and compose them into a complete pick-and-place behavior.

**Exercise 8.2**: Train an end-to-end multimodal policy on a simulated manipulation task with RGB, depth, and proprioceptive inputs. Visualize attention weights to understand what the policy focuses on.

**Exercise 8.3**: Design an asynchronous fusion system for a robot with 3+ sensors operating at different frequencies. Measure latency and discuss strategies for keeping the system responsive.

---

**Next Chapter**: [Chapter 9: Sim-to-Real Transfer](./chapter9-sim-to-real.md)
