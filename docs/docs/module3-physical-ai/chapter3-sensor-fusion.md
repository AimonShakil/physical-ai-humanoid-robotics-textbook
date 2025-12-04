---
sidebar_position: 4
title: "Chapter 3: Sensor Fusion and Perception"
---

# Sensor Fusion and Perception

## Multi-Modal Sensing in Robotics

A robot's perception system integrates multiple sensor modalities to build a rich, redundant understanding of its environment:

- **Vision** (RGB, Thermal): High bandwidth, spatial detail, color/texture information
- **Depth** (LiDAR, Structured Light, Stereo): 3D structure, range, object localization
- **Proprioception** (Joint Angles, IMU): Self-awareness, balance, force feedback
- **Touch** (Pressure Sensors, Force/Torque Sensors): Contact information, compliance, texture
- **Audition** (Microphones): Environmental context, human speech

No single sensor is perfect. Cameras may fail in low light; LiDAR struggles with reflective surfaces; proprioception alone cannot see obstacles. Sensor fusion—intelligently combining heterogeneous inputs—is essential for robust perception.

## Why Multi-Modal Fusion?

1. **Redundancy**: If one sensor fails, others provide fallback information
2. **Complementarity**: Different modalities capture different physical properties
3. **Robustness**: Outliers or noise in one modality are checked against others
4. **Information Density**: Fusing modalities often yields richer representations than unimodal processing

## Classical Approaches: Kalman Filtering

Before deep learning, **Kalman filters** were the gold standard for sensor fusion. They maintain a Gaussian belief over state variables and update it using noisy measurements.

### The Kalman Filter Cycle

```
Predict: x_predicted = A * x_prev + B * u
Update:  x_posterior = x_predicted + K * (z - H * x_predicted)
```

Where:
- `x`: state estimate (e.g., position, velocity)
- `A`: state transition model
- `z`: noisy measurement
- `K`: Kalman gain (weighs measurement vs. prediction)

### Code Example: Basic Kalman Filter for Robot Localization

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    """Simple 1D Kalman filter for robot position estimation."""

    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_estimate_error=1):
        """
        Args:
            process_variance: How much the process can deviate (system noise)
            measurement_variance: Sensor noise level
            initial_value: Initial state estimate
            initial_estimate_error: Initial uncertainty
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.position = initial_value
        self.estimate_error = initial_estimate_error
        self.history = [initial_value]

    def update(self, measurement):
        """Update state with a new measurement."""
        # Predict
        self.estimate_error += self.process_variance

        # Update
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.position += kalman_gain * (measurement - self.position)
        self.estimate_error *= (1 - kalman_gain)

        self.history.append(self.position)
        return self.position

# Usage: Fusing noisy measurements of robot position
np.random.seed(42)
filter = KalmanFilter1D(process_variance=0.001, measurement_variance=0.1)

# Simulate true position moving at constant velocity
true_positions = np.arange(0, 10, 0.1)
noisy_measurements = true_positions + np.random.normal(0, 0.3, len(true_positions))

estimated_positions = []
for meas in noisy_measurements:
    est = filter.update(meas)
    estimated_positions.append(est)

print(f"Final estimated position: {estimated_positions[-1]:.2f}")
print(f"True position: {true_positions[-1]:.2f}")
```

## Deep Learning for Sensor Fusion

Modern robotics moves beyond hand-tuned filters toward **end-to-end learned fusion**. Neural networks implicitly learn how to weight and combine modalities.

### Multi-Modal Transformer Fusion

**Transformers** excel at fusing heterogeneous sequences. Each modality becomes a sequence of tokens, and attention mechanisms learn how to combine them:

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiModalSensorFusionTransformer(nn.Module):
    """Fuse vision, depth, and proprioception using a Transformer."""

    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # Modality-specific encoders (project to shared embedding space)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model)
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, d_model)
        )

        self.proprioception_encoder = nn.Sequential(
            nn.Linear(7, 128),  # 7 joint angles
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        # Transformer fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head (e.g., for task prediction)
        self.output_head = nn.Linear(d_model * 3, 10)  # 3 modalities

    def forward(self, rgb_image, depth_image, proprioception):
        """
        Args:
            rgb_image: (B, 3, H, W)
            depth_image: (B, 1, H, W)
            proprioception: (B, 7)
        Returns:
            fused_output: (B, 10)
        """
        # Encode each modality
        vision_feat = self.vision_encoder(rgb_image).unsqueeze(1)  # (B, 1, d_model)
        depth_feat = self.depth_encoder(depth_image).unsqueeze(1)  # (B, 1, d_model)
        prop_feat = self.proprioception_encoder(proprioception).unsqueeze(1)  # (B, 1, d_model)

        # Stack into sequence
        multimodal_seq = torch.cat([vision_feat, depth_feat, prop_feat], dim=1)  # (B, 3, d_model)

        # Apply transformer (learns cross-modal attention)
        fused = self.transformer(multimodal_seq)  # (B, 3, d_model)

        # Flatten and predict
        fused_flat = fused.reshape(fused.size(0), -1)
        output = self.output_head(fused_flat)

        return output

# Usage
model = MultiModalSensorFusionTransformer(d_model=128, nhead=4)
rgb = torch.randn(4, 3, 64, 64)
depth = torch.randn(4, 1, 64, 64)
proprioception = torch.randn(4, 7)
output = model(rgb, depth, proprioception)
print(f"Output shape: {output.shape}")  # (4, 10)
```

## Temporal Fusion: Recurrent Networks

Robots benefit from temporal context. A robot observing a falling object should predict where it will land based on past frames, not just the current frame.

### LSTM for Temporal Sensor Fusion

```python
class TemporalSensorFusion(nn.Module):
    """Use LSTM to fuse sensor streams over time."""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.sensor_encoder = nn.Linear(input_dim, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sensor_sequence):
        """
        Args:
            sensor_sequence: (B, T, input_dim) - T timesteps
        Returns:
            predictions: (B, output_dim)
        """
        encoded = self.sensor_encoder(sensor_sequence)  # (B, T, 64)
        lstm_out, (h_n, c_n) = self.lstm(encoded)  # lstm_out: (B, T, hidden_dim)
        # Use final hidden state
        output = self.output(h_n[-1])  # (B, output_dim)
        return output
```

## Early Fusion vs. Late Fusion

- **Early Fusion**: Combine raw sensor data before processing → captures low-level correlations
- **Late Fusion**: Process each modality independently, combine high-level features → more modular, robust to missing modalities
- **Hybrid**: Progressive fusion at multiple levels (increasingly common)

## Real-World Example: Grasp Point Detection

A robot must locate where to grasp an object. Fusing vision + depth:

1. **Vision**: RGB image → CNN → semantic features ("handle", "flat surface")
2. **Depth**: Depth map → segmentation → object center of mass, surface normals
3. **Fusion**: Combine semantic and geometric cues to score grasp points
4. **Control**: Select highest-scoring grasp and execute

## Challenges

- **Modality Asynchrony**: Sensors operate at different frequencies (LiDAR: 10 Hz, Camera: 30 Hz, IMU: 100+ Hz)
- **Missing Data**: Sensors occasionally fail or return invalid data
- **Computational Cost**: Fusing high-frequency, high-dimensional streams strains compute budgets
- **Domain Shift**: A model trained on one robot may not transfer to sensors with different noise characteristics

## Exercises

**Exercise 3.1**: Implement an extended Kalman filter (EKF) for 2D robot localization with nonlinear motion and measurement models.

**Exercise 3.2**: Train a multi-modal fusion network on a simulated dataset (e.g., using a physics simulator to generate RGB, depth, and proprioception). Evaluate how performance degrades when one modality is removed.

**Exercise 3.3**: Propose a sensor fusion architecture for a robot tasked with picking objects from a shelf. What modalities are essential? Where would you apply early vs. late fusion?

---

**Next Chapter**: [Chapter 4: Computer Vision for Robotics](./chapter4-computer-vision.md)
