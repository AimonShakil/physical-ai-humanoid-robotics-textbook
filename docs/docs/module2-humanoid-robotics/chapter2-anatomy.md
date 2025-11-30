---
sidebar_position: 3
title: Chapter 2 - Humanoid Robot Anatomy and Design
---

# Humanoid Robot Anatomy and Design

## Overview

The physical structure of a humanoid robot defines what it can and cannot do. Unlike humans, whose anatomy evolved through millions of years of natural selection, robot designs must balance multiple competing objectives: stability, dexterity, computational cost, energy efficiency, and manufacturability. This chapter explores how humanoid robots are designed, focusing on joint configurations, actuator types, sensor placement, and the tradeoffs engineers make when building these complex systems.

## Anthropomorphic Structure Overview

A humanoid robot typically mirrors human anatomy with distinct body segments:

```
           Head (cameras, microphone)
              |
          Neck (1-3 DoF)
              |
    ┌─────────────────────┐
    |       Torso         |
    │  (Spine: 1-3 DoF)   │
    └─────────────────────┘
      /              \
   Left Arm         Right Arm
   (7 DoF)          (7 DoF)
    |                |
  Wrist            Wrist
  Hand             Hand
   |                |
   /     \          /     \
  /       \        /       \
Left Leg  Right Leg  (6 DoF each + wheels/feet)
```

### Degree of Freedom (DoF) Distribution

Different platforms allocate DoF differently based on their mission:

| Platform | Head | Neck | Spine | Arm (×2) | Hand | Leg (×2) | Total |
|----------|------|------|-------|----------|------|----------|-------|
| NAO      | 2    | 1    | 0     | 5×2=10   | 0    | 5×2=10   | 25    |
| Pepper   | 2    | 1    | 2     | 5×2=10   | 2    | 2 (base) | 23    |
| Atlas    | 2    | 1    | 0     | 6×2=12   | 5×2  | 6×2=12   | 28    |
| WABOT-1  | 2    | 3    | 1     | 7×2=14   | 8×2  | 6×2=12   | 50    |

## Joint Types and Configurations

### Revolute Joints (Rotational)

Most humanoid robots use revolute joints to approximate human movement:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Revolute joint rotation around z-axis
def revolute_joint(angle_degrees, axis='z'):
    """Model revolute joint rotation"""
    angle_rad = np.radians(angle_degrees)

    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    return rotation_matrix

# Example: Hip joint rotation
hip_pitch = revolute_joint(45, axis='x')  # Forward/backward movement
print("Hip pitch rotation matrix:\n", hip_pitch)

# Shoulder joint with multiple axes
def shoulder_3dof(roll, pitch, yaw):
    """Model 3-DoF shoulder joint"""
    r_roll = revolute_joint(roll, axis='x')
    r_pitch = revolute_joint(pitch, axis='y')
    r_yaw = revolute_joint(yaw, axis='z')

    # Combined rotation: roll → pitch → yaw
    combined = r_yaw @ r_pitch @ r_roll
    return combined
```

### Prismatic Joints

Less common in humanoid robots but occasionally used for specialized applications:

$$d(t) = d_0 + \int_0^t v(\tau) d\tau$$

Where $d(t)$ is joint position and $v(t)$ is velocity.

## Actuator Types

### Electric Motors (Most Common)

Used in most modern humanoid robots:

```python
class ElectricMotor:
    """Model of electric motor for joint actuation"""

    def __init__(self, torque_max, velocity_max, efficiency=0.85):
        self.torque_max = torque_max  # N·m
        self.velocity_max = velocity_max  # rad/s
        self.efficiency = efficiency

    def power_required(self, torque, velocity):
        """Calculate instantaneous power"""
        # P = τ × ω
        mechanical_power = torque * velocity
        electrical_power = mechanical_power / self.efficiency
        return electrical_power

    def thermal_loss(self, current, resistance):
        """Calculate resistive heating"""
        # P_loss = I²R
        return current**2 * resistance

# Typical motor for NAO shoulder joint
nao_shoulder = ElectricMotor(
    torque_max=1.3,  # N·m
    velocity_max=9.5,  # rad/s
    efficiency=0.80
)

print("Max power at rated torque/speed:")
max_power = nao_shoulder.power_required(1.3, 9.5)
print(f"  {max_power:.1f} W")
```

### Hydraulic Actuators

Used in high-performance platforms like Atlas:

- **Advantages**: High power density, smooth force control
- **Disadvantages**: Complex plumbing, maintenance, noise, thermal management
- **Use case**: Atlas can perform dynamic movements requiring rapid force changes

### Pneumatic Actuators

Rarely used in modern humanoid robots due to power efficiency concerns.

## Sensor Suite

A typical humanoid robot integrates multiple sensor types:

```python
import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Sensor:
    name: str
    location: str
    sampling_rate: float  # Hz
    data_type: str
    purpose: str

# Typical NAO sensor configuration
nao_sensors = [
    Sensor("USB2 Camera (Front)", "Head", 30, "RGB Image", "Object detection"),
    Sensor("USB2 Camera (Bottom)", "Foot", 30, "RGB Image", "Ground detection"),
    Sensor("Microphone Array", "Head", 16000, "Audio", "Speech recognition"),
    Sensor("IMU", "Torso", 100, "Accel + Gyro", "Balance sensing"),
    Sensor("Force/Torque Sensors", "Foot (×2)", 50, "6-axis F/T", "Ground contact"),
    Sensor("Joint Encoders", "All joints", 100, "Angle", "Proprioception"),
    Sensor("IR Range Sensors", "Head/Body", 10, "Distance", "Obstacle detection"),
    Sensor("Pressure Sensors", "Foot sole", 50, "Pressure map", "Weight distribution"),
]

# Print sensor summary
print("NAO Sensor Suite:")
print("-" * 70)
total_data_rate = 0
for sensor in nao_sensors:
    bytes_per_sample = 4 if "RGB" in sensor.data_type else 1
    data_rate = sensor.sampling_rate * bytes_per_sample
    total_data_rate += data_rate
    print(f"{sensor.name:30} | {sensor.sampling_rate:6.0f} Hz | {sensor.purpose}")

print(f"\nEstimated total data rate: {total_data_rate/1e6:.2f} Mbps")
```

## Center of Mass and Balance

A critical design parameter is the location of the center of mass (CoM):

$$\vec{r}_{CoM} = \frac{\sum_i m_i \vec{r}_i}{\sum_i m_i}$$

For bipedal stability, the CoM projection must remain within the support polygon (feet area):

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_center_of_mass(body_parts):
    """Calculate center of mass from body segments"""
    total_mass = sum(m for _, m in body_parts)
    weighted_pos = sum(np.array(r) * m for (r, m) in body_parts)
    com = weighted_pos / total_mass
    return com

# NAO body segments (position in meters, mass in kg)
nao_segments = [
    ((0, 0, 0.15), 0.4),      # Head
    ((0, 0, 0.10), 1.3),      # Torso
    ((0.1, 0, 0.08), 0.4),    # Left arm
    ((-0.1, 0, 0.08), 0.4),   # Right arm
    ((0.05, 0, -0.1), 0.6),   # Left leg
    ((-0.05, 0, -0.1), 0.6),  # Right leg
]

com = calculate_center_of_mass(nao_segments)
print(f"NAO Center of Mass: {com}")
print(f"Height above ground: {com[2]:.3f} m")
```

## Kinematic Chain Representation

Humanoid robots use Denavit-Hartenberg (DH) parameters to describe their kinematic structure:

| Joint | θ (deg) | d (m) | a (m) | α (deg) |
|-------|---------|-------|-------|---------|
| Hip Pitch | θ₁ | 0 | 0 | -90 |
| Hip Roll | θ₂ | 0 | 0 | 0 |
| Knee Pitch | θ₃ | 0 | 0.4 | 0 |
| Ankle Pitch | θ₄ | 0 | 0.4 | 0 |
| Ankle Roll | θ₅ | 0 | 0 | 0 |

## Material Selection

Humanoid robot materials must balance multiple properties:

```python
class Material:
    def __init__(self, name, density, strength, cost_per_kg):
        self.name = name
        self.density = density  # kg/m³
        self.strength = strength  # MPa
        self.cost_per_kg = cost_per_kg  # USD

    def specific_strength(self):
        """Strength-to-weight ratio"""
        return self.strength / self.density

    def cost_per_strength_unit(self):
        """Cost-effectiveness metric"""
        return self.cost_per_kg / self.strength

materials = [
    Material("Aluminum Alloy", 2700, 310, 2.0),
    Material("Titanium Alloy", 4500, 900, 15.0),
    Material("Carbon Fiber Composite", 1600, 600, 10.0),
    Material("Steel", 7850, 400, 0.5),
]

print("Material Comparison:")
print(f"{'Material':<25} {'Specific Strength':<20} {'Cost/Strength':<15}")
print("-" * 60)
for mat in materials:
    ss = mat.specific_strength()
    cs = mat.cost_per_strength_unit()
    print(f"{mat.name:<25} {ss:>15.3f} MPa·m³/kg {cs:>12.4f} $/MPa")
```

## Exercise 2.1: Design Analysis

**Objective**: Analyze the mechanical design of a humanoid robot.

**Task**:
1. Select a humanoid robot (e.g., HRP-4, REEM-C, or Valkyrie)
2. Document its specifications:
   - DoF distribution across body segments
   - Actuator types and power specifications
   - Mass distribution
   - Height and reach envelope
3. Create a DH parameter table for at least one leg
4. Analyze center of mass location relative to the base of support
5. Discuss design tradeoffs (speed vs. stability, dexterity vs. simplicity)

**Submission**: Design report with specification table, DH parameter table, and analysis.

---

## Next Chapter Preview

In Chapter 3, we transition from static anatomy to dynamic motion. You'll learn how to mathematically describe robot motion using forward kinematics, how to compute joint velocities and accelerations, and how dynamics equations predict forces and torques required for movement.

[→ Next: Chapter 3 - Kinematics and Dynamics](/docs/module2-humanoid-robotics/chapter3-kinematics)
