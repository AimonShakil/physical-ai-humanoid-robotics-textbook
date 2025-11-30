---
sidebar_position: 5
title: Chapter 4 - Bipedal Walking Fundamentals
---

# Bipedal Walking Fundamentals

## Overview

Bipedal walking is the defining characteristic of humanoid robots, yet it remains one of robotics' most challenging problems. Unlike quadrupeds, which have inherent stability with four contact points, bipeds must continuously balance while moving. This chapter explores gait patterns, the zero moment point (ZMP) criterion for stability, and fundamental walking control strategies. We'll examine how robots transition from one leg to the other while maintaining balance and forward momentum.

## Gait Phases and Terminology

A complete gait cycle consists of alternating stance and swing phases:

```
      Stance Phase                    Swing Phase
    (foot on ground)              (foot in air)
         |←→|←→|←→|←→|            |←→|←→|←→|←→|
    ————●————●————●————●————●————●————●————●————●————●————
    Left  R  R  R  R  R  L  L  L  L  L  R
    Double Single  Double Single  Double Single
    Support Support Support Support Support Support

    0%          50%             100%
    (Complete gait cycle)
```

### Key Phases

| Phase | Duration | Left Foot | Right Foot | Notes |
|-------|----------|-----------|------------|-------|
| Double Support | 0-10% | Stance | Stance | Both feet on ground |
| Right Single Support | 10-50% | Swing | Stance | Right foot pushes off |
| Double Support | 50-60% | Stance | Stance | Weight transfer |
| Left Single Support | 60-100% | Stance | Swing | Left foot pushes off |

```python
def gait_phase_analysis(cycle_time, gait_parameters):
    """
    Analyze gait phases during a walking cycle
    """
    import numpy as np

    # Gait parameter definition
    double_support_ratio = gait_parameters.get('double_support_ratio', 0.2)
    step_length = gait_parameters.get('step_length', 0.3)  # meters
    walking_speed = gait_parameters.get('walking_speed', 0.5)  # m/s

    # Stride time = one complete cycle
    stride_time = 2 * step_length / walking_speed

    # Phase durations
    double_support_time = double_support_ratio * stride_time
    single_support_time = (1 - double_support_ratio) * stride_time / 2

    phases = {
        'double_support_1': (0, double_support_time),
        'right_single_support': (double_support_time, double_support_time + single_support_time),
        'double_support_2': (double_support_time + single_support_time,
                             2 * double_support_time + single_support_time),
        'left_single_support': (2 * double_support_time + single_support_time, stride_time)
    }

    return stride_time, phases

# Example: NAO walking parameters
nao_gait = {
    'double_support_ratio': 0.15,
    'step_length': 0.04,  # 4 cm per step
    'walking_speed': 0.08  # m/s
}

stride_time, phases = gait_phase_analysis(1.0, nao_gait)
print(f"Stride time: {stride_time:.2f} seconds")
print(f"Walking speed: {nao_gait['walking_speed']:.2f} m/s")
print(f"\nPhase timings (seconds):")
for phase_name, (start, end) in phases.items():
    print(f"  {phase_name:25}: {start:.3f} - {end:.3f} ({(end-start)*100/stride_time:.1f}%)")
```

## Zero Moment Point (ZMP) Theory

The Zero Moment Point (ZMP) is the point on the ground where the moment (torque) is zero. For stable bipedal walking, the ZMP must remain within the convex hull of the support polygon (both feet during double support, one foot during single support).

### Mathematical Definition

The ZMP position is calculated from ground reaction forces:

```text
p_ZMP = (Σ m_i * r_i × (g + a_i)) / (Σ m_i * g)
```

Where:
- `m_i` = mass of segment i
- `r_i` = position of segment i
- a_i = acceleration of segment i
- g = gravitational acceleration

### Stability Criterion

For stable walking:

$$x_{ZMP} \in [x_{heel}, x_{toe}]$$
$$y_{ZMP} \in [y_{left edge}, y_{right edge}]$$

```python
import numpy as np
import matplotlib.pyplot as plt

class ZMPAnalyzer:
    """Compute and analyze Zero Moment Point"""

    def __init__(self, robot_mass, robot_height, foot_length, foot_width):
        self.mass = robot_mass  # kg
        self.height = robot_height  # m
        self.foot_length = foot_length  # m
        self.foot_width = foot_width  # m

    def compute_zmp(self, com_position, com_acceleration, contact_foot):
        """
        Calculate ZMP from center of mass dynamics
        com_position: (x, y, z) center of mass position
        com_acceleration: (ax, ay, az) center of mass acceleration
        contact_foot: 'left', 'right', or 'both'
        """
        g = 9.81  # m/s²
        x, y, z = com_position
        ax, ay, az = com_acceleration

        # Simplified ZMP calculation (inverted pendulum approximation)
        # zmp_x ≈ x - z * ax / g
        # zmp_y ≈ y - z * ay / g

        zmp_x = x - z * ax / g
        zmp_y = y - z * ay / g

        return np.array([zmp_x, zmp_y])

    def is_stable(self, zmp, contact_foot, foot_center):
        """
        Check if ZMP is within support polygon
        foot_center: (x, y) of foot center
        """
        # Support polygon boundaries (relative to foot center)
        x_margin = self.foot_length / 2
        y_margin = self.foot_width / 2

        x_bounds = [foot_center[0] - x_margin, foot_center[0] + x_margin]
        y_bounds = [foot_center[1] - y_margin, foot_center[1] + y_margin]

        # Check if ZMP is within bounds
        x_stable = x_bounds[0] <= zmp[0] <= x_bounds[1]
        y_stable = y_bounds[0] <= zmp[1] <= y_bounds[1]

        stability_margin = min(
            zmp[0] - x_bounds[0],
            x_bounds[1] - zmp[0],
            zmp[1] - y_bounds[0],
            y_bounds[1] - zmp[1]
        )

        return x_stable and y_stable, stability_margin

# Example: NAO ZMP analysis during walking
analyzer = ZMPAnalyzer(robot_mass=4.3, robot_height=0.58, foot_length=0.06, foot_width=0.03)

# Walking cycle with sinusoidal COM motion
time_points = np.linspace(0, 1.0, 100)
com_x = 0.15 * np.sin(2 * np.pi * time_points)  # Lateral sway
com_z = 0.29  # Constant height (inverted pendulum)
com_ax = 2 * np.pi * 0.15 * 2 * np.pi * np.cos(2 * np.pi * time_points)

stability_results = []
for t in time_points:
    idx = int(t * 100)
    com_pos = np.array([com_x[idx], 0.02 * np.sin(4*np.pi*t), com_z])
    com_acc = np.array([com_ax[idx], 0, 0])

    zmp = analyzer.compute_zmp(com_pos, com_acc, 'right')

    # Right foot at origin
    right_foot = np.array([0, -0.02])
    is_stable, margin = analyzer.is_stable(zmp, 'right', right_foot)

    stability_results.append({
        'time': t,
        'zmp': zmp,
        'stable': is_stable,
        'margin': margin
    })

# Report stability
stable_count = sum(1 for r in stability_results if r['stable'])
print(f"Stability: {stable_count}/{len(stability_results)} steps stable")
print(f"Min stability margin: {min(r['margin'] for r in stability_results):.4f} m")
```

## Gait Pattern Generation

Walking requires coordinated trajectory generation for each joint. Common approaches:

### 1. Linear Interpolation with Sinusoidal Timing

```python
def generate_sinusoidal_gait(step_length, step_height, cycle_time, num_steps):
    """
    Generate sinusoidal gait trajectories
    """
    import numpy as np

    trajectories = []
    dt = cycle_time / 100  # 100 samples per cycle

    for step in range(num_steps):
        for sample in range(100):
            # Normalized time in current cycle [0, 1]
            t_norm = sample / 100.0

            # Horizontal motion (linear)
            x = step * step_length + (step_length * t_norm)

            # Vertical motion (sinusoidal for smooth foot clearance)
            # Height = 0 at start and end, peak in middle
            z = step_height * np.sin(np.pi * t_norm)

            # Foot orientation (pitch)
            pitch = 0  # Keep foot parallel to ground

            trajectories.append({
                'time': (step * cycle_time) + (t_norm * cycle_time),
                'position': np.array([x, 0, z]),
                'orientation': pitch
            })

    return trajectories

# Generate walking trajectory
traj = generate_sinusoidal_gait(
    step_length=0.3,
    step_height=0.05,
    cycle_time=1.0,
    num_steps=5
)

print(f"Generated {len(traj)} trajectory points")
print(f"First step: x={traj[0]['position'][0]:.3f}, z={traj[0]['position'][2]:.3f}")
print(f"Peak height: {max(t['position'][2] for t in traj):.3f} m")
```

### 2. Minimal Jerk Trajectories

For smoother motion, use quintic (5th order) polynomials:

$$x(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

With constraints: $x(0) = 0$, $x(T) = L$, $\dot{x}(0) = \dot{x}(T) = 0$, $\ddot{x}(0) = \ddot{x}(T) = 0$

```python
def quintic_trajectory(x0, xf, T, num_points):
    """
    Generate minimal-jerk trajectory using quintic polynomial
    x0: initial position
    xf: final position
    T: total time
    num_points: number of trajectory samples
    """
    # Quintic polynomial coefficients
    # Constraints: x(0)=x0, x(T)=xf, x'(0)=0, x'(T)=0, x''(0)=0, x''(T)=0
    a0 = x0
    a1 = 0
    a2 = 0
    a3 = 10 * (xf - x0) / T**3
    a4 = -15 * (xf - x0) / T**4
    a5 = 6 * (xf - x0) / T**5

    t_array = np.linspace(0, T, num_points)
    x_array = a0 + a1*t_array + a2*t_array**2 + a3*t_array**3 + a4*t_array**4 + a5*t_array**5

    v_array = a1 + 2*a2*t_array + 3*a3*t_array**2 + 4*a4*t_array**3 + 5*a5*t_array**4

    return t_array, x_array, v_array

# Compare linear and quintic trajectories
t_lin, x_lin, v_lin = quintic_trajectory(0, 1, 2, 100)
print("Quintic trajectory properties:")
print(f"  Max velocity: {np.max(np.abs(v_lin)):.3f} m/s")
print(f"  Smoothness (jerk): {np.max(np.diff(v_lin, n=2)):.3f}")
```

## Walking Control Strategies

### Passive Dynamic Walking

Inspired by human walking without active control—using only gravity and inertia.

### Active Walking with Feedback Control

Most humanoid robots use active control with feedback from sensors:

```python
class WalkingController:
    """Simple proportional-derivative controller for walking"""

    def __init__(self, kp_zmp=100, kd_zmp=50):
        self.kp_zmp = kp_zmp
        self.kd_zmp = kd_zmp

    def compute_foot_adjustment(self, zmp_actual, zmp_desired, zmp_velocity):
        """
        Compute foot placement adjustment to stabilize ZMP
        """
        # ZMP error
        zmp_error = zmp_desired - zmp_actual

        # PD control
        adjustment = (self.kp_zmp * zmp_error -
                      self.kd_zmp * zmp_velocity)

        return adjustment

# Example: ZMP feedback control
controller = WalkingController(kp_zmp=50, kd_zmp=20)

# Simulate walking with perturbation
desired_zmp = 0.0
actual_zmp = 0.0
zmp_velocity = 0.0
results = {'time': [], 'zmp': [], 'adjustment': []}

for t in np.linspace(0, 5, 500):
    # Small perturbation at t=1s
    if t > 1.0 and t < 1.1:
        actual_zmp += 0.01  # 1 cm disturbance

    # Compute control adjustment
    adjustment = controller.compute_foot_adjustment(actual_zmp, desired_zmp, zmp_velocity)

    # Update ZMP (simplified dynamics)
    zmp_velocity = -0.5 * (actual_zmp - desired_zmp) + adjustment * 0.01
    actual_zmp += zmp_velocity * 0.01

    results['time'].append(t)
    results['zmp'].append(actual_zmp)
    results['adjustment'].append(adjustment)

print(f"ZMP stabilization: {np.std(results['zmp']):.6f} m")
```

## Exercise 4.1: Gait Simulation

**Objective**: Implement and analyze a walking gait pattern.

**Task**:
1. Design a gait pattern with specified:
   - Step length (0.2-0.4 m)
   - Walking speed (0.1-0.5 m/s)
   - Double support ratio (10-20%)
2. Generate foot trajectories for both legs
3. Compute COM motion using ZMP criterion
4. Simulate for 5 complete gait cycles
5. Analyze stability margins (minimum distance of ZMP to polygon boundary)
6. Identify any unstable phases and propose corrections
7. Visualize: ground contact, foot positions, COM trajectory, ZMP

**Submission**: Gait analysis report with visualizations and stability metrics.

---

## Next Chapter Preview

Chapter 5 explores the broader topic of balance and stability control beyond walking. You'll learn about control Lyapunov functions, capture steps, and how robots recover from disturbances.

[→ Next: Chapter 5 - Balance and Stability Control](/docs/module2-humanoid-robotics/chapter5-balance)
