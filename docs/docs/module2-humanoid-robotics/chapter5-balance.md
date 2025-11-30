---
sidebar_position: 6
title: Chapter 5 - Balance and Stability Control
---

# Balance and Stability Control

## Overview

Balance is the foundation of humanoid robot locomotion and manipulation. Unlike industrial robots that are typically anchored, humanoid robots must maintain balance while walking, standing, and interacting with their environment. This chapter explores the mathematics and control strategies for maintaining balance, including inverted pendulum models, control theory fundamentals, and recovery strategies for handling disturbances.

## Inverted Pendulum Model

The most widely used model for bipedal balance is the Linear Inverted Pendulum Model (LIPM):

### Simple Pendulum vs. Inverted Pendulum

For a pendulum with pivot at origin and mass m at distance l:

**Simple Pendulum** (stable equilibrium):
```text
θ̈ + (g/l)sin(θ) = 0
```

**Inverted Pendulum** (unstable equilibrium):
```text
θ̈ - (g/l)sin(θ) = 0
```

### Linear Inverted Pendulum Model (LIPM)

For small angles and constant height h:

```
ẍ = (g/h)(x - x_ZMP)
```

Where:
- x = horizontal position of center of mass
- x_ZMP = zero moment point location
- h = height of center of mass
- g = gravitational acceleration

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearInvertedPendulum:
    """Model of humanoid balance as linear inverted pendulum"""

    def __init__(self, com_height, g=9.81):
        """
        Args:
            com_height: Height of center of mass (m)
            g: gravitational acceleration (m/s²)
        """
        self.h = com_height
        self.g = g
        self.omega = np.sqrt(g / com_height)  # Natural frequency

    def compute_dynamics(self, x, v, zmp, dt):
        """
        Update LIPM state given ZMP reference
        x: COM position (m)
        v: COM velocity (m/s)
        zmp: zero moment point (m)
        dt: time step (s)
        Returns: (new_x, new_v)
        """
        # Acceleration from LIPM equation
        a = self.omega**2 * (x - zmp)

        # Update velocity and position
        v_new = v + a * dt
        x_new = x + v_new * dt

        return x_new, v_new

    def capture_step(self, x, v):
        """
        Compute capture point: location where robot must place foot
        to come to rest under ZMP control
        """
        # Capture point = x + v/ω
        capture_point = x + v / self.omega
        return capture_point

# Simulate LIPM balance control
lipm = LinearInvertedPendulum(com_height=1.0)

# Initial conditions
x = 0.01  # 1 cm perturbation
v = 0.0
zmp = 0.0
dt = 0.01
t = np.arange(0, 5, dt)

trajectory = {'t': [], 'x': [], 'v': [], 'cp': []}

for time in t:
    # Store state
    trajectory['t'].append(time)
    trajectory['x'].append(x)
    trajectory['v'].append(v)
    trajectory['cp'].append(lipm.capture_step(x, v))

    # Update LIPM
    x, v = lipm.compute_dynamics(x, v, zmp, dt)

print(f"LIPM Natural frequency: {lipm.omega:.2f} rad/s")
print(f"Final position: {x:.6f} m")
print(f"Final velocity: {v:.6f} m/s")
print(f"Max displacement: {np.max(np.abs(trajectory['x'])):.6f} m")
```

## Stability Analysis Using Lyapunov Theory

Lyapunov theory provides formal stability guarantees. A system is stable if we can find a Lyapunov function $V$ such that $\dot{V} < 0$.

### Quadratic Lyapunov Function for LIPM

For the LIPM, a suitable Lyapunov function is:

$$V = \frac{1}{2}(x - x_{ZMP})^2 + \frac{1}{2\omega^2}(\dot{x})^2$$

This represents the total "energy" relative to equilibrium.

```python
class LyapunovStability:
    """Stability analysis using Lyapunov functions"""

    def __init__(self, omega):
        self.omega = omega

    def lyapunov_function(self, x, v, zmp):
        """
        Compute Lyapunov function V for LIPM
        V > 0 everywhere except at equilibrium
        V -> 0 as state converges to equilibrium
        """
        position_term = 0.5 * (x - zmp)**2
        velocity_term = 0.5 * v**2 / (self.omega**2)
        V = position_term + velocity_term
        return V

    def lyapunov_derivative(self, x, v, zmp):
        """
        Compute dV/dt = ∇V · dx/dt
        Shows convergence rate
        """
        dx_dt = v
        dv_dt = self.omega**2 * (x - zmp)

        # dV/dt = (x - zmp) * v + v/ω² * ω²(x - zmp)
        #       = (x - zmp) * v + v * (x - zmp)
        #       = 2v(x - zmp)

        dV_dt = 2 * v * (x - zmp)
        return dV_dt

# Verify Lyapunov properties
lyap = LyapunovStability(omega=3.13)  # sqrt(9.81/1.0)

# Test at various perturbations
print("Lyapunov Function Stability Test:")
print(f"{'Position (m)':>15} {'Velocity (m/s)':>15} {'V':>15} {'dV/dt':>15}")
print("-" * 60)

for x_test in [-0.1, -0.05, 0, 0.05, 0.1]:
    for v_test in [-0.2, 0, 0.2]:
        V = lyap.lyapunov_function(x_test, v_test, 0)
        dV_dt = lyap.lyapunov_derivative(x_test, v_test, 0)
        print(f"{x_test:>15.3f} {v_test:>15.3f} {V:>15.6f} {dV_dt:>15.6f}")
        if V > 0 and dV_dt >= 0 and (x_test != 0 or v_test != 0):
            print("  ⚠ Non-decreasing Lyapunov!")
```

## Control Strategies for Balance

### 1. Proportional-Derivative (PD) Control

The simplest and most common approach:

$$u = -K_p e - K_d \dot{e}$$

Where $e$ is the error (difference from desired state).

```python
class PDBalanceController:
    """PD controller for balancing"""

    def __init__(self, kp, kd):
        """
        kp: proportional gain
        kd: derivative gain
        """
        self.kp = kp
        self.kd = kd
        self.error_integral = 0

    def compute_control(self, error, error_rate, dt):
        """
        Compute control output
        error: current error (e.g., ZMP position error)
        error_rate: rate of error change (derivative)
        dt: time step
        """
        u = -self.kp * error - self.kd * error_rate
        return u

# Simulate PD control for balance
controller = PDBalanceController(kp=100, kd=50)
lipm = LinearInvertedPendulum(com_height=1.0)

# Initial perturbation
x = 0.05  # 5 cm disturbance
v = 0.0
zmp_desired = 0.0
dt = 0.01

results = {'t': [], 'x': [], 'zmp_actual': []}

for _ in range(500):
    # Compute ZMP adjustment via PD control
    zmp_error = x - zmp_desired
    zmp_adjustment = controller.compute_control(zmp_error, v, dt)

    # New ZMP position
    zmp_actual = zmp_desired + zmp_adjustment

    # Update LIPM state
    x, v = lipm.compute_dynamics(x, v, zmp_actual, dt)

    results['t'].append(_ * dt)
    results['x'].append(x)
    results['zmp_actual'].append(zmp_actual)

print(f"PD Control Results:")
print(f"  Final position: {results['x'][-1]:.6f} m")
print(f"  Settling time: {np.argmax(np.abs(results['x']) < 0.001) * dt:.2f} s")
```

### 2. Capture Step Strategy

When the robot receives a large perturbation, it must take a step to recover:

```python
class CaptureStepController:
    """Compute safe capture step during disturbance"""

    def __init__(self, lipm, foot_length=0.3, max_step=0.5):
        self.lipm = lipm
        self.foot_length = foot_length
        self.max_step = max_step

    def should_take_step(self, x, v, support_foot_pos):
        """
        Determine if capture step is necessary
        """
        # Compute capture point where robot would come to rest
        capture_point = self.lipm.capture_step(x, v)

        # Check if capture point exceeds feasible step region
        feasible_region = [
            support_foot_pos - self.foot_length/2,
            support_foot_pos + self.max_step
        ]

        return not (feasible_region[0] <= capture_point <= feasible_region[1])

    def compute_step_location(self, x, v, current_foot_pos):
        """
        Compute optimal next foot placement
        """
        # Place foot at capture point
        capture_point = self.lipm.capture_point(x, v)

        # Clamp to feasible region
        step_location = np.clip(
            capture_point,
            current_foot_pos,
            current_foot_pos + self.max_step
        )

        return step_location

capture_controller = CaptureStepController(lipm, foot_length=0.3, max_step=0.5)

# Test with large perturbation
x = 0.3  # 30 cm perturbation
v = 0.0
support_foot = 0.0

if capture_controller.should_take_step(x, v, support_foot):
    new_foot_pos = capture_controller.compute_step_location(x, v, support_foot)
    print(f"Capture step required!")
    print(f"  Current position: {x:.3f} m")
    print(f"  Capture point: {lipm.capture_step(x, v):.3f} m")
    print(f"  New foot position: {new_foot_pos:.3f} m")
```

## Disturbance Rejection and Recovery

### Step Response to Impulse Disturbance

```python
def simulate_disturbance_recovery(lipm, impulse_magnitude, impulse_time, total_time=10):
    """
    Simulate recovery from impulse disturbance
    impulse_magnitude: velocity change from disturbance (m/s)
    impulse_time: when disturbance occurs (s)
    """
    x = 0.0
    v = 0.0
    zmp = 0.0  # Fixed ZMP
    dt = 0.01

    results = {'t': [], 'x': [], 'v': []}

    for t in np.arange(0, total_time, dt):
        # Apply impulse at specified time
        if abs(t - impulse_time) < dt:
            v += impulse_magnitude

        results['t'].append(t)
        results['x'].append(x)
        results['v'].append(v)

        # Update state
        x, v = lipm.compute_dynamics(x, v, zmp, dt)

    return results

# Simulate recovery from 0.2 m/s push
recovery = simulate_disturbance_recovery(lipm, impulse_magnitude=0.2, impulse_time=1.0)

max_displacement = np.max(np.abs(recovery['x']))
settling_time = np.argmax(np.array(recovery['x']) < 0.01) * 0.01

print(f"Disturbance Recovery:")
print(f"  Max displacement: {max_displacement:.4f} m")
print(f"  Settling time: {settling_time:.2f} s")
```

## Multi-Contact Stability

During transitions between single and double support, stability analysis is more complex:

$$\tau_{total} = \tau_{gravity} + \tau_{applied}$$

Where stability depends on contact forces at multiple contact points.

```python
class MultiContactBalance:
    """Balance analysis with multiple contact points"""

    def __init__(self, mass, gravity=9.81):
        self.m = mass
        self.g = gravity

    def compute_contact_forces(self, com_pos, com_acc, contact_points, num_contacts):
        """
        Compute required contact forces for balance
        com_pos: center of mass position (x, y, z)
        com_acc: center of mass acceleration
        contact_points: list of contact positions [(x,y), ...]
        num_contacts: number of active contacts
        """
        # Total force needed
        fz_total = self.m * (self.g + com_acc[2])  # Vertical
        fx_total = self.m * com_acc[0]  # Horizontal x
        fy_total = self.m * com_acc[1]  # Horizontal y

        # Distribute among contacts (simplified: equal distribution)
        if num_contacts > 0:
            fz_per_contact = fz_total / num_contacts
            fx_per_contact = fx_total / num_contacts
            fy_per_contact = fy_total / num_contacts
        else:
            fz_per_contact = fx_per_contact = fy_per_contact = 0

        return fz_per_contact, fx_per_contact, fy_per_contact

# Example: transition from single to double support
balance = MultiContactBalance(mass=50)  # 50 kg robot

# Single support: one foot
forces_single = balance.compute_contact_forces(
    com_pos=(0, 0, 1),
    com_acc=(0.5, 0, 0),  # 0.5 m/s² forward acceleration
    contact_points=[(0, 0)],
    num_contacts=1
)

# Double support: two feet
forces_double = balance.compute_contact_forces(
    com_pos=(0, 0, 1),
    com_acc=(0.5, 0, 0),
    contact_points=[(0.15, 0), (-0.15, 0)],
    num_contacts=2
)

print(f"Single Support - Force per contact: Z={forces_single[0]:.1f} N")
print(f"Double Support - Force per contact: Z={forces_double[0]:.1f} N")
print(f"Force reduction in double support: {(1 - forces_double[0]/forces_single[0])*100:.1f}%")
```

## Exercise 5.1: Balance Control Design

**Objective**: Design and simulate a balance controller.

**Task**:
1. Implement a Linear Inverted Pendulum Model for a 1.7 m tall humanoid (82 kg)
2. Design a PD controller for ZMP-based balance
3. Simulate response to:
   - Small sustained perturbation (1 cm displacement)
   - Impulse disturbance (0.1 m/s velocity push)
   - Step disturbance in ZMP reference
4. For each scenario, measure:
   - Settling time (time to reach ±1 cm)
   - Maximum displacement
   - Control effort (integral of |u|)
5. Optimize controller gains using performance metrics
6. Verify stability using Lyapunov analysis

**Submission**: Balance control report with simulation results, gain tuning analysis, and Lyapunov stability verification.

---

## Next Chapter Preview

Chapter 6 focuses on inverse kinematics and motion planning. You'll learn how to compute desired joint trajectories to achieve target end-effector positions while satisfying constraints and avoiding obstacles.

[→ Next: Chapter 6 - Inverse Kinematics and Motion Planning](/docs/module2-humanoid-robotics/chapter6-inverse-kinematics)
