---
sidebar_position: 4
title: Chapter 3 - Kinematics and Dynamics
---

# Kinematics and Dynamics

## Overview

Kinematics describes how a robot moves without considering the forces causing that motion. Dynamics explains *why* the robot moves—what forces and torques are required. Together, they form the mathematical foundation for robot control. In this chapter, you'll learn to compute end-effector positions from joint angles (forward kinematics), find joint angles from desired end-effector positions (inverse kinematics), and calculate the forces and torques needed to move the robot.

## Forward Kinematics

Forward kinematics answers: "Given the joint angles, where is the end-effector?"

### Homogeneous Transformation Matrices

We represent positions and orientations using 4×4 homogeneous transformation matrices:

$$T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & p_x \\ r_{21} & r_{22} & r_{23} & p_y \\ r_{31} & r_{32} & r_{33} & p_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Where $R$ is the 3×3 rotation matrix and $p$ is the position vector.

```python
import numpy as np
from scipy.spatial.transform import Rotation

class RobotArm:
    """Simple 3-DoF arm for forward kinematics demonstration"""

    def __init__(self, link_lengths):
        """
        Args:
            link_lengths: List of link lengths [L1, L2, L3]
        """
        self.lengths = link_lengths

    def forward_kinematics(self, theta):
        """
        Calculate end-effector position for planar 3-DoF arm
        theta: Joint angles [θ1, θ2, θ3] in radians
        Returns: (x, y, z) position and rotation matrix
        """
        theta1, theta2, theta3 = theta

        # Cumulative angles
        angle1 = theta1
        angle2 = theta1 + theta2
        angle3 = theta1 + theta2 + theta3

        # Position computation (planar)
        x = (self.lengths[0] * np.cos(angle1) +
             self.lengths[1] * np.cos(angle2) +
             self.lengths[2] * np.cos(angle3))

        y = (self.lengths[0] * np.sin(angle1) +
             self.lengths[1] * np.sin(angle2) +
             self.lengths[2] * np.sin(angle3))

        z = 0

        # Orientation (end-effector angle)
        orientation = angle3

        return np.array([x, y, z]), orientation

    def jacobian(self, theta):
        """
        Calculate Jacobian matrix (velocity mapping)
        J relates joint velocities to end-effector velocity: v = J * θ_dot
        """
        theta1, theta2, theta3 = theta

        # Cumulative angles
        a1 = theta1
        a2 = theta1 + theta2
        a3 = theta1 + theta2 + theta3

        # Jacobian for planar 3-DoF arm
        J = np.array([
            [
                -self.lengths[0] * np.sin(a1) - self.lengths[1] * np.sin(a2) - self.lengths[2] * np.sin(a3),
                -self.lengths[1] * np.sin(a2) - self.lengths[2] * np.sin(a3),
                -self.lengths[2] * np.sin(a3)
            ],
            [
                self.lengths[0] * np.cos(a1) + self.lengths[1] * np.cos(a2) + self.lengths[2] * np.cos(a3),
                self.lengths[1] * np.cos(a2) + self.lengths[2] * np.cos(a3),
                self.lengths[2] * np.cos(a3)
            ]
        ])

        return J

# Example: NAO arm (simplified)
nao_arm_lengths = [0.105, 0.120, 0.105]  # meters
arm = RobotArm(nao_arm_lengths)

# Test forward kinematics
test_angles = np.array([np.pi/6, -np.pi/4, np.pi/3])  # 30°, -45°, 60°
pos, ori = arm.forward_kinematics(test_angles)
print(f"End-effector position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
print(f"End-effector orientation: {np.degrees(ori):.1f}°")

# Calculate Jacobian
J = arm.jacobian(test_angles)
print(f"\nJacobian matrix:\n{J}")

# Singular configuration detection
det_J = np.linalg.det(J[:, :2])
print(f"Determinant (singularity measure): {det_J:.4f}")
```

### Denavit-Hartenberg (DH) Convention

For complex robots, we use DH parameters to systematically build transformation matrices:

$$T_i^{i-1} = \text{Rot}_z(\theta_i) \cdot \text{Trans}_z(d_i) \cdot \text{Trans}_x(a_i) \cdot \text{Rot}_x(\alpha_i)$$

```python
def dh_transform(theta, d, a, alpha):
    """
    Generate transformation matrix from DH parameters
    theta: rotation angle around z (radians)
    d: translation along z (meters)
    a: translation along x (meters)
    alpha: rotation angle around x (radians)
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    T = np.array([
        [c_theta, -s_theta*c_alpha, s_theta*s_alpha, a*c_theta],
        [s_theta, c_theta*c_alpha, -c_theta*s_alpha, a*s_theta],
        [0, s_alpha, c_alpha, d],
        [0, 0, 0, 1]
    ])

    return T

# Example: 2-link planar manipulator
# Link 1: length 0.3 m, Link 2: length 0.2 m
T_01 = dh_transform(np.pi/6, 0, 0.3, 0)  # Joint 1: 30°
T_12 = dh_transform(np.pi/4, 0, 0.2, 0)  # Joint 2: 45°
T_02 = T_01 @ T_12  # Base to end-effector

print("End-effector transformation matrix:")
print(T_02)
print(f"End-effector position: {T_02[:3, 3]}")
```

## Inverse Kinematics (IK)

Inverse kinematics solves: "Given desired end-effector position, what joint angles achieve it?"

This is significantly harder than forward kinematics:

- Multiple solutions may exist
- No closed-form solution exists for most 7-DoF arms
- Solution must satisfy joint limits
- Must avoid singularities

### Analytical IK for Planar 3-DoF Arm

For the planar arm, we can derive closed-form solutions:

```python
def inverse_kinematics_planar_3dof(target_x, target_y, arm_lengths):
    """
    Analytical inverse kinematics for planar 3-DoF arm
    Using elbow-up configuration
    """
    L1, L2, L3 = arm_lengths

    # Using law of cosines
    r = np.sqrt(target_x**2 + target_y**2)

    # Check workspace
    if r > L1 + L2 + L3 or r < abs(L1 - L2 - L3):
        return None, "Target out of workspace"

    # Joint 3 (wrist) angle using law of cosines
    cos_theta3 = (r**2 - L1**2 - L2**2 - L3**2) / (2*L2*L3)
    if abs(cos_theta3) > 1:
        return None, "No solution"

    theta3 = -np.arccos(cos_theta3)  # Elbow-down config

    # Joint 2 (elbow) angle
    k1 = L1 + L2*np.cos(theta3)
    k2 = L2*np.sin(theta3)
    theta2 = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

    # Joint 1 (shoulder) angle
    theta1 = np.arctan2(target_y, target_x) - theta2 - theta3

    return np.array([theta1, theta2, theta3]), "Success"

# Test IK
target_pos = np.array([0.2, 0.15])
theta_ik, status = inverse_kinematics_planar_3dof(target_pos[0], target_pos[1], nao_arm_lengths)

if status == "Success":
    print(f"IK Solution: θ1={np.degrees(theta_ik[0]):.1f}°, θ2={np.degrees(theta_ik[1]):.1f}°, θ3={np.degrees(theta_ik[2]):.1f}°")

    # Verify with forward kinematics
    pos_check, _ = arm.forward_kinematics(theta_ik)
    print(f"Verification: ({pos_check[0]:.3f}, {pos_check[1]:.3f})")
```

### Numerical IK Using Jacobian Transpose Method

For complex robots, we use iterative methods:

```python
def numerical_ik_jacobian_transpose(arm, target_pos, initial_theta, max_iterations=100, learning_rate=0.1):
    """
    Numerical inverse kinematics using Jacobian transpose method
    """
    theta = initial_theta.copy()
    error_history = []

    for iteration in range(max_iterations):
        # Forward kinematics
        current_pos, _ = arm.forward_kinematics(theta)

        # Position error
        error = target_pos - current_pos[:2]  # Only x, y
        error_norm = np.linalg.norm(error)
        error_history.append(error_norm)

        if error_norm < 1e-4:
            print(f"Converged in {iteration} iterations")
            return theta, True

        # Jacobian and its transpose
        J = arm.jacobian(theta)
        J_T = J.T

        # Compute pseudo-inverse using transpose
        dtheta = learning_rate * J_T @ error

        # Update joint angles
        theta += dtheta

    print(f"Did not converge. Final error: {error_history[-1]:.6f}")
    return theta, False

# Test numerical IK
theta_init = np.zeros(3)
theta_numerical, converged = numerical_ik_jacobian_transpose(
    arm, np.array([0.25, 0.15]), theta_init
)
print(f"Numerical IK: {np.degrees(theta_numerical)}")
```

## Dynamics

Dynamics describes the relationship between forces and motion. The fundamental equation is:

```text
τ = M(θ)θ̈ + C(θ, θ̇)θ̇ + G(θ)
```

Where:
- τ = applied torques
- M(θ) = mass/inertia matrix
- C(θ, θ̇) = Coriolis and centrifugal terms
- G(θ) = gravitational torques

### Lagrangian Mechanics Approach

```python
def compute_gravity_torque(theta, arm_segments):
    """
    Compute gravitational torque for each joint
    arm_segments: List of (mass, length) for each segment
    """
    n_joints = len(theta)
    gravity = 9.81

    G = np.zeros(n_joints)

    # Simplified calculation (planar case)
    for i in range(n_joints):
        # Center of mass of all segments beyond joint i
        mass_sum = sum(m for m, _ in arm_segments[i:])
        com_distance = sum(l for _, l in arm_segments[i:]) / len(arm_segments[i:])

        # Cumulative angle to segment COM
        cumulative_angle = sum(theta[:i+1])

        # Torque = mass * g * distance * cos(angle)
        G[i] = -mass_sum * gravity * com_distance * np.cos(cumulative_angle)

    return G

# Example: gravity compensation
arm_segments = [(0.5, 0.1), (0.3, 0.12), (0.2, 0.1)]  # (mass kg, length m)
test_config = np.array([np.pi/4, -np.pi/6, np.pi/3])

gravity_torques = compute_gravity_torque(test_config, arm_segments)
print(f"Gravity compensation torques: {gravity_torques} N·m")
```

## Exercise 3.1: Kinematics Simulation

**Objective**: Implement forward and inverse kinematics for a 2-DoF robot arm.

**Task**:
1. Create a 2-DoF planar arm with 0.3 m and 0.2 m links
2. Implement forward kinematics
3. Implement analytical inverse kinematics
4. Create 10 test trajectories (straight lines, circles)
5. For each trajectory point:
   - Compute required joint angles (IK)
   - Verify positions (FK)
   - Calculate position error
6. Visualize the arm configuration and trajectory

**Submission**: Code with 2D visualization of arm motion and error analysis.

---

## Next Chapter Preview

Chapter 4 focuses on bipedal walking—how humanoid robots maintain balance while moving on two legs. You'll learn about gait patterns, zero moment point (ZMP) theory, and the control strategies that enable stable walking.

[→ Next: Chapter 4 - Bipedal Walking Fundamentals](/docs/module2-humanoid-robotics/chapter4-walking)
