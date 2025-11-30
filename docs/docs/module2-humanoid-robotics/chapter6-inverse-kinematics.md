---
sidebar_position: 7
title: Chapter 6 - Inverse Kinematics and Motion Planning
---

# Inverse Kinematics and Motion Planning

## Overview

While forward kinematics tells us where the end-effector is, inverse kinematics answers the practical question: "What joint angles achieve a desired end-effector position?" This chapter explores both analytical and numerical methods for solving inverse kinematics, then extends to trajectory planning—computing smooth, collision-free paths through space. These techniques are essential for any manipulation task a humanoid robot performs.

## Inverse Kinematics Problem Formulation

Given:
- Desired end-effector position: `p_d` ∈ ℝ³
- Desired end-effector orientation: `R_d` ∈ SO(3)
- Current joint configuration: `θ₀`

Find: `θ` such that:

```text
f_k(θ) = p_d,  R_k(θ) = R_d
```

### Challenges

1. **Non-uniqueness**: Multiple solutions may exist (redundancy)
2. **Non-existence**: Target may be outside workspace
3. **Computational complexity**: No closed-form solution for 7+ DoF
4. **Singularities**: Jacobian becomes singular at certain configurations
5. **Constraints**: Joint limits, collision avoidance, etc.

```python
import numpy as np
from scipy.optimize import minimize

class InverseKinematicsSolver:
    """IK solver for general manipulator"""

    def __init__(self, forward_kinematics_func, jacobian_func, joint_limits):
        """
        forward_kinematics_func: theta -> (position, orientation)
        jacobian_func: theta -> jacobian_matrix
        joint_limits: list of (min, max) for each joint
        """
        self.fk = forward_kinematics_func
        self.jacobian = jacobian_func
        self.limits = joint_limits
        self.n_joints = len(joint_limits)

    def objective_function(self, theta, target_pos, target_orient=None, weight_pos=1.0):
        """
        Minimize error between current and target configuration
        """
        # Forward kinematics
        pos, orient = self.fk(theta)

        # Position error
        pos_error = np.linalg.norm(pos - target_pos)**2

        # Optional: orientation error
        total_error = weight_pos * pos_error

        # Add joint limit penalty
        for i, (min_lim, max_lim) in enumerate(self.limits):
            if theta[i] < min_lim or theta[i] > max_lim:
                total_error += 1000 * (theta[i] - np.clip(theta[i], min_lim, max_lim))**2

        return total_error

    def solve(self, target_pos, initial_theta=None, max_iterations=1000, tolerance=1e-6):
        """
        Solve IK using numerical optimization
        """
        if initial_theta is None:
            initial_theta = np.zeros(self.n_joints)
        else:
            initial_theta = np.array(initial_theta)

        # Constrain to joint limits
        bounds = self.limits

        # Minimize position error
        result = minimize(
            self.objective_function,
            initial_theta,
            args=(target_pos,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )

        return result.x, result.fun
```

## Numerical IK Methods

### Jacobian Transpose Method (Simplest)

The Jacobian relates joint velocities to end-effector velocities:

$$\vec{v} = J(\vec{\theta}) \dot{\vec{\theta}}$$

To move toward target, use:

$$\dot{\vec{\theta}} = J^T(\vec{\theta}) \cdot \vec{e}$$

Where $\vec{e}$ is the position error.

```python
def jacobian_transpose_ik(forward_kinematics, jacobian_func, target_pos,
                          initial_theta, max_iterations=100, learning_rate=0.1,
                          tolerance=1e-4):
    """
    Iterative IK using Jacobian transpose method
    """
    theta = np.array(initial_theta)
    error_history = []

    for iteration in range(max_iterations):
        # Forward kinematics
        pos, _ = forward_kinematics(theta)

        # Position error
        error = target_pos - pos
        error_norm = np.linalg.norm(error)
        error_history.append(error_norm)

        if error_norm < tolerance:
            return theta, True, error_history

        # Jacobian and its transpose
        J = jacobian_func(theta)
        J_T = J.T

        # Update rule: θ += α * J^T * e
        dtheta = learning_rate * J_T @ error

        # Update joint angles
        theta += dtheta

        if iteration % 20 == 0:
            print(f"Iteration {iteration}: Error = {error_norm:.6f}")

    return theta, False, error_history
```

### Pseudo-Inverse Method (Damped Least Squares)

More robust than transpose, using damped pseudo-inverse:

$$\dot{\vec{\theta}} = (J^T J + \lambda I)^{-1} J^T \vec{e}$$

Where $\lambda$ is a damping factor (typically 0.001-0.01).

```python
def damped_least_squares_ik(forward_kinematics, jacobian_func, target_pos,
                            initial_theta, damping=0.01, max_iterations=100,
                            learning_rate=0.1, tolerance=1e-4):
    """
    Damped Least Squares (Levenberg-Marquardt) IK method
    """
    theta = np.array(initial_theta, dtype=float)
    error_history = []

    for iteration in range(max_iterations):
        pos, _ = forward_kinematics(theta)
        error = target_pos - pos
        error_norm = np.linalg.norm(error)
        error_history.append(error_norm)

        if error_norm < tolerance:
            return theta, True, error_history

        # Jacobian
        J = jacobian_func(theta)

        # Damped pseudo-inverse
        JTJ = J.T @ J
        damped_term = damping * np.eye(J.shape[1])
        J_pinv = np.linalg.inv(JTJ + damped_term) @ J.T

        # Update
        dtheta = learning_rate * J_pinv @ error
        theta += dtheta

        if iteration % 20 == 0:
            print(f"Iteration {iteration}: Error = {error_norm:.6f}")

    return theta, False, error_history
```

## Workspace Analysis

Understanding the robot's workspace is crucial for determining feasible targets.

```python
class WorkspaceAnalyzer:
    """Analyze robot workspace by sampling joint configurations"""

    def __init__(self, forward_kinematics, joint_limits):
        self.fk = forward_kinematics
        self.limits = joint_limits
        self.n_joints = len(joint_limits)

    def compute_workspace(self, samples_per_joint=10):
        """
        Sample workspace by evaluating FK at grid of joint configurations
        """
        positions = []

        # Create sample joint angles
        sample_points = [np.linspace(min_l, max_l, samples_per_joint)
                        for min_l, max_l in self.limits]

        # Grid evaluation
        import itertools
        for config in itertools.product(*sample_points):
            pos, _ = self.fk(np.array(config))
            positions.append(pos[:3])  # Only x, y, z

        return np.array(positions)

    def compute_reachability_map(self, target_pos, resolution=0.01):
        """
        Determine if target is reachable (within workspace)
        """
        workspace = self.compute_workspace(samples_per_joint=15)

        # Find nearest point in workspace
        distances = np.linalg.norm(workspace - target_pos, axis=1)
        min_distance = np.min(distances)

        is_reachable = min_distance < resolution

        return is_reachable, min_distance

    def visualize_workspace_projection(self, plane='xy'):
        """
        Project 3D workspace to 2D plane for visualization
        """
        workspace = self.compute_workspace(samples_per_joint=8)

        if plane == 'xy':
            return workspace[:, 0], workspace[:, 1]
        elif plane == 'xz':
            return workspace[:, 0], workspace[:, 2]
        elif plane == 'yz':
            return workspace[:, 1], workspace[:, 2]
```

## Trajectory Planning

Once we have IK for individual points, we need to plan smooth trajectories through intermediate configurations.

### Path vs. Trajectory

- **Path**: Sequence of positions (geometric)
- **Trajectory**: Path with timing information (t, $\theta$(t), $\dot{\theta}$(t), $\ddot{\theta}$(t))

### Cartesian Trajectory Planning

Plan motion in Cartesian space, then convert to joint space:

```python
def cartesian_linear_trajectory(start_pos, end_pos, num_waypoints):
    """
    Generate linear trajectory in Cartesian space
    """
    waypoints = []
    for alpha in np.linspace(0, 1, num_waypoints):
        waypoint = (1 - alpha) * start_pos + alpha * end_pos
        waypoints.append(waypoint)

    return np.array(waypoints)

def trajectory_with_orientation(start_pos, end_pos, start_orient, end_orient,
                               num_waypoints):
    """
    Generate trajectory with smooth position and orientation
    Using SLERP (Spherical Linear Interpolation) for orientation
    """
    from scipy.spatial.transform import Rotation as R

    positions = cartesian_linear_trajectory(start_pos, end_pos, num_waypoints)

    # SLERP for orientation
    R_start = R.from_matrix(start_orient)
    R_end = R.from_matrix(end_orient)

    orientations = []
    for alpha in np.linspace(0, 1, num_waypoints):
        R_interp = R_start.slerp(alpha, R_end)
        orientations.append(R_interp.as_matrix())

    return positions, orientations
```

### Joint Space Trajectory Planning

More direct: plan trajectories in joint configuration space.

#### Trapezoidal Velocity Profile

Simplest approach: accelerate to max velocity, then decelerate.

```python
def trapezoidal_velocity_profile(q_start, q_goal, v_max, a_max, dt=0.01):
    """
    Generate trapezoidal velocity profile for smooth joint motion
    q_start, q_goal: initial and final joint angles
    v_max: maximum velocity
    a_max: maximum acceleration
    """
    distance = abs(q_goal - q_start)

    # Calculate phases
    t_accel = v_max / a_max
    s_accel = 0.5 * a_max * t_accel**2

    if 2 * s_accel <= distance:
        # Trapezoidal profile (has constant velocity phase)
        t_coast = (distance - 2*s_accel) / v_max
        t_total = 2*t_accel + t_coast
    else:
        # Triangular profile (no constant velocity phase)
        t_accel = np.sqrt(distance / a_max)
        t_coast = 0
        t_total = 2*t_accel

    # Generate trajectory
    t = np.arange(0, t_total, dt)
    q = np.zeros_like(t)
    v = np.zeros_like(t)

    for i, time in enumerate(t):
        if time <= t_accel:
            # Acceleration phase
            v[i] = a_max * time
            q[i] = q_start + 0.5 * a_max * time**2
        elif time <= t_accel + t_coast:
            # Constant velocity phase
            v[i] = v_max
            q[i] = q_start + s_accel + v_max * (time - t_accel)
        else:
            # Deceleration phase
            time_decel = time - (t_accel + t_coast)
            v[i] = v_max - a_max * time_decel
            q[i] = q_goal - 0.5 * a_max * (t_total - time)**2

    return t, q, v

# Example: move joint from 0 to 1 radian
t, q, v = trapezoidal_velocity_profile(0, 1.0, v_max=0.5, a_max=0.5)
print(f"Trajectory duration: {t[-1]:.2f} s")
print(f"Max velocity reached: {np.max(v):.3f} rad/s")
```

#### Quintic (5th Order) Polynomials

For smoother motion with zero jerk at endpoints:

```python
def quintic_polynomial_trajectory(q_start, q_goal, t_total, num_points):
    """
    Generate trajectory using quintic polynomial
    Guarantees: q(0)=q_start, q(T)=q_goal, v(0)=v(T)=0, a(0)=a(T)=0
    """
    t = np.linspace(0, t_total, num_points)

    # Normalize time to [0, 1]
    tau = t / t_total

    # Quintic coefficients (normalized)
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 10
    a4 = -15
    a5 = 6

    # Position, velocity, acceleration
    q = q_start + (q_goal - q_start) * (a3*tau**3 + a4*tau**4 + a5*tau**5)
    v = (q_goal - q_start) / t_total * (3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4)
    a = (q_goal - q_start) / t_total**2 * (6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3)

    return t, q, v, a

# Compare quintic vs trapezoidal
t_quint, q_quint, v_quint, a_quint = quintic_polynomial_trajectory(0, 1, 2, 200)
t_trap, q_trap, v_trap = trapezoidal_velocity_profile(0, 1, v_max=0.5, a_max=0.5)

print("Trajectory Smoothness Comparison:")
print(f"Quintic max acceleration: {np.max(np.abs(a_quint)):.3f} rad/s²")
print(f"Trapezoidal max acceleration: {0.5:.3f} rad/s² (by design)")
print(f"Quintic smoothness (max jerk): {np.max(np.diff(a_quint, n=1)):.3f}")
```

## Collision Avoidance

Basic collision checking during trajectory planning:

```python
def check_self_collision(theta, collision_geometry):
    """
    Check if configuration results in self-collision
    collision_geometry: list of (link1, link2, min_distance) tuples
    """
    for link1, link2, min_dist in collision_geometry:
        # Compute distance between link segments
        # Simplified: check if distance < min_dist
        distance = compute_link_distance(theta, link1, link2)

        if distance < min_dist:
            return True, distance

    return False, None

def plan_collision_free_trajectory(start_config, goal_config, collision_geometry,
                                   max_iterations=100):
    """
    Plan trajectory avoiding collisions using RRT-style sampling
    """
    path = [start_config]
    current = start_config.copy()

    for iteration in range(max_iterations):
        # Random goal bias
        if np.random.random() < 0.1:
            target = goal_config
        else:
            # Random configuration
            target = np.random.uniform(
                [-np.pi]*len(current),
                [np.pi]*len(current)
            )

        # Linear interpolation toward target
        direction = target - current
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        new_config = current + 0.1 * direction

        # Check collision
        is_colliding, dist = check_self_collision(new_config, collision_geometry)

        if not is_colliding:
            path.append(new_config)
            current = new_config

            if np.linalg.norm(current - goal_config) < 0.1:
                path.append(goal_config)
                return path, True

    return path, False
```

## Exercise 6.1: IK and Motion Planning

**Objective**: Implement complete manipulation pipeline.

**Task**:
1. Implement 3-DoF arm forward kinematics and Jacobian
2. Implement numerical IK solver using both:
   - Jacobian transpose method
   - Damped least squares method
3. Generate 5 target positions within workspace
4. For each target:
   - Solve IK from random initial configuration
   - Verify solution with FK
   - Measure convergence speed (iterations, error)
5. Plan smooth Cartesian trajectory between two reachable positions:
   - Generate 10 waypoints
   - Convert each to joint space (IK)
   - Create quintic polynomial trajectory
   - Check for collisions along path
6. Analyze:
   - Workspace coverage (cone of reachability)
   - IK solution quality
   - Trajectory smoothness

**Submission**: Motion planning report with IK convergence analysis, workspace visualization, and smooth trajectory demonstration.

---

## Next Chapter Preview

Chapter 7 extends local motion control to whole-body control—coordinating multiple limbs simultaneously while maintaining balance. You'll learn about task-space control, constraint handling, and coordination strategies.

[→ Next: Chapter 7 - Whole-Body Control](/docs/module2-humanoid-robotics/chapter7-whole-body-control)
