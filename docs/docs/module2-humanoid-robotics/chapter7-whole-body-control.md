---
sidebar_position: 8
title: Chapter 7 - Whole-Body Control
---

# Whole-Body Control

## Overview

A humanoid robot is not just an arm or legs in isolation—it's an integrated system where the movement of every joint affects balance, reachability, and task performance. Whole-body control coordinates all joints simultaneously to achieve multiple objectives: reach a target with the hand, maintain balance, avoid obstacles, and respect joint limits. This chapter explores the mathematical framework for multi-objective control, hierarchical task management, and practical implementation strategies for humanoid robots.

## Multi-Objective Control Framework

Unlike single-arm robots that optimize only end-effector motion, humanoids must balance competing objectives:

1. **Primary Task**: Reach desired position with hand
2. **Balance Constraint**: Keep center of mass stable
3. **Joint Limits**: Respect mechanical constraints
4. **Self-Collision Avoidance**: Prevent arm-body interference
5. **Task Priority**: Some tasks more important than others

### Task Hierarchy

Tasks are organized by priority. Higher-priority tasks must be satisfied; lower-priority tasks use null-space of higher-priority tasks:

```
θ̇ = J₁⁺ẋ₁ + (I - J₁⁺J₁)J₂⁺ẋ₂ + ⋯
```

Where J_i⁺ is the pseudo-inverse of Jacobian for task i.

```python
import numpy as np
from scipy.linalg import svd

class WholeBodyController:
    """Multi-objective whole-body control using task hierarchy"""

    def __init__(self, n_joints, task_hierarchy):
        """
        n_joints: number of robot joints
        task_hierarchy: list of Task objects in priority order
        """
        self.n_joints = n_joints
        self.tasks = task_hierarchy

    def compute_joint_velocities(self, theta, target_config):
        """
        Compute joint velocities satisfying task hierarchy
        """
        # Start with zero velocity
        dtheta = np.zeros(self.n_joints)
        P = np.eye(self.n_joints)  # Projection matrix (starts as identity)

        for task in self.tasks:
            if not task.active:
                continue

            # Get Jacobian for this task
            J = task.compute_jacobian(theta)

            # Project into null-space of higher-priority tasks
            J_projected = J @ P

            # Pseudo-inverse using damping for numerical stability
            J_pinv = self._damped_pinv(J_projected, damping=0.001)

            # Task error
            error = task.compute_error(theta, target_config)

            # Contribution to joint velocity
            dtheta_task = P @ J_pinv @ error

            # Update joint velocity
            dtheta += dtheta_task

            # Update null-space projection for next task
            # P_new = (I - J_pinv @ J) @ P
            P = (np.eye(self.n_joints) - J_pinv @ J_projected) @ P

        return dtheta

    @staticmethod
    def _damped_pinv(J, damping=0.001):
        """
        Compute damped pseudo-inverse to improve numerical stability
        """
        U, s, Vt = svd(J, full_matrices=False)

        # Damped singular values
        s_damped = s / (s**2 + damping**2)

        # Pseudo-inverse
        J_pinv = Vt.T @ np.diag(s_damped) @ U.T

        return J_pinv

class Task:
    """Abstract task for whole-body control"""

    def __init__(self, name, priority, weight=1.0):
        self.name = name
        self.priority = priority
        self.weight = weight
        self.active = True

    def compute_jacobian(self, theta):
        """Return Jacobian for this task"""
        raise NotImplementedError

    def compute_error(self, theta, target):
        """Return error vector to minimize"""
        raise NotImplementedError

class EndEffectorTask(Task):
    """Task: reach target position with end-effector"""

    def __init__(self, forward_kinematics, jacobian_func, target_pos, name="EE", priority=0):
        super().__init__(name, priority)
        self.fk = forward_kinematics
        self.jacobian = jacobian_func
        self.target_pos = target_pos

    def compute_jacobian(self, theta):
        return self.jacobian(theta)

    def compute_error(self, theta, target):
        pos, _ = self.fk(theta)
        return target - pos[:3]

class BalanceTask(Task):
    """Task: maintain zero moment point within support polygon"""

    def __init__(self, com_jacobian, zmp_target, support_polygon, priority=1):
        super().__init__("Balance", priority)
        self.jacobian_com = com_jacobian
        self.zmp_target = zmp_target
        self.support = support_polygon

    def compute_jacobian(self, theta):
        # Jacobian relating joint velocities to ZMP velocity
        return self.jacobian_com(theta)

    def compute_error(self, theta, target):
        # Error: current ZMP - desired ZMP
        zmp_current = self._compute_zmp(theta)
        return self.zmp_target - zmp_current

    @staticmethod
    def _compute_zmp(theta):
        # Simplified: placeholder
        return np.array([0.0, 0.0])

class JointLimitTask(Task):
    """Task: keep joints within limits (low priority)"""

    def __init__(self, joint_limits, priority=2):
        super().__init__("Joint Limits", priority)
        self.limits = joint_limits
        self.n_joints = len(joint_limits)

    def compute_jacobian(self, theta):
        # Gradient of potential function near limits
        J = np.zeros((self.n_joints, self.n_joints))
        for i, (min_l, max_l) in enumerate(self.limits):
            # Repulsive gradient near limits
            mid = (min_l + max_l) / 2
            J[i, i] = self._limit_gradient(theta[i], min_l, max_l)
        return J

    @staticmethod
    def _limit_gradient(theta, min_lim, max_lim):
        """Compute gradient pushing away from limits"""
        margin = (max_lim - min_lim) * 0.1
        grad = 0
        if theta < min_lim + margin:
            grad = -(min_lim + margin - theta)
        elif theta > max_lim - margin:
            grad = (theta - (max_lim - margin))
        return grad

    def compute_error(self, theta, target):
        # Error: gradient pushing away from limits
        error = np.zeros(self.n_joints)
        for i, (min_l, max_l) in enumerate(self.limits):
            error[i] = self._limit_gradient(theta[i], min_l, max_l)
        return error
```

## Optimization-Based Control

An alternative to task hierarchy is formulating control as an optimization problem:

$$\min_{\dot{\theta}} \sum_i w_i \| e_i(\dot{\theta}) \|^2$$

Subject to:
- Joint velocity limits: $|\dot{\theta}_i| \leq v_{max}$
- Joint position limits: $\theta_{min} \leq \theta \leq \theta_{max}$
- Collision constraints: $d_{ij} \geq d_{min}$

```python
from scipy.optimize import minimize

class OptimizationBasedControl:
    """Whole-body control using quadratic programming"""

    def __init__(self, robot_model, tasks):
        self.robot = robot_model
        self.tasks = tasks

    def compute_joint_velocities(self, theta, dt=0.01):
        """
        Compute optimal joint velocities using constrained optimization
        """
        def objective(dtheta):
            """Sum of weighted task errors"""
            cost = 0
            for task in self.tasks:
                J = task.compute_jacobian(theta)
                e = task.compute_error(theta, task.target)
                residual = (J @ dtheta - e) if e is not None else (J @ dtheta)
                cost += task.weight * np.linalg.norm(residual)**2
            return cost

        # Velocity bounds
        v_max = 0.5  # rad/s
        bounds = [(-v_max, v_max) for _ in range(len(theta))]

        # Initial guess
        x0 = np.zeros(len(theta))

        # Solve
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6}
        )

        return result.x

    def execute_control_step(self, theta, dt):
        """Execute one control step"""
        dtheta = self.compute_joint_velocities(theta, dt)
        theta_new = theta + dtheta * dt

        # Enforce hard joint limits
        for i, (min_l, max_l) in enumerate(self.robot.joint_limits):
            theta_new[i] = np.clip(theta_new[i], min_l, max_l)

        return theta_new
```

## Inverse Dynamics with Constraints

For high-speed motion, we need to consider actual torques required:

$$\tau = M(\theta)\ddot{\theta} + C(\theta, \dot{\theta})\dot{\theta} + G(\theta)$$

Whole-body inverse dynamics computes required torques while satisfying constraints:

```python
class InverseDynamicsController:
    """Compute joint torques using inverse dynamics"""

    def __init__(self, mass_matrix_func, coriolis_func, gravity_func):
        """
        mass_matrix_func: theta -> M(theta)
        coriolis_func: (theta, dtheta) -> C(theta, dtheta)
        gravity_func: theta -> G(theta)
        """
        self.M = mass_matrix_func
        self.C = coriolis_func
        self.G = gravity_func

    def compute_torques(self, theta, dtheta, ddtheta_desired):
        """
        Compute required torques for desired acceleration
        Using inverse dynamics: τ = M*θ'' + C*θ' + G
        """
        M = self.M(theta)
        C = self.C(theta, dtheta)
        G = self.G(theta)

        tau = M @ ddtheta_desired + C @ dtheta + G

        return tau

    def trajectory_tracking_control(self, theta, dtheta, theta_desired, dtheta_desired,
                                   kp=100, kd=50):
        """
        Trajectory tracking with PD feedback on top of inverse dynamics
        """
        # Desired acceleration from PD control
        error = theta_desired - theta
        error_rate = dtheta_desired - dtheta

        ddtheta_desired = dtheta_desired + kp * error + kd * error_rate

        # Inverse dynamics
        tau = self.compute_torques(theta, dtheta, ddtheta_desired)

        return tau

# Example: mass matrix for 2-DoF arm
def simple_2dof_mass_matrix(theta):
    """
    Mass matrix for 2-DoF arm with equal link masses
    """
    m1 = m2 = 1.0  # Link masses (kg)
    l1 = l2 = 0.5  # Link lengths (m)
    Ic1 = Ic2 = 0.1  # Link moments of inertia

    c2 = np.cos(theta[1])

    M = np.array([
        [Ic1 + Ic2 + m1*(l1/2)**2 + m2*(l1**2 + (l2/2)**2 + 2*l1*(l2/2)*c2),
         Ic2 + m2*((l2/2)**2 + l1*(l2/2)*c2)],
        [Ic2 + m2*((l2/2)**2 + l1*(l2/2)*c2),
         Ic2 + m2*(l2/2)**2]
    ])

    return M

def simple_2dof_coriolis(theta, dtheta):
    """Coriolis and centrifugal terms"""
    m2 = 1.0
    l1 = l2 = 0.5
    g = 9.81

    s2 = np.sin(theta[1])

    C = np.array([
        [-m2 * l1 * (l2/2) * s2 * dtheta[1],
         -m2 * l1 * (l2/2) * s2 * (dtheta[0] + dtheta[1])],
        [m2 * l1 * (l2/2) * s2 * dtheta[0],
         0]
    ]) @ dtheta

    return C

def simple_2dof_gravity(theta):
    """Gravity terms"""
    m1 = m2 = 1.0
    l1 = l2 = 0.5
    g = 9.81

    c1 = np.cos(theta[0])
    c12 = np.cos(theta[0] + theta[1])

    G = np.array([
        g * (m1*(l1/2)*c1 + m2*(l1*c1 + (l2/2)*c12)),
        g * m2 * (l2/2) * c12
    ])

    return G

# Test inverse dynamics
id_controller = InverseDynamicsController(
    simple_2dof_mass_matrix,
    simple_2dof_coriolis,
    simple_2dof_gravity
)

theta = np.array([0.5, 0.3])
dtheta = np.array([0.1, -0.05])
ddtheta_desired = np.array([0.2, -0.1])

tau = id_controller.compute_torques(theta, dtheta, ddtheta_desired)
print(f"Required torques: {tau} N·m")
```

## Practical Implementation: Multi-Task Example

```python
class HumanoidManipulationController:
    """Complete whole-body controller for humanoid manipulation"""

    def __init__(self, robot_model):
        self.robot = robot_model
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: t.priority)

    def execute_grasp_task(self, target_pos, target_orientation, duration=5.0, dt=0.01):
        """
        Execute object grasping task maintaining balance
        """
        theta = self.robot.get_current_configuration()
        t = 0
        trajectory = []

        while t < duration:
            # Update task targets
            self.tasks[0].target_pos = target_pos  # End-effector
            self.tasks[1].zmp_target = np.array([0, 0])  # Balance

            # Compute control
            dtheta = self._compute_hierarchical_velocities(theta)

            # Integrate
            theta = theta + dtheta * dt
            t += dt

            trajectory.append(theta.copy())

        return np.array(trajectory)

    def _compute_hierarchical_velocities(self, theta):
        """Compute velocities using task hierarchy"""
        dtheta = np.zeros(len(theta))
        P = np.eye(len(theta))

        for task in self.tasks:
            if not task.active:
                continue

            J = task.compute_jacobian(theta)
            J_proj = J @ P

            # Damped pseudo-inverse
            U, s, Vt = svd(J_proj, full_matrices=False)
            s_damped = s / (s**2 + 0.001**2)
            J_pinv = Vt.T @ np.diag(s_damped) @ U.T

            e = task.compute_error(theta, task.target_pos)
            dtheta_task = P @ J_pinv @ e

            dtheta += dtheta_task
            P = (np.eye(len(theta)) - J_pinv @ J_proj) @ P

        return dtheta
```

## Exercise 7.1: Whole-Body Control Implementation

**Objective**: Implement multi-task control for a humanoid robot.

**Task**:
1. Model a 7-DoF arm attached to a mobile base
2. Define 3 tasks with priorities:
   - Priority 0: End-effector reaches target position
   - Priority 1: Center of mass stays above base
   - Priority 2: Joint 4 stays away from limits
3. Implement hierarchical inverse kinematics
4. Generate optimal joint velocities satisfying all constraints
5. Simulate for 5 seconds with:
   - Moving target for end-effector
   - Disturbance to base (simulating floor tilt)
6. Analyze:
   - Task error convergence
   - Joint velocity and acceleration profiles
   - Null-space utilization (how much joint 4 avoids limits)

**Submission**: Whole-body control report with:
- Hierarchical task decomposition
- Joint velocity profiles
- Task error over time
- Null-space manipulation demonstration

---

## Next Chapter Preview

Chapter 8 shifts focus from motion control to perception. You'll learn how humanoid robots sense and understand their environment using cameras, depth sensors, and other modalities to make intelligent decisions.

[→ Next: Chapter 8 - Humanoid Perception and Sensing](/docs/module2-humanoid-robotics/chapter8-perception)
