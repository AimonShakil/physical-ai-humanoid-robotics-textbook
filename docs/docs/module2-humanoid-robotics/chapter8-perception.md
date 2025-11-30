---
sidebar_position: 9
title: Chapter 8 - Humanoid Perception and Sensing
---

# Humanoid Perception and Sensing

## Overview

A humanoid robot is only as capable as its ability to perceive and understand its environment. While motion control determines how a robot moves, perception systems determine what the robot needs to do. This chapter explores the sensor types used in humanoid robots, data fusion techniques, and algorithms for extracting meaningful information from raw sensor data. We'll cover vision, inertial measurement, force/torque sensing, and multi-modal fusion strategies.

## Sensor Types and Modalities

### 1. Visual Sensors

Humanoid robots typically use multiple cameras for different purposes:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraSpecification:
    """Camera sensor specification"""
    name: str
    resolution: tuple  # (width, height) in pixels
    fov: float  # Field of view in degrees
    frame_rate: float  # Hz
    focal_length: float  # mm
    sensor_size: tuple  # (width, height) in mm

    @property
    def focal_length_pixels(self):
        """Convert focal length to pixels"""
        return self.focal_length / (self.sensor_size[0] / self.resolution[0])

    def project_3d_to_2d(self, point_3d):
        """
        Project 3D point to image plane using pinhole camera model
        point_3d: (x, y, z) in meters
        Returns: (u, v) in pixels, or None if behind camera
        """
        x, y, z = point_3d

        if z <= 0:
            return None  # Behind camera

        # Principal point (image center)
        cx, cy = self.resolution[0] / 2, self.resolution[1] / 2

        # Focal length in pixels
        f = self.focal_length_pixels

        # Project
        u = f * (x / z) + cx
        v = f * (y / z) + cy

        # Check bounds
        if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
            return (u, v)
        return None

# NAO camera specifications
nao_front_camera = CameraSpecification(
    name="Front Camera",
    resolution=(640, 480),
    fov=60.97,
    frame_rate=30,
    focal_length=2.8,
    sensor_size=(2.8, 2.1)
)

# Test projection
point_world = np.array([0.5, 0.1, 1.5])  # (x, y, z) in meters
pixel = nao_front_camera.project_3d_to_2d(point_world)
print(f"World point {point_world} projects to pixel {pixel}")
```

### 2. Depth Sensors

Structured light, time-of-flight, and stereo depth sensors provide 3D information:

```python
class DepthSensorModel:
    """Model of depth sensor (e.g., RGB-D camera)"""

    def __init__(self, resolution, fov, min_range, max_range, noise_std=0.01):
        """
        resolution: (width, height) in pixels
        fov: field of view in degrees
        min_range, max_range: depth range in meters
        noise_std: depth measurement noise (standard deviation)
        """
        self.resolution = resolution
        self.fov = fov
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.focal_length = self._compute_focal_length()

    def _compute_focal_length(self):
        """Compute focal length from FOV and resolution"""
        return self.resolution[0] / (2 * np.tan(np.radians(self.fov / 2)))

    def generate_point_cloud(self, depth_image, rgb_image=None):
        """
        Convert depth image to 3D point cloud
        depth_image: (height, width) array of depth values in meters
        rgb_image: optional (height, width, 3) RGB image
        Returns: point cloud as (N, 3) array of (x, y, z) coordinates
        """
        height, width = depth_image.shape
        cy, cx = height / 2, width / 2

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Backproject to 3D
        x = (u - cx) * depth_image / self.focal_length
        y = (v - cy) * depth_image / self.focal_length
        z = depth_image

        # Stack coordinates
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Remove invalid points (depth out of range)
        valid = (z.flatten() > self.min_range) & (z.flatten() < self.max_range)
        points = points[valid]

        return points

    def simulate_measurement(self, world_points, camera_pose):
        """
        Simulate depth measurement from world points
        camera_pose: (x, y, z) position, (roll, pitch, yaw) orientation
        Returns: depth image
        """
        # Transform world points to camera frame
        # (Simplified: assume camera at origin looking down -Z axis)
        depth_image = np.zeros(self.resolution)

        for point in world_points:
            z = point[2]  # Depth

            if z < self.min_range or z > self.max_range:
                continue

            # Project to image
            x, y = point[0], point[1]
            u = int(self.focal_length * x / z + self.resolution[0] / 2)
            v = int(self.focal_length * y / z + self.resolution[1] / 2)

            # Check bounds
            if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
                # Add measurement noise
                noisy_depth = z + np.random.normal(0, self.noise_std)
                depth_image[v, u] = noisy_depth

        return depth_image

# Example: Pepper's depth sensor
pepper_depth = DepthSensorModel(
    resolution=(640, 480),
    fov=58,
    min_range=0.3,
    max_range=4.0,
    noise_std=0.02
)

# Simulate point cloud from depth image
depth_image = np.random.uniform(0.5, 3.0, (480, 640))
point_cloud = pepper_depth.generate_point_cloud(depth_image)
print(f"Generated point cloud with {len(point_cloud)} points")
print(f"Depth range: {point_cloud[:, 2].min():.2f} - {point_cloud[:, 2].max():.2f} m")
```

### 3. Inertial Measurement Unit (IMU)

Measures acceleration and angular velocity:

```python
class InertialMeasurementUnit:
    """6-DoF IMU: accelerometer + gyroscope"""

    def __init__(self, accel_range=2.0, gyro_range=250.0,
                 accel_noise=0.001, gyro_noise=0.001):
        """
        accel_range: maximum measurable acceleration (g)
        gyro_range: maximum measurable angular velocity (deg/s)
        accel_noise, gyro_noise: measurement noise (std dev)
        """
        self.accel_range = accel_range * 9.81  # Convert to m/s²
        self.gyro_range = np.radians(gyro_range)  # Convert to rad/s
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.gravity = np.array([0, 0, 9.81])

    def measure(self, true_accel, true_gyro, orientation):
        """
        Generate noisy measurements
        true_accel: true acceleration in world frame
        true_gyro: true angular velocity
        orientation: current orientation (rotation matrix)
        Returns: (accel_measurement, gyro_measurement)
        """
        # Add gravity to acceleration (accelerometer measures
        # linear accel + gravitational effect)
        # In body frame: a_body = R^T * (a_true - g)
        accel_with_gravity = true_accel - self.gravity
        accel_body = orientation.T @ accel_with_gravity

        # Add noise
        accel_meas = accel_body + np.random.normal(0, self.accel_noise, 3)
        gyro_meas = true_gyro + np.random.normal(0, self.gyro_noise, 3)

        # Saturate to sensor range
        accel_meas = np.clip(accel_meas, -self.accel_range, self.accel_range)
        gyro_meas = np.clip(gyro_meas, -self.gyro_range, self.gyro_range)

        return accel_meas, gyro_meas

    def estimate_orientation(self, accel_meas, gyro_meas, dt, prev_orientation):
        """
        Estimate robot orientation from IMU measurements
        Using complementary filter approach
        """
        # Extract tilt from accelerometer (gravity direction)
        accel_norm = np.linalg.norm(accel_meas)
        if accel_norm > 0:
            gravity_est = accel_meas / accel_norm

            # Angle between measured gravity and body z-axis
            tilt = np.arccos(np.clip(gravity_est[2], -1, 1))
        else:
            tilt = 0

        # Integrate gyro to get rotation
        # Simple first-order integration (Euler method)
        delta_rotation = np.eye(3) + self._skew(gyro_meas) * dt
        orientation_new = prev_orientation @ delta_rotation

        # Complementary filter: blend gravity-based and gyro-based estimates
        alpha = 0.95  # Weight of gyro integration
        # (Simplified: would need proper quaternion/rotation matrix blending)

        return orientation_new

    @staticmethod
    def _skew(v):
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

# Test IMU
imu = InertialMeasurementUnit()

# Simulate falling robot
true_accel = np.array([0, 0, -5])  # Falling at 5 m/s²
true_gyro = np.array([0, 0, 0])
orientation = np.eye(3)

accel_meas, gyro_meas = imu.measure(true_accel, true_gyro, orientation)
print(f"Measured acceleration: {accel_meas} m/s²")
print(f"Measured angular velocity: {gyro_meas} rad/s")
```

### 4. Force/Torque Sensors

Measure forces and torques at contact points:

```python
class ForceTorqueSensor:
    """6-axis F/T sensor (typically at wrist or foot)"""

    def __init__(self, force_range=1000, torque_range=100,
                 force_noise=1.0, torque_noise=0.1):
        """
        force_range: maximum measurable force (N)
        torque_range: maximum measurable torque (N·m)
        """
        self.force_range = force_range
        self.torque_range = torque_range
        self.force_noise = force_noise
        self.torque_noise = torque_noise

    def measure(self, true_force, true_torque):
        """
        Measure force and torque with noise
        true_force: (fx, fy, fz) in N
        true_torque: (tx, ty, tz) in N·m
        """
        # Add noise
        force_meas = true_force + np.random.normal(0, self.force_noise, 3)
        torque_meas = true_torque + np.random.normal(0, self.torque_noise, 3)

        # Saturate
        force_norm = np.linalg.norm(force_meas)
        if force_norm > self.force_range:
            force_meas = force_meas * self.force_range / force_norm

        torque_norm = np.linalg.norm(torque_meas)
        if torque_norm > self.torque_range:
            torque_meas = torque_meas * self.torque_range / torque_norm

        return force_meas, torque_meas

    def detect_contact(self, force_meas, threshold=5.0):
        """
        Detect if object in contact
        threshold: minimum force magnitude to indicate contact (N)
        """
        force_magnitude = np.linalg.norm(force_meas)
        return force_magnitude > threshold

    def estimate_contact_location(self, force, torque, sensor_offset=np.array([0, 0, 0])):
        """
        Estimate contact point location given F/T measurement
        Using relationship: τ = r × F
        r: position of contact relative to sensor
        """
        # Solve r × F = τ for r
        # Simplified: assume contact on Z=0 plane
        F_xy = force[:2]
        F_z = force[2]

        if np.abs(F_z) < 0.1:  # No vertical force
            return None

        # Contact point
        r_x = -torque[1] / F_z if F_z != 0 else 0
        r_y = torque[0] / F_z if F_z != 0 else 0

        return np.array([r_x, r_y, 0]) + sensor_offset

# Example: foot F/T sensor
foot_ft = ForceTorqueSensor()

# Simulate robot standing with uneven weight
ground_forces = np.array([10, 5, 400])  # Forward-left bias
ground_torques = np.array([1, -0.5, 0])

force_meas, torque_meas = foot_ft.measure(ground_forces, ground_torques)
is_contact = foot_ft.detect_contact(force_meas)
contact_loc = foot_ft.estimate_contact_location(force_meas, torque_meas)

print(f"Measured force: {force_meas} N")
print(f"Contact detected: {is_contact}")
if contact_loc is not None:
    print(f"Contact location: {contact_loc} m")
```

## Sensor Fusion

Combining multiple sensor modalities improves robustness:

```python
class SensorFusion:
    """Multi-sensor fusion for state estimation"""

    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.meas_dim = measurement_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)

    def kalman_filter_update(self, z, H, R, Q, F):
        """
        Kalman filter prediction and update
        z: measurement
        H: measurement matrix
        R: measurement noise covariance
        Q: process noise covariance
        F: state transition matrix
        """
        # Prediction
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

        # Update
        y = z - H @ self.state  # Innovation
        S = H @ self.covariance @ H.T + R  # Innovation covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance

        return self.state, self.covariance

class OdometryFusion:
    """Fuse IMU and visual odometry for localization"""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.eye(3)
        self.velocity = np.array([0.0, 0.0, 0.0])

    def integrate_imu(self, accel, gyro, dt):
        """Integrate IMU measurements"""
        # Update orientation from gyro
        gyro_mag = np.linalg.norm(gyro)
        if gyro_mag > 1e-6:
            rotation_axis = gyro / gyro_mag
            rotation_angle = gyro_mag * dt
            # Rodrigues rotation formula
            K = self._skew(rotation_axis)
            R_delta = (np.eye(3) + np.sin(rotation_angle) * K +
                      (1 - np.cos(rotation_angle)) * K @ K)
            self.orientation = self.orientation @ R_delta

        # Update velocity and position from acceleration
        accel_world = self.orientation @ accel
        self.velocity += accel_world * dt
        self.position += self.velocity * dt

    def update_visual_odometry(self, visual_displacement, confidence=1.0):
        """Update from visual odometry with confidence weighting"""
        # Blend visual update with previous state
        alpha = confidence * 0.5  # Visual confidence factor
        self.position += alpha * visual_displacement

    @staticmethod
    def _skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

# Test fusion
fusion = OdometryFusion()

for step in range(10):
    accel = np.array([0.1, 0, -9.81])  # Small forward acceleration, gravity
    gyro = np.array([0, 0, 0.01])  # Slight turning
    fusion.integrate_imu(accel, gyro, 0.1)

print(f"Estimated position: {fusion.position}")
print(f"Estimated orientation trace: {np.trace(fusion.orientation):.3f}")
```

## Exercise 8.1: Perception System Design

**Objective**: Build and test a sensor fusion system for humanoid perception.

**Task**:
1. Implement camera projection model for front and bottom cameras
2. Implement depth sensor point cloud generation
3. Implement IMU measurement simulation with noise
4. Implement F/T sensor contact detection
5. Create sensor fusion system combining:
   - IMU for orientation and acceleration
   - Visual odometry (simulated feature tracking)
   - F/T sensors for contact feedback
6. Simulate 10 seconds of walking with:
   - Ground truth: sinusoidal forward motion with small lateral sway
   - Sensor noise and delays
   - Foot contact changes during swing/stance phases
7. Fuse all sensor data to estimate:
   - Robot position and orientation
   - Velocity
   - Contact state (which foot on ground)
8. Compare fused estimate to ground truth

**Submission**: Sensor fusion report with:
- Camera projection visualization
- Point cloud examples
- IMU noise characteristics
- Fusion algorithm block diagram
- Position/orientation estimation error over time

---

## Next Chapter Preview

Chapter 9 addresses human-robot interaction—how humanoids can safely collaborate with humans, interpret human gestures, and communicate effectively. This is crucial for robots in human environments.

[→ Next: Chapter 9 - Human-Robot Interaction](/docs/module2-humanoid-robotics/chapter9-human-robot-interaction)
