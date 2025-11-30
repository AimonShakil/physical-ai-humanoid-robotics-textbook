---
sidebar_position: 9
title: Working with Sensors
---

# Working with Sensors

Sensors are the eyes and ears of a robot. ROS 2 provides standardized message types for common sensors like cameras, LiDAR, IMU (Inertial Measurement Units), and distance sensors. Understanding how to interface with sensors, process their data, and publish meaningful information is fundamental to robotics development.

## Common Sensor Message Types

ROS 2 defines standard message types for different sensor modalities:

- **Camera**: `sensor_msgs/Image`, `sensor_msgs/CameraInfo`
- **LiDAR**: `sensor_msgs/LaserScan`, `sensor_msgs/PointCloud2`
- **IMU**: `sensor_msgs/Imu`
- **Distance**: `sensor_msgs/Range`
- **Joint State**: `sensor_msgs/JointState`

## Publishing Camera Data

This example reads images from a camera and publishes them to ROS 2:

```python
# robot_sensors/robot_sensors/camera_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

        # Camera calibration parameters
        self.camera_info = CameraInfo()
        self.camera_info.width = 640
        self.camera_info.height = 480
        self.camera_info.K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize to standard resolution
            frame = cv2.resize(frame, (640, 480))
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = 'camera_optical_frame'

            self.image_pub.publish(image_msg)

            self.camera_info.header = image_msg.header
            self.info_pub.publish(self.camera_info)

def main(args=None):
    rclpy.init(args=args)
    publisher = CameraPublisher()
    rclpy.spin(publisher)

if __name__ == '__main__':
    main()
```

## Subscribing to Sensor Data

Process incoming sensor data in real-time:

```python
# robot_sensors/robot_sensors/imu_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class ImuSubscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)

        self.total_acceleration = 0.0
        self.samples = 0

    def imu_callback(self, msg):
        # Extract acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        # Calculate magnitude
        acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        self.total_acceleration += acc_magnitude
        self.samples += 1

        # Extract angular velocity
        roll_rate = msg.angular_velocity.x
        pitch_rate = msg.angular_velocity.y
        yaw_rate = msg.angular_velocity.z

        self.get_logger().info(
            f'Accel: {acc_magnitude:.2f} m/s², '
            f'Rotation: {yaw_rate:.2f} rad/s'
        )

        # Extract orientation (quaternion)
        q = msg.orientation
        self.get_logger().debug(
            f'Quaternion: x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}, w={q.w:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    subscriber = ImuSubscriber()
    rclpy.spin(subscriber)

if __name__ == '__main__':
    main()
```

## LiDAR Data Processing

Work with point cloud data from LiDAR sensors:

```python
# robot_sensors/robot_sensors/lidar_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.obstacle_pub = self.create_publisher(
            LaserScan,
            '/obstacles_detected',
            10)
        self.obstacle_threshold = 1.5  # meters

    def scan_callback(self, msg):
        # Filter obstacles closer than threshold
        filtered_ranges = []
        for i, range_val in enumerate(msg.ranges):
            if 0 < range_val < self.obstacle_threshold:
                filtered_ranges.append(range_val)
            elif range_val >= self.obstacle_threshold or range_val == 0:
                filtered_ranges.append(float('inf'))

        # Find closest obstacle
        valid_ranges = [r for r in filtered_ranges if r != float('inf')]
        if valid_ranges:
            closest = min(valid_ranges)
            angle_idx = filtered_ranges.index(closest)
            angle = msg.angle_min + angle_idx * msg.angle_increment

            self.get_logger().warn(
                f'Obstacle detected: {closest:.2f}m at {np.degrees(angle):.1f}°'
            )

        # Publish filtered scan
        filtered_scan = msg
        filtered_scan.ranges = filtered_ranges
        self.obstacle_pub.publish(filtered_scan)

def main(args=None):
    rclpy.init(args=args)
    processor = LidarProcessor()
    rclpy.spin(processor)

if __name__ == '__main__':
    main()
```

## CLI Commands

Inspect and monitor sensor data:

```bash
# List all active topics
ros2 topic list

# View sensor data stream
ros2 topic echo /imu/data

# Monitor camera frames
ros2 topic echo /camera/image_raw --print-all-fields

# Get topic publication rate
ros2 topic hz /imu/data

# Inspect message structure
ros2 interface show sensor_msgs/msg/Imu
```

## Exercise

**Task**: Create a sensor fusion node that subscribes to both IMU and LiDAR data, calculates when the robot is moving forward (positive acceleration) and has no obstacles ahead (LiDAR > 1.0m), then publishes a `std_msgs/Bool` message indicating if it's safe to move forward.

**Acceptance Criteria**:
- Subscribes to both `/imu/data` and `/scan` topics
- Publishes `Bool` message to `/safe_to_move` topic
- Correctly identifies forward motion from IMU acceleration
- Correctly identifies clear path from LiDAR data
- Updates at minimum 10 Hz

---

**Next**: [Chapter 9 - Robot Simulation with Gazebo](/docs/docs/module1-ros2/chapter9-gazebo)
