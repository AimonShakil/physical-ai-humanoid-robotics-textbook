---
sidebar_position: 6
title: Chapter 5 - Publishers & Subscribers
---

# Chapter 5: Publishers & Subscribers

## Topic Communication

Topics are the primary way nodes exchange data in ROS 2. They use a publish-subscribe pattern for asynchronous, one-to-many communication.

## Message Types

Common message types:

```python
from std_msgs.msg import String, Int32, Float64, Bool
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image, LaserScan, Imu
```

## Advanced Publisher

```python
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_robot)  # 10Hz

    def move_robot(self):
        msg = Twist()
        msg.linear.x = 0.5  # Forward 0.5 m/s
        msg.angular.z = 0.2  # Turn 0.2 rad/s
        self.cmd_vel_pub.publish(msg)
```

## Advanced Subscriber

```python
from sensor_msgs.msg import LaserScan

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

    def scan_callback(self, msg):
        min_distance = min(msg.ranges)
        if min_distance < 0.5:  # Obstacle within 0.5m
            self.get_logger().warn(f'Obstacle detected: {min_distance}m')
```

## QoS Policies

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Sensor data: best effort, transient local
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=10
)

# Control commands: reliable
control_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    depth=10
)

self.sub = self.create_subscription(LaserScan, 'scan', callback, sensor_qos)
```

## Custom Messages

1. Create message file `msg/CustomMsg.msg`:
```
string name
int32 id
float64[] values
```

2. Update `CMakeLists.txt` and `package.xml`

3. Use in code:
```python
from my_package.msg import CustomMsg

msg = CustomMsg()
msg.name = "sensor_1"
msg.id = 42
msg.values = [1.0, 2.0, 3.0]
```

## CLI Tools

```bash
# List topics
ros2 topic list

# Echo topic
ros2 topic echo /cmd_vel

# Info
ros2 topic info /scan

# Hz
ros2 topic hz /scan

# Pub from CLI
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"
```

**Exercise**: Create node that subscribes to `/scan` and publishes `/cmd_vel` to avoid obstacles.

[Next: Services](/docs/module1-ros2/chapter6-services)
