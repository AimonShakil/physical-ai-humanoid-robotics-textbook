---
sidebar_position: 5
title: Chapter 4 - Creating Your First Node
---

# Chapter 4: Creating Your First Node

## Creating a Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_controller
```

## Publisher Node

`my_robot_controller/publisher_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'robot_news', 10)
        self.timer = self.create_timer(1.0, self.publish_news)

    def publish_news(self):
        msg = String()
        msg.data = f'Robot update'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

## Subscriber Node

`my_robot_controller/subscriber_node.py`:

```python
class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String, 'robot_news', self.callback, 10)

    def callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')
```

## Setup & Run

Edit `setup.py`:
```python
entry_points={
    'console_scripts': [
        'publisher = my_robot_controller.publisher_node:main',
        'subscriber = my_robot_controller.subscriber_node:main',
    ],
},
```

Build and run:
```bash
colcon build --packages-select my_robot_controller
source install/setup.bash
ros2 run my_robot_controller publisher
```

**Exercise**: Create temperature sensor publisher/subscriber pair.

[Next: Chapter 5](/docs/module1-ros2/chapter5-pub-sub)
