---
sidebar_position: 11
title: Navigation and SLAM
---

# Navigation and SLAM

Navigation is one of the most complex challenges in robotics. SLAM (Simultaneous Localization and Mapping) allows a robot to explore unknown environments while building a map and tracking its position. The ROS 2 Navigation Stack provides production-ready components for these tasks.

## SLAM Overview

SLAM solves two problems simultaneously:
1. **Localization**: Where am I?
2. **Mapping**: What does the environment look like?

Common SLAM algorithms include Cartographer (2D/3D) and RTAB-Map (RGB-D). They consume sensor data (LiDAR, cameras, IMU) and produce occupancy grids and pose estimates.

## Running Cartographer SLAM

Launch Cartographer to build a map from your robot's sensors:

```bash
# Install Cartographer
sudo apt install ros-humble-cartographer-ros

# Launch Cartographer with your robot's configuration
ros2 launch cartographer_ros cartographer.launch.py \
  use_sim_time:=true \
  configuration_directory:=/path/to/config \
  configuration_basename:=your_robot.lua
```

## Cartographer Configuration

Configure Cartographer for your sensor setup:

```lua
-- your_robot.lua
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-2,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_collate_multimesages = false,
}

MAP_BUILDER.use_trajectory_builder_2d = true
TRAJECTORY_BUILDER_2D.use_imu_data = true

return options
```

## Navigation Stack Setup

Create a navigation launch file that sets up path planning and costmaps:

```python
# robot_nav/launch/navigation.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('robot_nav').find('robot_nav')

    return LaunchDescription([
        # Map server - provides static map
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(pkg_share, 'maps', 'my_map.yaml')
            }]
        ),

        # Amcl - localization using particle filter
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(pkg_share, 'config', 'amcl_config.yaml')
            }]
        ),

        # Nav2 planner
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(pkg_share, 'config', 'planner_config.yaml')
            }]
        ),

        # Nav2 controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(pkg_share, 'config', 'controller_config.yaml')
            }]
        ),

        # Navigation behavior tree
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(pkg_share, 'config', 'nav2_bt_config.yaml')
            }]
        ),
    ])
```

## Navigation Client

Create a client that sends navigation goals to the Nav2 stack:

```python
# robot_nav/robot_nav/nav_client.py
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

class NavClient(Node):
    def __init__(self):
        super().__init__('nav_client')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)

    def send_nav_goal(self):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set goal position
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.position.z = 0.0

        # Set goal orientation (facing forward)
        q = quaternion_from_euler(0, 0, 0)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        distance = feedback.distance_remaining
        self.get_logger().info(f'Distance to goal: {distance:.2f}m')

    def get_result_callback(self, future):
        result = future.result().result
        if result.result_code == 0:
            self.get_logger().info('Navigation succeeded!')
        else:
            self.get_logger().warn(f'Navigation failed with code {result.result_code}')

def main(args=None):
    rclpy.init(args=args)
    client = NavClient()
    client.send_nav_goal()
    rclpy.spin(client)

if __name__ == '__main__':
    main()
```

## Building an Occupancy Grid Map

Convert sensor data into a map representation:

```python
# robot_nav/robot_nav/map_builder.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapBuilder(Node):
    def __init__(self):
        super().__init__('map_builder')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Grid parameters
        self.grid_size = 100
        self.resolution = 0.05  # meters per cell
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))

    def scan_callback(self, msg):
        # Update grid based on laser scan
        center = self.grid_size // 2

        for i, range_val in enumerate(msg.ranges):
            if 0 < range_val < 10:
                angle = msg.angle_min + i * msg.angle_increment
                x = int(center + (range_val * np.cos(angle)) / self.resolution)
                y = int(center + (range_val * np.sin(angle)) / self.resolution)

                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.occupancy_grid[y, x] = 100  # Occupied

        # Publish map
        self.publish_occupancy_grid()

    def publish_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.data = self.occupancy_grid.flatten().astype(np.int8).tolist()
        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    builder = MapBuilder()
    rclpy.spin(builder)

if __name__ == '__main__':
    main()
```

## CLI Commands

Useful navigation commands:

```bash
# Run SLAM and save map when complete
ros2 run nav2_map_server map_saver_cli -f ~/my_map

# View current map
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=my_map.yaml

# Estimate initial pose interactively (in RViz)
rviz2 -d nav2_bringup/rviz/tb3_nav2_default_view.rviz

# Send navigation goal via command line
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 2.0}}}}"
```

## Exercise

**Task**: Create a SLAM and navigation pipeline that:
1. Launches Cartographer to build a map from simulated LiDAR data
2. Once map is built, switch to Nav2 for localization
3. Create a client node that sends the robot to three waypoints in sequence

**Acceptance Criteria**:
- Cartographer successfully builds map from /scan topic
- Map saved and can be reloaded
- AMCL localizes robot on the map
- Navigation client successfully navigates between waypoints
- Each waypoint is reached within 0.5m tolerance

---

**Next**: [Chapter 11 - Parameters and Launch Files](/docs/docs/module1-ros2/chapter11-parameters)
