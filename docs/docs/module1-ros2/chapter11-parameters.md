---
sidebar_position: 12
title: Parameters and Launch Files
---

# Parameters and Launch Files

Launch files orchestrate complex multi-node systems, while parameters configure node behavior at runtime without code changes. Mastering these tools is essential for scalable, maintainable robotics systems.

## Understanding Parameters

Parameters are runtime configuration values that nodes can read and modify. They support type safety, dynamic reconfiguration, and hierarchical namespacing.

## Creating and Reading Parameters

Define and use parameters in nodes:

```python
# robot_config/robot_config/configurable_node.py
import rclpy
from rclpy.node import Node

class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')

        # Declare parameters with default values
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('debug_mode', False)

        # Get parameter values
        self.max_speed = self.get_parameter('max_speed').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.robot_name = self.get_parameter('robot_name').value
        self.debug_enabled = self.get_parameter('debug_mode').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max speed: {self.max_speed}')

        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.timer = self.create_timer(1.0, self.timer_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_speed':
                self.max_speed = param.value
                self.get_logger().info(f'Updated max_speed to {self.max_speed}')
            elif param.name == 'debug_mode':
                self.debug_enabled = param.value

        return rclpy.parameter_descriptors.SetParametersResult(successful=True)

    def timer_callback(self):
        if self.debug_enabled:
            self.get_logger().info(f'Current max speed: {self.max_speed}')

def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

## Parameter Configuration Files

Store parameters in YAML files for easy management:

```yaml
# robot_config/config/robot_params.yaml
robot_config_node:
  ros__parameters:
    max_speed: 2.5
    wheel_radius: 0.125
    robot_name: "robot_arm"
    debug_mode: true
    controller:
      p_gain: 1.0
      i_gain: 0.1
      d_gain: 0.05
    sensors:
      camera_enabled: true
      lidar_enabled: true
      imu_enabled: true

arm_controller:
  ros__parameters:
    joint_names: ["joint1", "joint2", "joint3"]
    home_position: [0.0, 0.0, 0.0]
    max_velocity: [1.57, 1.57, 1.57]
    timeout_ms: 1000
```

## Basic Launch File

Create a Python launch file to start multiple nodes:

```python
# robot_config/launch/robot_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('robot_config').find('robot_config')
    config_dir = os.path.join(pkg_share, 'config')

    # Declare command-line arguments
    launch_args = [
        DeclareLaunchArgument(
            'robot_name',
            default_value='robot_001',
            description='Name of the robot'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Enable debug mode'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
    ]

    # Load parameter file
    robot_config = os.path.join(config_dir, 'robot_params.yaml')

    # Define nodes
    config_node = Node(
        package='robot_config',
        executable='configurable_node',
        name='robot_node',
        namespace='robot',
        output='screen',
        parameters=[
            robot_config,
            {'robot_name': LaunchConfiguration('robot_name')},
            {'debug_mode': LaunchConfiguration('debug')},
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    tf_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        output='screen'
    )

    rvis_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription(launch_args + [
        config_node,
        tf_broadcaster,
        rvis_node,
    ])
```

## Advanced Launch Features

Implement conditional execution and remapping:

```python
# robot_config/launch/multi_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('robot_config').find('robot_config')

    # Launch arguments
    args = [
        DeclareLaunchArgument('num_robots', default_value='2'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
    ]

    nodes = []

    # Conditionally launch nav2
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'nav2.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_nav2'))
    )

    # Launch robot nodes with namespace
    num_robots = LaunchConfiguration('num_robots')

    for i in range(2):  # Example: 2 robots
        robot_node = Node(
            package='robot_config',
            executable='configurable_node',
            name=f'robot_{i}',
            namespace=f'robot_{i}',
            parameters=[
                {'robot_name': f'robot_{i}'},
                {'max_speed': 2.0 if i == 0 else 1.5},
            ],
            remappings=[
                ('/scan', f'/robot_{i}/scan'),
                ('/cmd_vel', f'/robot_{i}/cmd_vel'),
            ],
            output='screen'
        )
        nodes.append(robot_node)

    return LaunchDescription(args + nodes + [nav2_launch])
```

## Updating Parameters at Runtime

Modify parameters without restarting nodes:

```python
# robot_config/robot_config/param_updater.py
import rclpy
from rclpy.node import Node

class ParameterUpdater(Node):
    def __init__(self):
        super().__init__('param_updater')

    def update_remote_parameter(self, node_name, param_name, value):
        """Update parameter in another node"""
        client = self.create_client(
            rclpy.parameter_client.SetParametersClient,
            f'/{node_name}'
        )

        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter, ParameterValue

        request = SetParameters.Request()
        param = Parameter()
        param.name = param_name
        param.value = ParameterValue(double_value=float(value))
        request.parameters = [param]

        future = client.call_async(request)
        future.add_done_callback(
            lambda f: self.get_logger().info(f'Parameter update result: {f.result()}')
        )

def main(args=None):
    rclpy.init(args=args)
    updater = ParameterUpdater()

    # Example: Update max_speed on robot_node
    updater.update_remote_parameter('robot', 'max_speed', 3.0)

    rclpy.spin(updater)

if __name__ == '__main__':
    main()
```

## CLI Commands

Manage parameters from the command line:

```bash
# List all parameters
ros2 param list

# Get a parameter value
ros2 param get /robot_node max_speed

# Set a parameter
ros2 param set /robot_node max_speed 3.0

# Load parameters from YAML file
ros2 param load /robot_node robot_params.yaml

# Dump all parameters to file
ros2 param dump /robot_node > robot_state.yaml

# Launch with command-line parameter override
ros2 launch robot_config robot_system.launch.py \
  robot_name:=my_special_robot \
  debug:=true \
  use_sim_time:=false
```

## Exercise

**Task**: Create a complete launch system that:
1. Defines a parameter YAML file with robot kinematics (link lengths, max velocities)
2. Creates a launch file that loads parameters and starts a node
3. Implements a node that reads parameters and logs them at startup
4. Supports command-line overrides for at least 3 parameters
5. Includes a parameter update service that other nodes can call

**Acceptance Criteria**:
- YAML configuration file is properly formatted and loaded
- Launch file runs without errors with default parameters
- Parameters can be overridden via command line
- Node successfully reads and logs all parameters
- Parameter changes are reflected in node behavior
- Test with: `ros2 launch robot_config robot_system.launch.py param1:=value1`

---

**Next**: [Chapter 12 - Best Practices](/docs/docs/module1-ros2/chapter12-best-practices)
