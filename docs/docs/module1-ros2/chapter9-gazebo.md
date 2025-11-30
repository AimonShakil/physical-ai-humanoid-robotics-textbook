---
sidebar_position: 10
title: Robot Simulation with Gazebo
---

# Robot Simulation with Gazebo

Gazebo is the de facto standard simulator for ROS 2 robotics. It provides physics simulation, realistic sensor simulation, and complex multi-robot environments. Developing and testing robot behaviors in simulation before hardware deployment saves time, money, and reduces risk.

## Understanding URDF and Gazebo

A robot's structure is defined in URDF (Unified Robot Description Format) files. Gazebo loads URDF files and adds physics properties. The workflow is: design robot → create URDF → add Gazebo plugins → test in simulation.

## Creating a Simple Robot URDF

This URDF defines a simple mobile robot with a base, two wheels, and a caster:

```xml
<?xml version="1.0" ?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.4 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.4 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.08" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.08" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin rpy="0 0 0" xyz="-0.15 0.15 0"/>
    <limit effort="10" velocity="5"/>
  </joint>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.08" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.08" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin rpy="0 0 0" xyz="-0.15 -0.15 0"/>
    <limit effort="10" velocity="5"/>
  </joint>

</robot>
```

## Gazebo Launch File

Launch the robot in Gazebo with proper configuration:

```python
# robot_sim/launch/gazebo.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare('robot_sim').find('robot_sim')
    urdf_path = os.path.join(pkg_share, 'urdf', 'simple_robot.urdf')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'simple_robot'],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot
    ])
```

## Gazebo Plugins for Physics and Control

Add plugins to simulate motors and sensors:

```xml
<!-- Add to URDF inside robot tag -->
<gazebo>
  <plugin name='gazebo_ros_control' filename='libgazebo_ros_control.so'/>
</gazebo>

<!-- Wheel friction properties -->
<gazebo reference="left_wheel">
  <mu1>0.5</mu1>
  <mu2>0.5</mu2>
  <kp>1e6</kp>
  <kd>100</kd>
</gazebo>

<gazebo reference="right_wheel">
  <mu1>0.5</mu1>
  <mu2>0.5</mu2>
  <kp>1e6</kp>
  <kd>100</kd>
</gazebo>
```

## Creating a Robot Controller

Write a Python node that publishes joint commands to control the simulated robot:

```python
# robot_sim/robot_sim/gazebo_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/simple_robot/joint_velocity_controller/commands',
            10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.step = 0

    def timer_callback(self):
        msg = Float64MultiArray()

        # Oscillate wheel velocities
        if self.step < 50:
            msg.data = [2.0, 2.0]  # Move forward
        elif self.step < 100:
            msg.data = [2.0, -2.0]  # Turn right
        else:
            self.step = 0

        self.pub.publish(msg)
        self.step += 1

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()
    rclpy.spin(controller)

if __name__ == '__main__':
    main()
```

## CLI Commands

Control Gazebo from the command line:

```bash
# Launch Gazebo with empty world
gazebo

# Launch with specific world file
gazebo worlds/empty.world

# Run Gazebo headless (no GUI, faster)
gazebo --headless -s libgazebo_ros_init.so -s libgazebo_ros_factory.so

# List spawned models
ros2 service call /gazebo/get_model_list gazebo_msgs/srv/GetModelList

# Pause simulation
ros2 service call /gazebo/pause_physics std_srvs/srv/Empty

# Unpause simulation
ros2 service call /gazebo/unpause_physics std_srvs/srv/Empty

# Delete a model
ros2 service call /gazebo/delete_model gazebo_msgs/srv/DeleteModel '{model_name: simple_robot}'
```

## Exercise

**Task**: Create a Gazebo world with a simple mobile robot that has two differential-drive wheels. Launch it in Gazebo and create a controller node that makes the robot drive in a square pattern (forward 10 steps, turn right 5 steps, repeat 4 times).

**Acceptance Criteria**:
- URDF file defines robot with two wheels and proper inertia
- Gazebo launches successfully with robot spawned
- Controller node publishes to correct joint velocity topic
- Robot completes full square pattern in simulation
- Total simulation runtime under 30 seconds

---

**Next**: [Chapter 10 - Navigation and SLAM](/docs/docs/module1-ros2/chapter10-navigation)
