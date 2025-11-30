---
sidebar_position: 8
title: Actions and Action Servers
---

# Actions and Action Servers

Actions in ROS 2 provide a mechanism for long-running, goal-oriented tasks that require feedback during execution. Unlike services (which are request-response), actions allow clients to send goals, receive periodic feedback, and handle cancellation. This is ideal for operations like robot arm movements, navigation, or object manipulation that take significant time to complete.

## Understanding Actions

An action consists of three main parts: a goal (sent to the server), feedback (periodic updates), and a result (final outcome). Actions use a goal ID to track requests, allowing the client to cancel or monitor progress independently.

## Action Definition

Actions are defined using `.action` files with three message sections:

```yaml
# robot_actions/action/MoveArm.action
geometry_msgs/Pose target_pose
float32 speed
---
geometry_msgs/Pose current_pose
float32 progress
---
bool success
string message
```

## Action Server Implementation

Here's a complete action server that simulates arm movement:

```python
# robot_actions/robot_actions/arm_server.py
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from robot_actions.action import MoveArm
import time

class ArmActionServer(Node):
    def __init__(self):
        super().__init__('arm_action_server')
        self._action_server = ActionServer(
            self,
            MoveArm,
            'move_arm',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'Executing goal: {goal_handle.goal_id}')

        goal = goal_handle.goal
        feedback = MoveArm.Feedback()
        feedback.progress = 0.0

        # Simulate arm movement
        for i in range(11):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return MoveArm.Result()

            feedback.progress = float(i) / 10.0
            feedback.current_pose = goal.target_pose
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(f'Progress: {feedback.progress:.1%}')
            time.sleep(0.5)

        goal_handle.succeed()
        result = MoveArm.Result()
        result.success = True
        result.message = 'Arm movement completed'
        return result

def main(args=None):
    rclpy.init(args=args)
    server = ArmActionServer()
    rclpy.spin(server)

if __name__ == '__main__':
    main()
```

## Action Client Implementation

A client sends goals and handles feedback:

```python
# robot_actions/robot_actions/arm_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_actions.action import MoveArm
from geometry_msgs.msg import Pose

class ArmActionClient(Node):
    def __init__(self):
        super().__init__('arm_action_client')
        self._action_client = ActionClient(self, MoveArm, 'move_arm')

    def send_goal(self, target_x, target_y, target_z):
        goal = MoveArm.Goal()
        goal.target_pose = Pose(position=Point(x=target_x, y=target_y, z=target_z))
        goal.speed = 0.1

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.progress:.1%}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    client = ArmActionClient()
    client.send_goal(1.0, 0.5, 0.3)
    rclpy.spin(client)

if __name__ == '__main__':
    main()
```

## CLI Commands

Monitor action servers and send goals from the command line:

```bash
# List all action servers
ros2 action list

# Get action server type
ros2 action info /move_arm

# Send a goal via CLI
ros2 action send_goal /move_arm robot_actions/action/MoveArm \
  "{target_pose: {position: {x: 1.0, y: 0.5, z: 0.3}}, speed: 0.1}"

# Send goal with feedback
ros2 action send_goal /move_arm robot_actions/action/MoveArm \
  "{target_pose: {position: {x: 1.0, y: 0.5, z: 0.3}}, speed: 0.1}" \
  --feedback
```

## Exercise

**Task**: Create an action server for a robot gripper that opens/closes over 2 seconds, provides feedback every 0.5 seconds on gripper opening percentage, and can be canceled mid-operation. Include both server and client implementations.

**Acceptance Criteria**:
- Action definition supports goal (gripper open percentage) and feedback (current percentage)
- Server executes over 2 seconds with feedback published every 0.5s
- Client sends goal and logs feedback messages
- Both can be launched and communicate successfully

---

**Next**: [Chapter 8 - Working with Sensors](/docs/docs/module1-ros2/chapter8-sensors)
