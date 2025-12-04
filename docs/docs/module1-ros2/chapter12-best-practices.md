---
sidebar_position: 13
title: Best Practices
---

# Best Practices

Writing production-quality ROS 2 code requires discipline, testing, and adherence to established patterns. This chapter consolidates best practices for robust, maintainable, scalable robotics systems.

## Code Organization and Structure

Structure your packages for clarity and maintainability:

```
robot_package/
├── robot_package/          # Python module
│   ├── __init__.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── sensor_node.py
│   │   └── controller_node.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── conversions.py
│   └── algorithms/
│       ├── __init__.py
│       └── pathfinding.py
├── launch/                 # Launch files
│   ├── robot_system.launch.py
│   └── simulation.launch.py
├── config/                 # Parameter files
│   ├── robot_params.yaml
│   └── sensor_config.yaml
├── srv/                    # Service definitions
├── msg/                    # Message definitions
├── action/                 # Action definitions
├── package.xml
├── setup.py
└── README.md
```

## Error Handling and Resilience

Write robust error handling in critical paths:

```python
# robot_package/robot_package/nodes/sensor_node.py
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import logging

class RobustSensorNode(Node):
    def __init__(self):
        super().__init__('robust_sensor_node')

        # Setup logging
        self.logger = logging.getLogger('RobustSensorNode')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Parameters
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('timeout_sec', 5.0)
        self.max_retries = self.get_parameter('max_retries').value

        # Use callback groups for concurrency
        self.callback_group = ReentrantCallbackGroup()

        self.sensor_sub = self.create_subscription(
            SensorData,
            '/sensor/data',
            self.sensor_callback,
            10,
            callback_group=self.callback_group
        )

        self.timer = self.create_timer(
            1.0,
            self.health_check_callback,
            callback_group=self.callback_group
        )

        self.last_sensor_time = None
        self.sensor_timeout = self.get_parameter('timeout_sec').value

    def sensor_callback(self, msg):
        try:
            # Validate data
            if not self._validate_sensor_data(msg):
                self.logger.warning('Invalid sensor data received')
                return

            self.last_sensor_time = self.get_clock().now()
            # Process valid data
            self._process_sensor_data(msg)

        except Exception as e:
            self.logger.error(f'Error processing sensor data: {e}', exc_info=True)

    def _validate_sensor_data(self, msg):
        """Validate sensor message integrity"""
        try:
            # Check message fields
            if msg.timestamp <= 0:
                self.logger.warn('Invalid timestamp')
                return False

            if not all(isinstance(v, float) for v in msg.values):
                self.logger.warn('Non-numeric sensor values')
                return False

            return True
        except AttributeError as e:
            self.logger.error(f'Sensor message validation error: {e}')
            return False

    def _process_sensor_data(self, msg):
        """Process validated sensor data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                result = self._expensive_computation(msg)
                self.logger.info(f'Sensor processing succeeded: {result}')
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f'Processing attempt {attempt + 1} failed, retrying: {e}'
                    )
                    continue
                else:
                    self.logger.error(f'Processing failed after {self.max_retries} attempts')

    def _expensive_computation(self, msg):
        """Simulate expensive computation that might fail"""
        # Implementation
        pass

    def health_check_callback(self):
        """Monitor node health"""
        if self.last_sensor_time is None:
            self.logger.warning('No sensor data received yet')
            return

        time_since_last = (self.get_clock().now() - self.last_sensor_time).nanoseconds / 1e9
        if time_since_last > self.sensor_timeout:
            self.logger.error(
                f'Sensor timeout: {time_since_last:.1f}s without data'
            )

def main(args=None):
    rclpy.init(args=args)
    node = RobustSensorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()
```

## Testing Strategies

Implement comprehensive testing:

```python
# robot_package/test/test_utils.py
import unittest
from robot_package.utils.transforms import quaternion_to_euler, euler_to_quaternion

class TestTransforms(unittest.TestCase):
    def test_quaternion_to_euler_identity(self):
        """Test identity quaternion converts to zero angles"""
        q = (0, 0, 0, 1)  # identity
        roll, pitch, yaw = quaternion_to_euler(q)
        self.assertAlmostEqual(roll, 0.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(yaw, 0.0, places=5)

    def test_quaternion_euler_roundtrip(self):
        """Test conversion roundtrip consistency"""
        euler_in = (0.5, 0.3, 1.2)
        q = euler_to_quaternion(*euler_in)
        euler_out = quaternion_to_euler(q)

        for e_in, e_out in zip(euler_in, euler_out):
            self.assertAlmostEqual(e_in, e_out, places=5)

    def test_euler_to_quaternion_90deg_roll(self):
        """Test 90-degree roll rotation"""
        roll, pitch, yaw = 1.5708, 0, 0  # 90 degrees
        q = euler_to_quaternion(roll, pitch, yaw)
        # Validate quaternion properties
        norm = sum(x**2 for x in q) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
```

## Logging Best Practices

Implement structured logging for debugging:

```python
# robot_package/robot_package/utils/logging.py
import logging
import functools
from datetime import datetime

def setup_node_logging(node, level=logging.INFO):
    """Configure node logging with consistent format"""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(node.get_name())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def log_execution_time(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = datetime.now()
        logger.debug(f'[ENTER] {func.__name__}')

        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(f'[EXIT] {func.__name__} ({elapsed:.3f}s)')
            return result
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(f'[ERROR] {func.__name__} ({elapsed:.3f}s): {e}')
            raise

    return wrapper
```

## Thread Safety and Concurrency

Use proper synchronization when accessing shared state:

```python
# robot_package/robot_package/nodes/concurrent_node.py
import threading
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class ConcurrentNode(Node):
    def __init__(self):
        super().__init__('concurrent_node')

        # State protection
        self._state_lock = threading.RLock()
        self._shared_state = {'value': 0}

        # Mutually exclusive group for serial execution
        self._serial_group = MutuallyExclusiveCallbackGroup()

        # Reentrant group for parallel execution
        self._parallel_group = ReentrantCallbackGroup()

        # Serial subscription (one at a time)
        self.sub_a = self.create_subscription(
            Msg, '/topic_a', self.callback_a, 10,
            callback_group=self._serial_group
        )

        # Parallel subscriptions (can run concurrently)
        self.sub_b = self.create_subscription(
            Msg, '/topic_b', self.callback_b, 10,
            callback_group=self._parallel_group
        )

    def callback_a(self, msg):
        with self._state_lock:
            self._shared_state['value'] += 1

    def callback_b(self, msg):
        with self._state_lock:
            current = self._shared_state['value']
            # Use consistent state
            self.get_logger().info(f'Current state: {current}')
```

## Performance Optimization

Monitor and optimize critical paths:

```python
# robot_package/robot_package/utils/profiling.py
import cProfile
import pstats
import io
from contextlib import contextmanager

@contextmanager
def profile_block(description=''):
    """Context manager for profiling code blocks"""
    pr = cProfile.Profile()
    pr.enable()

    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print(f'\n=== Profile: {description} ===')
        print(s.getvalue())
```

## CLI Commands for Development

Essential development commands:

```bash
# Run linting
colcon build --packages-select robot_package
rclint robot_package/

# Run tests
colcon test --packages-select robot_package
colcon test-result --all --verbose

# Monitor node performance
ros2 run ros2_performance_test perf_test

# Record and replay rosbags for debugging
ros2 bag record -a -o session1
ros2 bag play session1

# Check for memory leaks
valgrind --leak-check=full ros2 run robot_package sensor_node

# Profile node
python -m cProfile -o stats.prof robot_package/nodes/sensor_node.py
python -c "import pstats; p = pstats.Stats('stats.prof'); p.sort_stats('cumulative').print_stats(20)"
```

## Common Pitfalls to Avoid

Key mistakes to prevent:

1. **Blocking callbacks**: Never use sleep() in callbacks. Use timers instead.
2. **No timeout handling**: Always set timeouts on service calls and waits.
3. **Ignoring frame transforms**: Always use proper TF2 for coordinate transforms.
4. **Hardcoded values**: Use parameters for all tunable constants.
5. **Ignoring ROS2 lifecycle**: Properly implement on_configure, on_activate, etc.
6. **No graceful shutdown**: Implement cleanup in node destructors.
7. **Unbounded queues**: Always set reasonable queue sizes (default 10 is often too small).

## Deployment Checklist

Before deploying to hardware:

- [ ] All nodes use parameters for configuration
- [ ] Comprehensive logging at INFO and ERROR levels
- [ ] Timeout handling on all service/action calls
- [ ] Unit tests pass with >80% coverage
- [ ] Integration tests in sim verify full workflows
- [ ] Hardware interface properly implements safety stops
- [ ] All dependencies documented in package.xml
- [ ] Launch files tested on target hardware
- [ ] Monitoring and alerting configured
- [ ] Disaster recovery plan documented

## Exercise

**Task**: Create a production-ready ROS 2 node that:
1. Implements proper error handling with retries
2. Includes comprehensive logging
3. Has thread-safe state management
4. Includes unit tests (>70% coverage)
5. Uses callback groups appropriately
6. Implements health monitoring

**Acceptance Criteria**:
- Node runs without crashes for 5 minutes under simulated load
- All logs are properly formatted and useful
- Unit tests pass with pytest
- No thread safety issues under concurrent access
- Performance profile shows reasonable CPU/memory usage
- Documentation clearly explains all parameters and behavior

---

**Congratulations!** You've completed the ROS 2 Module 1. You now have the foundational knowledge to build, deploy, and maintain production robotics systems using ROS 2.

For continued learning:
- Explore the [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- Study advanced topics in Module 2: Advanced ROS 2
- Practice with real hardware projects
- Contribute to open-source ROS 2 packages
