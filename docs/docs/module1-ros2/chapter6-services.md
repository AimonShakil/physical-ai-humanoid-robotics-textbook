---
sidebar_position: 7
title: Chapter 6 - Services & Clients
---

# Chapter 6: Services & Clients

## Service Overview

Services provide synchronous request-response communication. Use services for infrequent operations like configuration, triggering actions, or querying state.

## Service Types

```bash
# List service types
ros2 interface list | grep srv

# Common types
std_srvs/srv/SetBool
std_srvs/srv/Trigger
example_interfaces/srv/AddTwoInts
```

## Service Server

```python
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )
        self.get_logger().info('Service ready')

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response
```

## Service Client (Synchronous)

```python
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result().sum
        else:
            self.get_logger().error('Service call failed')
            return None
```

## Service Client (Async)

```python
class AsyncClient(Node):
    def __init__(self):
        super().__init__('async_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

    async def send_request_async(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.client.call_async(request)
        result = await future
        return result.sum
```

## Custom Service

Create `srv/ComputeRectangle.srv`:
```
float64 length
float64 width
---
float64 area
float64 perimeter
```

Use in code:
```python
from my_package.srv import ComputeRectangle

def compute_callback(self, request, response):
    response.area = request.length * request.width
    response.perimeter = 2 * (request.length + request.width)
    return response
```

## CLI Usage

```bash
# List services
ros2 service list

# Service type
ros2 service type /add_two_ints

# Call service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 3}"
```

## Error Handling

```python
def add_callback(self, request, response):
    try:
        if request.a < 0 or request.b < 0:
            self.get_logger().warn('Negative values provided')
        response.sum = request.a + request.b
        return response
    except Exception as e:
        self.get_logger().error(f'Error: {e}')
        return response
```

**Exercise**: Create service to set robot speed limits (min/max velocity).

[Next: Actions](/docs/module1-ros2/chapter7-actions)
