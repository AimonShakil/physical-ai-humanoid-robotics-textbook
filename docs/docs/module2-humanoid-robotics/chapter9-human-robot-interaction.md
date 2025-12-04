---
sidebar_position: 10
title: Chapter 9 - Human-Robot Interaction
---

# Human-Robot Interaction

## Overview

Humanoid robots are designed to work alongside humans in shared environments. Unlike industrial robots isolated behind barriers, humanoids must understand human intentions, communicate naturally, and ensure safety during close physical proximity. This chapter covers the technical aspects of human-robot interaction: safety considerations, gesture recognition, natural language processing, and control strategies for collaborative manipulation. Effective HRI is what transforms a powerful machine into a useful assistant.

## Safety in Human-Robot Interaction

Safety is the foundational requirement for HRI. When humans and robots share space, collisions must be prevented or managed gracefully.

### Force Limiting and Impedance Control

Instead of rigid trajectory tracking, collaborative robots use soft control that yields to external forces:

```python
import numpy as np

class ImpedanceController:
    """
    Impedance control: Make robot behave like a spring-damper system
    F = -K(x - x_desired) - D(v - v_desired)
    """

    def __init__(self, mass=50, stiffness=1000, damping=100):
        """
        mass: effective mass (kg)
        stiffness: spring constant (N/m)
        damping: damping coefficient (N·s/m)
        """
        self.M = mass
        self.K = stiffness
        self.D = damping
        self.x_desired = np.zeros(3)
        self.v_desired = np.zeros(3)

    def compute_force(self, x_actual, v_actual, external_force=None):
        """
        Compute control force with impedance model
        x_actual: current position (m)
        v_actual: current velocity (m/s)
        external_force: external force from human/environment (N)
        Returns: control force (N)
        """
        # Spring-damper force
        F_cmd = (-self.K * (x_actual - self.x_desired) -
                 self.D * (v_actual - self.v_desired))

        # If external force applied, reduce control force
        if external_force is not None:
            # Allow the system to move in direction of external force
            F_net = F_cmd + external_force
            # Acceleration from impedance model
            a = F_net / self.M
        else:
            a = F_cmd / self.M

        return F_cmd, a

    def safety_constraint(self, x_actual, force_threshold=150):
        """
        Stop motion if force exceeds threshold (human intervention)
        """
        external_force_est = np.linalg.norm(x_actual - self.x_desired)

        if external_force_est > force_threshold:
            self.x_desired = x_actual.copy()
            return False  # Stop motion
        return True  # Continue

# Example: Compliant arm during object handover
arm = ImpedanceController(mass=5, stiffness=500, damping=50)

# Simulate human pulling on arm
time_steps = 100
x_desired_traj = np.array([0, 0, 0.5])  # Target: 50 cm forward
arm.x_desired = x_desired_traj

x_actual = np.array([0, 0, 0])
v_actual = np.zeros(3)
dt = 0.01

for t in range(time_steps):
    if t == 50:
        # Human applies 200 N resistance at t=0.5s
        external_force = np.array([200, 0, 0])
    else:
        external_force = None

    F_cmd, a = arm.compute_force(x_actual, v_actual, external_force)
    is_safe = arm.safety_constraint(x_actual)

    if not is_safe:
        print(f"Safety limit triggered at t={t*dt:.2f}s")
        break

    # Update state
    v_actual += a * dt
    x_actual += v_actual * dt

print(f"Final position: {x_actual}")
```

### Power and Force Limiting

For human safety, we need to limit both power (energy transfer rate) and force:

```python
class SafetyController:
    """Enforce safety limits for collaborative operation"""

    def __init__(self, max_force=300, max_power=300):
        """
        max_force: maximum allowed force (N)
        max_power: maximum allowed power (W)
        """
        self.max_force = max_force
        self.max_power = max_power

    def limit_force(self, force_cmd):
        """Limit force to safe maximum"""
        force_norm = np.linalg.norm(force_cmd)
        if force_norm > self.max_force:
            return force_cmd * (self.max_force / force_norm)
        return force_cmd

    def limit_power(self, force, velocity):
        """Limit power = F · v"""
        power = np.dot(force, velocity)
        if power > self.max_power:
            # Reduce force to limit power
            velocity_norm = np.linalg.norm(velocity)
            if velocity_norm > 1e-6:
                scale_factor = self.max_power / (power + 1e-6)
                force = force * np.sqrt(scale_factor)
        return force

    def enforce_limits(self, force_cmd, velocity):
        """Apply all safety limits"""
        force_limited = self.limit_force(force_cmd)
        force_limited = self.limit_power(force_limited, velocity)
        return force_limited

# Test safety limits
safety = SafetyController(max_force=300, max_power=300)

force = np.array([400, 100, 0])  # Exceeds force limit
velocity = np.array([0.5, 0, 0])

force_safe = safety.enforce_limits(force, velocity)
print(f"Original force: {np.linalg.norm(force):.1f} N")
print(f"Limited force: {np.linalg.norm(force_safe):.1f} N")
```

## Gesture Recognition

Humanoids must understand human hand gestures for intuitive interaction:

```python
import numpy as np
from scipy.spatial.distance import euclidean

class GestureRecognizer:
    """Recognize hand gestures from skeleton data"""

    def __init__(self):
        # Reference gesture templates
        self.templates = {
            'point': self._get_point_template(),
            'stop': self._get_stop_template(),
            'thumbs_up': self._get_thumbs_up_template(),
            'wave': self._get_wave_template(),
        }

    def _get_point_template(self):
        """Extended arm with finger pointed forward"""
        # [shoulder, elbow, wrist, index_finger]
        return np.array([
            [0, 0, 0],
            [0.3, 0.2, 0],
            [0.6, 0.1, 0],
            [0.75, 0, 0]
        ])

    def _get_stop_template(self):
        """Palm facing forward at shoulder height"""
        return np.array([
            [0, 0, 0],
            [0.2, -0.1, 0],
            [0.3, -0.2, 0],
            [0.3, -0.3, 0]
        ])

    def _get_thumbs_up_template(self):
        """Fist with thumb up"""
        return np.array([
            [0, 0, 0],
            [0.1, 0.1, 0],
            [0.1, 0.2, 0.1],
            [0.1, 0.35, 0.1]
        ])

    def _get_wave_template(self):
        """Waving motion (multiple frames)"""
        # Just one frame for simplicity
        return np.array([
            [0, 0, 0],
            [0.15, 0.05, 0],
            [0.3, 0.1, 0],
            [0.35, 0.05, 0]
        ])

    def recognize(self, skeleton_points, threshold=0.15):
        """
        Recognize gesture from skeleton points
        skeleton_points: (n_joints, 3) array of joint positions
        Returns: (gesture_name, confidence)
        """
        best_match = None
        best_distance = float('inf')

        for gesture_name, template in self.templates.items():
            # Normalize both point sets
            skel_norm = self._normalize_skeleton(skeleton_points)
            template_norm = self._normalize_skeleton(template)

            # Compute distance
            distance = self._compute_skeleton_distance(skel_norm, template_norm)

            if distance < best_distance:
                best_distance = distance
                best_match = gesture_name

        confidence = 1.0 - (best_distance / threshold)

        if best_distance < threshold:
            return best_match, confidence
        return None, 0

    @staticmethod
    def _normalize_skeleton(skeleton):
        """Normalize skeleton for rotation/scale invariance"""
        # Center at origin
        skeleton = skeleton - skeleton[0]

        # Scale to unit length
        scale = np.max(np.linalg.norm(skeleton, axis=1))
        if scale > 0:
            skeleton = skeleton / scale

        return skeleton

    @staticmethod
    def _compute_skeleton_distance(skel1, skel2):
        """Compute distance between two skeletons (L2 norm)"""
        if len(skel1) != len(skel2):
            return float('inf')

        return np.linalg.norm(skel1 - skel2)

# Test gesture recognition
recognizer = GestureRecognizer()

# Simulate detected skeleton (pointing gesture)
detected_skeleton = np.array([
    [0, 0, 0],
    [0.3, 0.19, 0.01],
    [0.61, 0.09, -0.01],
    [0.74, -0.02, 0]
])

gesture, confidence = recognizer.recognize(detected_skeleton)
print(f"Recognized gesture: {gesture} (confidence: {confidence:.3f})")
```

## Speech and Natural Language Processing

Voice interaction is crucial for natural HRI:

```python
class SpeechProcessor:
    """Simple speech command processor"""

    def __init__(self):
        # Command vocabulary
        self.commands = {
            'pick_up': self._parse_pick_up,
            'put_down': self._parse_put_down,
            'move_left': self._parse_move,
            'move_right': self._parse_move,
            'follow_me': self._parse_follow,
            'stop': self._parse_stop,
        }

    def process_utterance(self, text):
        """
        Process speech input and extract intent + parameters
        Returns: (command, parameters)
        """
        text = text.lower().strip()

        # Simple keyword matching
        for command_name, parser in self.commands.items():
            if any(word in text for word in command_name.split('_')):
                params = parser(text)
                return command_name, params

        return None, {}

    @staticmethod
    def _parse_pick_up(text):
        """Extract object name from 'pick up [object]' command"""
        # Simple pattern matching
        objects = ['cup', 'ball', 'box', 'book', 'bottle']
        for obj in objects:
            if obj in text:
                return {'object': obj}
        return {}

    @staticmethod
    def _parse_put_down(text):
        """Extract location from 'put down [location]' command"""
        locations = ['table', 'floor', 'shelf', 'counter']
        for loc in locations:
            if loc in text:
                return {'location': loc}
        return {}

    @staticmethod
    def _parse_move(text):
        """Extract distance from move command"""
        # Look for distance: "move left 50 centimeters"
        import re
        match = re.search(r'(\d+)\s*(cm|centimeter|meter|m)', text)
        if match:
            distance = int(match.group(1))
            unit = match.group(2)[0]  # 'c' or 'm'
            if 'c' in unit:
                distance = distance / 100  # Convert to meters
            return {'distance': distance}
        return {'distance': 0.1}  # Default: 10 cm

    @staticmethod
    def _parse_follow(text):
        return {}

    @staticmethod
    def _parse_stop(text):
        return {}

# Test speech processing
speech = SpeechProcessor()

test_utterances = [
    "pick up the blue ball",
    "move left thirty centimeters",
    "put down the book on the table",
    "please follow me",
]

for utterance in test_utterances:
    cmd, params = speech.process_utterance(utterance)
    print(f"'{utterance}' -> {cmd} {params}")
```

## Emotion and Social Cues

Humanoid robots can express emotion through motion to improve interaction:

```python
class EmotionController:
    """Generate expressive robot motion"""

    @staticmethod
    def generate_happy_motion(duration=2.0, dt=0.01):
        """Generate happy/excited motion"""
        t = np.arange(0, duration, dt)

        # Bouncy up-down motion
        vertical_motion = 0.1 * np.sin(3 * np.pi * t)

        # Arm gestures (wide movements)
        left_arm_pitch = 0.4 * np.sin(2 * np.pi * t)
        left_arm_roll = 0.3 * np.cos(4 * np.pi * t)

        # Combined motion
        motion = {
            'torso_z': vertical_motion,
            'left_arm_pitch': left_arm_pitch,
            'left_arm_roll': left_arm_roll,
        }

        return t, motion

    @staticmethod
    def generate_sad_motion(duration=2.0, dt=0.01):
        """Generate sad/disappointed motion"""
        t = np.arange(0, duration, dt)

        # Slow, drooping motion
        head_pitch = 0.3 * (1 - np.cos(np.pi * t / duration))
        torso_sag = 0.05 * (1 - np.cos(2 * np.pi * t / duration))

        motion = {
            'head_pitch': head_pitch,
            'torso_z': torso_sag,
        }

        return t, motion

    @staticmethod
    def generate_curious_motion(duration=1.0, dt=0.01):
        """Generate inquisitive motion (head tilt, look around)"""
        t = np.arange(0, duration, dt)

        # Head tilt
        head_roll = 0.2 * np.sin(3 * np.pi * t)

        # Scanning motion
        head_yaw = 0.3 * np.sin(2 * np.pi * t)

        motion = {
            'head_roll': head_roll,
            'head_yaw': head_yaw,
        }

        return t, motion

# Generate and analyze motions
t_happy, motion_happy = EmotionController.generate_happy_motion()
vertical_amplitude = np.max(motion_happy['torso_z']) - np.min(motion_happy['torso_z'])
print(f"Happy motion vertical oscillation: {vertical_amplitude:.3f} m")

t_sad, motion_sad = EmotionController.generate_sad_motion()
droop_amount = np.max(motion_sad['head_pitch'])
print(f"Sad motion head droop: {droop_amount:.3f} rad")
```

## Collaborative Task Execution

Humans and robots working together on shared tasks:

```python
class CollaborativeTask:
    """Manage collaborative human-robot task"""

    def __init__(self, task_name):
        self.task_name = task_name
        self.subtasks = []
        self.current_subtask = 0
        self.state = 'waiting'  # waiting, executing, complete, error

    def add_subtask(self, description, robot_action, human_action):
        """Add a subtask requiring coordination"""
        self.subtasks.append({
            'description': description,
            'robot_action': robot_action,
            'human_action': human_action,
            'status': 'pending'
        })

    def execute(self):
        """Execute collaborative task"""
        if self.current_subtask >= len(self.subtasks):
            self.state = 'complete'
            return

        subtask = self.subtasks[self.current_subtask]

        # Execute robot action
        print(f"Robot: {subtask['robot_action']}")
        subtask['status'] = 'robot_executing'

        # Wait for human response
        print(f"Waiting for human: {subtask['human_action']}")
        subtask['status'] = 'waiting_human'

        # In real system: wait for confirmation from human
        # For now, simulate human completion
        print(f"Human: {subtask['human_action']} (simulated)")
        subtask['status'] = 'complete'

        # Move to next subtask
        self.current_subtask += 1

# Example: collaborative assembly task
task = CollaborativeTask("Assemble Robot Arm")

task.add_subtask(
    "Align connector",
    robot_action="Move connector to alignment position",
    human_action="Guide connector into socket"
)

task.add_subtask(
    "Lock connector",
    robot_action="Hold connector steady",
    human_action="Tighten locking screw"
)

task.add_subtask(
    "Verify assembly",
    robot_action="Check alignment with sensors",
    human_action="Visually inspect connection"
)

# Execute task
print(f"Starting task: {task.task_name}")
while task.state != 'complete':
    task.execute()

print(f"Task status: {task.state}")
```

## Exercise 9.1: HRI System Implementation

**Objective**: Build an integrated human-robot interaction system.

**Task**:
1. Implement impedance controller for safe arm compliance
2. Implement force limiting (max force and max power constraints)
3. Implement gesture recognition for 4 common gestures:
   - Pointing
   - Stop (hand up)
   - Thumbs up
   - Wave
4. Implement simple speech command processor
5. Create emotional expression system (happy, sad, curious)
6. Design a collaborative task: "Handover an object"
   - Robot approaches human with object
   - Human reaches out (detected by force sensor)
   - Robot transfers grip control based on force feedback
   - Release when human confirms grip (force threshold)
7. Simulate complete interaction scenario:
   - Human requests object with speech
   - Robot acknowledges (emotional expression)
   - Robot approaches and offers object
   - Human takes object with force feedback
   - Robot releases and confirms completion

**Submission**: HRI system report with:
- Safety constraint verification
- Gesture recognition accuracy on test set
- Force/impedance control demonstration
- Collaborative handover sequence visualization
- Interaction timeline and state transitions

---

## Next Chapter Preview

Chapter 10 explores real-world applications of humanoid robots, examining successful deployments in manufacturing, service industries, healthcare, and research. You'll see how the concepts from Chapters 1-9 combine to solve practical problems.

[→ Next: Chapter 10 - Real-World Applications and Case Studies](/docs/module2-humanoid-robotics/chapter10-applications)
