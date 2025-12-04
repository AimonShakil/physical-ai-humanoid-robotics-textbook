---
sidebar_position: 8
title: "Chapter 7: Language Models for Robotics"
---

# Language Models for Robotics (LLMs + VLMs)

## The Language-Robotics Bridge

Large Language Models (LLMs) like GPT-4 and vision-language models (VLMs) like GPT-4V introduce a new paradigm: **robots that understand and reason about natural language instructions**.

Rather than hardcoding task specifications, a robot can now be commanded with English: *"Pick up the red cube and place it on the shelf."* The system interprets this, decomposes it into sub-tasks, and executes.

### Why LLMs Matter for Robotics

1. **Task Decomposition**: "Make a cup of tea" → "Boil water" → "Steep tea bag" → "Pour into cup"
2. **Common Sense**: LLMs encode vast world knowledge ("Boiling water is hot, don't touch it")
3. **Natural Interfaces**: Humans don't want to learn robot APIs; natural language is intuitive
4. **Few-Shot Reasoning**: LLMs can adapt to new tasks with a few examples
5. **Planning**: LLMs can reason about multi-step plans before execution

## VLMs for Scene Understanding

Vision-Language Models like GPT-4V or CLIP enable robots to:
- Answer visual questions: "What color is the object in the center?"
- Describe scenes and identify relationships
- Ground language in visual observations
- Make decisions based on visual input and language context

### Code Example: Using GPT-4V for Robot Perception

```python
import base64
import json
from openai import OpenAI

class RobotVisionLanguageAgent:
    """Use GPT-4V to interpret robot observations and plan actions."""

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4-vision-preview"

    def describe_scene(self, image_path):
        """
        Ask GPT-4V to describe a robot's current scene.

        Args:
            image_path: Path to RGB image from robot camera
        Returns:
            description: Natural language description
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "You are a robot vision system. Describe what you see in this image. "
                                   "Focus on objects, their colors, positions, and any notable relationships."
                        }
                    ]
                }
            ]
        )

        return message.content[0].text

    def plan_grasping(self, image_path, target_object):
        """
        Ask GPT-4V where to grasp an object.

        Args:
            image_path: Robot camera image
            target_object: Description (e.g., "red cube")
        Returns:
            grasp_advice: Recommended grasp strategy
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"You are advising a robot gripper. Where should the robot grasp the {target_object}? "
                                   "Describe the grasp point (e.g., 'grasp the handle on the left side') "
                                   "and the approach angle."
                        }
                    ]
                }
            ]
        )

        return message.content[0].text

# Usage
agent = RobotVisionLanguageAgent(api_key="your-api-key")
scene_description = agent.describe_scene("robot_view.jpg")
grasp_plan = agent.plan_grasping("robot_view.jpg", "red cube")
print(f"Scene: {scene_description}")
print(f"Grasp: {grasp_plan}")
```

## Task Planning with LLMs

LLMs excel at decomposing high-level goals into executable sub-tasks. This is the essence of hierarchical planning.

### Chain-of-Thought Planning

```python
from openai import OpenAI

class RobotPlannerLLM:
    """Use LLM for hierarchical task planning."""

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"

    def decompose_task(self, task_description, robot_capabilities=None):
        """
        Break high-level task into steps.

        Args:
            task_description: Goal (e.g., "Organize books on the shelf")
            robot_capabilities: List of robot skills
        Returns:
            steps: List of executable sub-tasks
        """
        if robot_capabilities is None:
            robot_capabilities = [
                "move_to(position)",
                "grasp(object)",
                "release()",
                "push(object, direction)",
                "observe()"
            ]

        prompt = f"""You are a robot task planner. The robot has these capabilities:
{chr(10).join(f"- {cap}" for cap in robot_capabilities)}

Break down the following task into a sequence of executable steps:
"{task_description}"

For each step, specify:
1. The action (using robot capabilities)
2. Required preconditions
3. Expected outcome

Format as a numbered list."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    def check_feasibility(self, plan, constraints=None):
        """
        Ask LLM if a plan is feasible given constraints.

        Args:
            plan: Proposed task plan (string)
            constraints: List of constraints (e.g., "cannot use gripper more than 10 times")
        Returns:
            feasibility_report: Analysis of feasibility and risks
        """
        if constraints is None:
            constraints = []

        prompt = f"""Analyze this robot task plan for feasibility:

Plan:
{plan}

Constraints:
{chr(10).join(f"- {c}" for c in constraints)}

Identify potential issues, safety concerns, and suggest improvements."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

# Usage
planner = RobotPlannerLLM(api_key="your-api-key")
task = "Pick up the coffee mug and place it on the table"
plan = planner.decompose_task(task)
print(f"Task Plan:\n{plan}")

feasibility = planner.check_feasibility(plan, constraints=["avoid hot liquids"])
print(f"Feasibility Analysis:\n{feasibility}")
```

## RT-2: Robotics Transformer

**RT-2** (Robotics Transformer 2) combines a vision-language model backbone with robotic action tokens. Instead of learning separate vision and control modules, RT-2 learns a unified policy that:
- Takes images and language instructions as input
- Outputs actions directly (tokenized joint angles, gripper commands)

### Conceptual Example: Unified Visuomotor Policy

```python
import torch
import torch.nn as nn
from transformers import ViTModel, AutoTokenizer

class RT2LikeRobotPolicy(nn.Module):
    """Simplified RT-2 style unified visuomotor policy."""

    def __init__(self, action_dim=7, hidden_dim=512, num_action_tokens=256):
        super().__init__()

        # Vision encoder (pretrained ViT)
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # Language tokenizer and encoder
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = nn.Sequential(
            nn.Embedding(30522, hidden_dim),  # Vocab size of BERT
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
        )

        # Shared fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Action decoder: predicts action tokens
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_tokens)
        )

        # Continuous action regression (post-tokenization)
        self.action_regressor = nn.Sequential(
            nn.Linear(hidden_dim + num_action_tokens, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, image, instruction_text):
        """
        Args:
            image: (B, 3, H, W)
            instruction_text: List of strings (e.g., ["pick up red cube"])
        Returns:
            action: (B, action_dim)
        """
        # Encode vision
        vision_features = self.vision_encoder(image).last_hidden_state  # (B, seq_len, 768)
        vision_features = vision_features.mean(dim=1, keepdim=True)  # Global average

        # Encode language
        tokenized = self.tokenizer(
            instruction_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_ids = tokenized.input_ids
        # Assume text_ids is (B, text_seq_len)

        # Simple text encoding (in practice, use full BERT-like encoder)
        text_features = text_ids.float().unsqueeze(-1).expand(-1, -1, 512)  # Mock
        text_features = text_features.mean(dim=1, keepdim=True)  # Global average

        # Fuse modalities
        multimodal_input = torch.cat([vision_features, text_features], dim=1)  # (B, 2, hidden_dim)
        fused = self.fusion_transformer(multimodal_input)  # (B, 2, hidden_dim)

        # Decode action
        fused_final = fused.mean(dim=1)  # (B, hidden_dim)
        action_logits = self.action_decoder(fused_final)  # (B, num_action_tokens)

        # Regress continuous action
        combined = torch.cat([fused_final, action_logits], dim=1)
        action = self.action_regressor(combined)  # (B, action_dim)

        return action

# Usage (mock)
policy = RT2LikeRobotPolicy(action_dim=7)
image = torch.randn(2, 3, 224, 224)
instructions = ["pick up the red cube", "move to the table"]
actions = policy(image, instructions)
print(f"Predicted actions shape: {actions.shape}")  # (2, 7)
```

## PaLM-E: Embodied Language Models

**PaLM-E** grounds a large language model (PaLM) with proprioceptive and visual observations. The model directly outputs actions based on:
- Continuous sensor readings (joint angles, end-effector position)
- Visual observations
- Natural language instructions

Key innovation: treating sensorimotor data as tokens in the same sequence as text, enabling the LLM to reason about physical interactions.

## In-Context Learning for Robotics

LLMs can rapidly adapt to new tasks using **in-context learning** (few-shot prompting):

```python
class InContextRobotLearner:
    """Few-shot robotic task learning via LLM prompting."""

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def learn_from_examples(self, task_examples, new_task):
        """
        Learn a new task from a few demonstrations.

        Args:
            task_examples: List of (observation, action) pairs
            new_task: New task description
        Returns:
            action_plan: Recommended action for new task
        """
        # Format examples as context
        context = "Here are examples of robot tasks:\n\n"
        for i, (obs, action) in enumerate(task_examples):
            context += f"Example {i+1}:\n"
            context += f"Observation: {obs}\n"
            context += f"Action: {action}\n\n"

        prompt = context + f"Now, for this new observation: {new_task}\n" \
                           "What action should the robot take?"

        response = self.client.messages.create(
            model="gpt-4",
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

# Usage
learner = InContextRobotLearner(api_key="your-api-key")
examples = [
    ("red cube on table", "approach and grasp"),
    ("blue sphere in bin", "reach into bin, then grasp"),
]
task = "green cylinder on shelf"
action = learner.learn_from_examples(examples, task)
```

## Challenges and Limitations

1. **Hallucinations**: LLMs can confidently propose infeasible or unsafe actions
2. **Real-World Grounding**: LLMs trained on text lack true understanding of physics and constraints
3. **Latency**: API calls to cloud LLMs introduce latency; real-time control requires local deployment
4. **Safety**: Robots must verify LLM suggestions before acting; blind following is dangerous
5. **Embodiment Gap**: Knowledge in LLMs ≠ understanding of physical constraints and consequences

## Safe Robot Control with LLMs

A practical architecture combines LLM reasoning with classical safety:

```python
class SafeLLMRobotController:
    """LLM planning + safety verification."""

    def __init__(self, api_key, robot_model):
        self.llm = RobotPlannerLLM(api_key)
        self.robot = robot_model
        self.safety_checker = RobotSafetyValidator(robot_model)

    def execute_task(self, task_description):
        """
        Execute task safely.
        1. LLM generates plan
        2. Safety checker validates
        3. Execute validated steps
        """
        # Generate plan
        plan = self.llm.decompose_task(task_description)

        # Check safety
        is_safe, issues = self.safety_checker.validate(plan)

        if not is_safe:
            print(f"Safety concerns detected: {issues}")
            # Ask for human approval or modification
            return False

        # Execute each step with monitoring
        steps = parse_plan(plan)
        for step in steps:
            print(f"Executing: {step}")
            try:
                self.robot.execute(step)
                self.robot.verify_success(step)
            except Exception as e:
                print(f"Execution failed: {e}")
                return False

        return True

class RobotSafetyValidator:
    """Simple safety checker (in production, much more sophisticated)."""

    def __init__(self, robot_model):
        self.robot = robot_model

    def validate(self, plan):
        """Check for obvious safety violations."""
        issues = []
        if "destroy" in plan.lower() or "break" in plan.lower():
            issues.append("Plan mentions destructive actions")
        if "human" in plan.lower() and "collision" in plan.lower():
            issues.append("Potential collision with human")
        return len(issues) == 0, issues
```

## Exercises

**Exercise 7.1**: Use GPT-4V to analyze a robot manipulation image. Ask it to identify graspable objects, suggest grasp points, and describe what it sees. Compare to a standard CV approach.

**Exercise 7.2**: Implement a multi-step planning task where an LLM decomposes a goal (e.g., "make breakfast") and the robot executes each step in simulation.

**Exercise 7.3**: Explore in-context learning: provide 3-5 task examples and ask GPT-4 to generalize to a novel task. How well does it transfer? What breaks down?

---

**Next Chapter**: [Chapter 8: Multi-Modal Robot AI Systems](./chapter8-multimodal-ai.md)
