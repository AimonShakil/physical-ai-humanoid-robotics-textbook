---
sidebar_position: 5
title: "Chapter 4: Computer Vision for Robotics"
---

# Computer Vision for Robotics

## Vision as the Primary Sense

For humanoid robots and manipulators, vision is often the richest source of information. A robot must:

- **Localize Objects**: Where is the target in 3D space?
- **Recognize Context**: What is the scene? Are there obstacles?
- **Estimate States**: How is an object oriented? Is it moving?
- **Reason About Interactions**: Which surfaces are graspable? Will that structure support weight?

Modern computer vision—powered by deep learning—enables robots to extract all these insights from RGB (or RGB-D) images.

## Convolutional Neural Networks (CNNs)

CNNs are the workhorse of robotic vision. They leverage local feature hierarchy: early layers detect edges, middle layers detect shapes, and later layers recognize objects.

### Standard Architectures

**ResNet-50**: Fast, accurate, widely supported. Standard backbone for many robotics applications.

**ViT (Vision Transformer)**: Treats image as sequence of patches; excellent generalization, but requires more compute.

**EfficientNet**: Scales compute/accuracy tradeoff; useful for edge robots.

## Object Detection for Robotics

Robots must not only classify what they see but localize objects spatially.

### YOLO for Real-Time Detection

YOLO (You Only Look Once) predicts bounding boxes and class probabilities in a single forward pass, making it ideal for real-time robotics:

```python
import torch
from torchvision.models import detection

class RoboticObjectDetector:
    """Wraps YOLO for real-time object detection on robots."""

    def __init__(self, model_name="fasterrcnn_resnet50_fpn", num_classes=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        if model_name == "fasterrcnn_resnet50_fpn":
            self.model = detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        """
        Args:
            image_tensor: (1, 3, H, W) or (3, H, W)
        Returns:
            detections: List of dicts with 'boxes', 'labels', 'scores'
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        predictions = self.model([image_tensor.squeeze(0)])

        # Parse predictions
        detections = {
            'boxes': predictions[0]['boxes'].cpu().numpy(),      # (N, 4)
            'labels': predictions[0]['labels'].cpu().numpy(),    # (N,)
            'scores': predictions[0]['scores'].cpu().numpy()     # (N,)
        }
        return detections

# Usage
detector = RoboticObjectDetector()
sample_image = torch.randn(3, 480, 640)
detections = detector.detect(sample_image)
print(f"Detected {len(detections['boxes'])} objects")
```

## Semantic and Instance Segmentation

Beyond bounding boxes, robots often need pixel-level understanding.

### Semantic Segmentation with DeepLabV3

```python
import torch
import torch.nn as nn
from torchvision import models

class RoboticSegmentation(nn.Module):
    """Semantic segmentation for scene understanding."""

    def __init__(self, num_classes=21):
        super().__init__()
        # Load pretrained DeepLabV3
        self.segmenter = models.segmentation.deeplabv3_resnet50(
            pretrained=True,
            num_classes=num_classes
        )
        self.segmenter.eval()

    @torch.no_grad()
    def segment(self, image):
        """
        Args:
            image: (B, 3, H, W)
        Returns:
            segmentation_map: (B, num_classes, H, W)
        """
        output = self.segmenter(image)
        return output['out']  # Raw logits

    def get_class_masks(self, image):
        """
        Returns probability masks for each class.
        """
        logits = self.segment(image)  # (B, num_classes, H, W)
        probs = torch.softmax(logits, dim=1)
        return probs

# Usage
segmenter = RoboticSegmentation(num_classes=21)
image = torch.randn(1, 3, 256, 256)
seg_map = segmenter.segment(image)
print(f"Segmentation map shape: {seg_map.shape}")  # (1, 21, 256, 256)
```

## Pose Estimation and 6D Object Pose

For manipulation, robots need to know the 6D pose (3D position + 3D orientation) of objects.

### 6D Pose from RGBD

```python
class SixDPoseEstimator(nn.Module):
    """Estimate 6D pose (position + rotation) from RGB-D."""

    def __init__(self):
        super().__init__()
        # Feature extractor (shared for all modalities)
        self.feature_extractor = models.resnet50(pretrained=True)
        num_features = self.feature_extractor.fc.in_features

        # Remove final classification layer
        self.feature_extractor.fc = nn.Identity()

        # Pose prediction heads
        self.position_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # x, y, z
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Quaternion (qx, qy, qz, qw)
        )

    def forward(self, rgb_image, depth_image):
        """
        Args:
            rgb_image: (B, 3, H, W)
            depth_image: (B, 1, H, W) normalized to [0, 1]
        Returns:
            position: (B, 3)
            quaternion: (B, 4)
        """
        # Concatenate RGB and depth
        rgbd = torch.cat([rgb_image, depth_image], dim=1)

        # Adapt first conv layer to accept 4 channels
        # (In practice, use a preprocessing or modify model)
        features = self.feature_extractor(rgb_image)  # Use RGB only for simplicity

        position = self.position_head(features)
        quaternion = self.rotation_head(features)

        # Normalize quaternion
        quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=1)

        return position, quaternion

# Usage
estimator = SixDPoseEstimator()
rgb = torch.randn(4, 3, 224, 224)
depth = torch.randn(4, 1, 224, 224)
pos, quat = estimator(rgb, depth)
print(f"Position shape: {pos.shape}, Quaternion shape: {quat.shape}")  # (4, 3), (4, 4)
```

## Foundation Models: CLIP and Vision-Language Models

**CLIP** (Contrastive Language-Image Pre-training) bridges vision and language. Robots can now use natural language descriptions to find objects:

```python
from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPBasedRobotVision:
    """Use CLIP for language-grounded object detection."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def find_objects(self, image, text_queries):
        """
        Args:
            image: PIL Image or numpy array
            text_queries: List of text descriptions (e.g., ["red cube", "green ball"])
        Returns:
            similarity_scores: (len(text_queries),) - higher means more similar
        """
        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_queries)

        similarity_scores = torch.softmax(logits_per_image, dim=-1).squeeze()
        return similarity_scores.cpu().numpy()

# Usage
clip_robot = CLIPBasedRobotVision()
# (Assume 'image' is loaded)
# queries = ["red cube", "blue cylinder", "green sphere"]
# scores = clip_robot.find_objects(image, queries)
# best_object = queries[scores.argmax()]
# print(f"Found: {best_object}")
```

## Dense Prediction: Optical Flow and Normal Estimation

For dynamic scenes, optical flow estimates motion between frames. Surface normals reveal 3D geometry.

### Optical Flow for Tracking

```python
class OpticalFlowTracker(nn.Module):
    """Track moving objects using optical flow."""

    def __init__(self):
        super().__init__()
        # Use RAFT (Recurrent All-Pairs Field Transforms) for high-quality flow
        # For simplicity, we show a basic architecture
        self.encoder = models.resnet50(pretrained=True)
        # In practice, use: from torchvision.models.optical_flow import raft_large

    def forward(self, frame1, frame2):
        """
        Args:
            frame1, frame2: (B, 3, H, W)
        Returns:
            flow: (B, 2, H, W) - (dx, dy) for each pixel
        """
        # Placeholder; actual implementation uses RAFT
        raise NotImplementedError("Use torchvision.models.optical_flow.raft_large()")
```

## Robotic Vision Pipeline: End-to-End Example

```python
class RobotVisionPipeline:
    """Complete vision pipeline for a robotic manipulator."""

    def __init__(self):
        self.detector = RoboticObjectDetector()
        self.segmenter = RoboticSegmentation(num_classes=5)
        self.pose_estimator = SixDPoseEstimator()
        self.clip_model = CLIPBasedRobotVision()

    def process_frame(self, rgb_image, depth_image, task_description):
        """
        Args:
            rgb_image: (3, H, W)
            depth_image: (1, H, W)
            task_description: "pick up the red cube"
        """
        rgb_batch = rgb_image.unsqueeze(0)
        depth_batch = depth_image.unsqueeze(0)

        # Step 1: Detect objects
        detections = self.detector.detect(rgb_image)

        # Step 2: Segment scene
        seg_probs = self.segmenter.get_class_masks(rgb_batch)

        # Step 3: Estimate poses
        positions, quats = self.pose_estimator(rgb_batch, depth_batch)

        # Step 4: Match with language
        scores = self.clip_model.find_objects(rgb_image, [task_description])

        return {
            'detections': detections,
            'segmentation': seg_probs,
            'poses': (positions, quats),
            'task_match_score': scores[0]
        }
```

## Challenges in Robotic Vision

1. **Domain Shift**: Models trained on internet images fail on robot-specific scenes
2. **Speed vs. Accuracy**: Real-time inference on edge hardware vs. state-of-the-art models
3. **Robustness to Lighting**: Robots operate under varied lighting; models must generalize
4. **Occlusion**: Objects hidden behind others are common in cluttered environments
5. **Continual Learning**: Robots encounter novel objects; static models are limited

## Exercises

**Exercise 4.1**: Fine-tune a pretrained YOLO model on a custom robotic manipulation dataset (e.g., YCB objects). Evaluate performance on novel backgrounds.

**Exercise 4.2**: Build a grasp detection pipeline using semantic segmentation + 6D pose estimation. Predict gripper poses for top-down grasps on detected objects.

**Exercise 4.3**: Investigate domain generalization techniques (style transfer, adversarial training) to improve robotic vision model robustness across different cameras and lighting conditions.

---

**Next Chapter**: [Chapter 5: Reinforcement Learning for Robot Control](./chapter5-reinforcement-learning.md)
