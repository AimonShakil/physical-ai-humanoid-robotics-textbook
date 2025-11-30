---
sidebar_position: 3
title: Chapter 2 - Installation & Setup
---

# Chapter 2: Installation & Setup

## Introduction

In this chapter, you'll set up a complete ROS 2 development environment. We'll cover installation on Ubuntu (recommended), as well as alternatives for Windows and macOS users.

## System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 20 GB free space
- **OS**: Ubuntu 22.04 or 24.04 LTS (recommended)

### Recommended Setup
- **CPU**: Quad-core 3.0 GHz+
- **RAM**: 8 GB+
- **Storage**: 50 GB+ SSD
- **GPU**: NVIDIA GPU (for simulation/ML)

## Installation Options

### Option 1: Ubuntu 22.04 (Humble) - Recommended

#### Step 1: Set Locale

```bash
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

#### Step 2: Enable Ubuntu Universe Repository

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

#### Step 3: Add ROS 2 GPG Key

```bash
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
```

#### Step 4: Add Repository to Sources

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

#### Step 5: Install ROS 2

```bash
# Update package index
sudo apt update
sudo apt upgrade

# Install ROS 2 Humble Desktop (recommended)
sudo apt install ros-humble-desktop

# Or install minimal version (no GUI tools)
# sudo apt install ros-humble-ros-base
```

#### Step 6: Install Development Tools

```bash
sudo apt install ros-dev-tools
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep
```

#### Step 7: Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### Option 2: Ubuntu 24.04 (Jazzy)

Replace `humble` with `jazzy` in the commands above:

```bash
sudo apt install ros-jazzy-desktop
```

### Option 3: Windows 10/11

#### Using WSL2 (Recommended)

1. Install WSL2:
```powershell
wsl --install -d Ubuntu-22.04
```

2. Follow Ubuntu installation steps inside WSL2

#### Native Windows Installation

Follow the [Windows installation guide](https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html)

**Note**: Some packages may not be available on Windows.

### Option 4: macOS

Install via Homebrew (Tier 2 support):

```bash
brew install ros-humble-desktop
```

Or use Docker (recommended for macOS).

## Environment Setup

### Sourcing ROS 2

Add to your `~/.bashrc`:

```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Optional: Add to PATH
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

Apply changes:
```bash
source ~/.bashrc
```

### Verify Installation

```bash
# Check ROS 2 environment
printenv | grep ROS

# Expected output:
# ROS_DISTRO=humble
# ROS_VERSION=2
# ROS_PYTHON_VERSION=3
```

## Create Your Workspace

### Setup Workspace Structure

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source workspace
source install/setup.bash
```

### Workspace Structure

```
~/ros2_ws/
├── build/          # Build artifacts
├── install/        # Install files
├── log/            # Build logs
└── src/            # Source packages
    └── your_package/
        ├── package.xml
        ├── setup.py
        └── your_package/
            └── __init__.py
```

## Test Installation

### Test 1: Run Demo Nodes

**Terminal 1** (Talker):
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

**Terminal 2** (Listener):
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

Expected output:
```
[INFO] [listener]: I heard: [Hello World: 1]
[INFO] [listener]: I heard: [Hello World: 2]
```

### Test 2: Check ROS 2 Commands

```bash
# List running nodes
ros2 node list

# List active topics
ros2 topic list

# Show topic info
ros2 topic info /chatter

# Echo topic messages
ros2 topic echo /chatter
```

## Development Tools

### 1. Visual Studio Code

Install VS Code and ROS extension:

```bash
# Install VS Code
sudo snap install code --classic

# Install ROS extension
code --install-extension ms-iot.vscode-ros
```

### 2. Terminator (Multi-terminal)

```bash
sudo apt install terminator
```

Split terminals with:
- **Ctrl+Shift+E**: Split vertically
- **Ctrl+Shift+O**: Split horizontally

### 3. RQt (ROS Qt GUI)

```bash
sudo apt install ros-humble-rqt*
rqt
```

### 4. Gazebo Simulator

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
```

### 5. RViz2 (Visualization)

```bash
sudo apt install ros-humble-rviz2
rviz2
```

## Common Installation Issues

### Issue 1: Missing Dependencies

```bash
# Solution: Install missing dependencies
rosdep install --from-paths src --ignore-src -r -y
```

### Issue 2: Locale Not Set

```bash
# Solution: Set locale
export LC_ALL=C
sudo locale-gen en_US.UTF-8
```

### Issue 3: GPG Key Error

```bash
# Solution: Re-add GPG key
sudo rm /usr/share/keyrings/ros-archive-keyring.gpg
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
```

## Best Practices

### 1. Use Workspaces

Organize projects in separate workspaces:

```bash
~/ros2_ws/          # General development
~/robot_ws/         # Robot-specific packages
~/simulation_ws/    # Simulation packages
```

### 2. Source Workspace Automatically

Add to `~/.bashrc`:

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace (if exists)
if [ -f ~/ros2_ws/install/setup.bash ]; then
    source ~/ros2_ws/install/setup.bash
fi
```

### 3. Use colcon_cd

```bash
# Navigate to package directory
colcon_cd <package_name>
```

## Docker Alternative

For isolated environments:

```dockerfile
FROM ros:humble

# Install additional packages
RUN apt-get update && apt-get install -y \
    ros-humble-demo-nodes-cpp \
    ros-humble-demo-nodes-py \
    && rm -rf /var/lib/apt/lists/*

# Setup entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
```

Build and run:

```bash
docker build -t my-ros2-image .
docker run -it my-ros2-image
```

## Summary Checklist

- ✅ Ubuntu 22.04/24.04 installed or WSL2 configured
- ✅ ROS 2 Humble/Jazzy installed
- ✅ Development tools installed (colcon, rosdep)
- ✅ Environment sourced in `~/.bashrc`
- ✅ Workspace created (`~/ros2_ws`)
- ✅ Demo nodes run successfully
- ✅ VS Code with ROS extension installed
- ✅ Gazebo and RViz2 installed

## Exercise

1. **Installation**: Complete ROS 2 installation on your system
2. **Verification**: Run talker/listener demo nodes
3. **Workspace**: Create and build a workspace
4. **Exploration**: Use `ros2` CLI commands to explore demo nodes

## Next Chapter

In [Chapter 3](/docs/module1-ros2/chapter3-core-concepts), we'll dive deep into ROS 2 core concepts: nodes, topics, services, and actions!

## Additional Resources

- [Official Installation Guide](https://docs.ros.org/en/humble/Installation.html)
- [Troubleshooting Guide](https://docs.ros.org/en/humble/How-To-Guides/Installation-Troubleshooting.html)
- [Docker Images](https://hub.docker.com/_/ros)
