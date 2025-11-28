---
id: 0001
title: Physical AI Textbook Constitution
stage: constitution
date: 2025-11-28
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: hackathon-participant
command: /sp.constitution
labels: ["constitution", "hackathon", "physical-ai", "textbook", "initial-setup"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A (constitution document, no tests)
---

## Prompt

lets build constitution rules for our hackathon project """Hackathon: Create a Textbook for Teaching Physical AI & Humanoid Robotics Course
The future of work will be a partnership between people, intelligent agents (AI software), and robots. This shift won't necessarily eliminate jobs but will change what humans do, leading to a massive demand for new skills. We have already written a book on AI agents. Therefore, we want you to write a textbook to teach a course in Physical AI & Humanoid Robotics (The course details are documented below).

Excel in the Hackathon and Launch Your Journey as an AI Startup Founder 🚀
We've recently launched Panaversity (panaversity.org), an initiative focused on teaching cutting-edge AI courses. Alongside this, we're working on publishing our first book, which you can explore at ai-native.panaversity.org. Our next milestone is to build a portal where authors can create AI-native technical textbooks, and readers can easily access and learn from them using AI Agents. We also plan to publish O/A Level, Science, Engineering, and Medical AI-native books to support students and professionals across disciplines. If you perform well in this hackathon, you may be invited for an interview to join the Panaversity core team and potentially step into the role of a startup founder within this growing ecosystem. You will get a chance to work with Panaversity founders Zia, Rehan, Junaid, and Wania and become the very best. You may also get a chance to teach at Panaversity, PIAIC, and GIAIC.
Requirements

You are required to complete a unified book project using Claude Code and Spec-Kit Plus. The core deliverables are:

1. AI/Spec-Driven Book Creation: Write a book using Docusaurus and deploy it to GitHub Pages. You will use Spec-Kit Plus ( https://github.com/panaversity/spec-kit-plus/ ) and Claude Code ( https://www.claude.com/product/claude-code ) to write the book.

2. Integrated RAG Chatbot Development: Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book. This chatbot, utilizing the OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres database, and Qdrant Cloud Free Tier, must be able to answer user questions about the book's content, including answering questions based only on text selected by the user.

3. Participants will receive points out of 100, for base functionality defined above.

4. Participants can earn up to 50 extra bonus points by creating and using reusable intelligence via Claude Code Subagents and Agent Skills in the book project.

5. Participants can receive up to 50 extra bonus points if they also implement Signup and Signin using https://www.better-auth.com/ At signup you will ask questions from the user about their software and hardware background. Knowing the background of the user we will be able to personalize the content.

6.  Participants can receive up to 50 extra bonus points if the logged user can personalise the content in the chapters by pressing a button at the start of each chapter.

7. Participants can receive up to 50 extra bonus points if the logged user can translate the content in Urdu in the chapters by pressing a button at the start of each chapter.

Timeline
Submission Deadline: Sunday, Nov 30, 2025 at 06:00 PM (form will close)
Live Presentations: Sunday, Nov 30, 2025 starting at 6:00 PM on Zoom

Top submissions will be invited via WhatsApp to present live on Zoom.
Note: All submissions will be evaluated. Live presentation is by invitation only, but does not affect final scoring.
Submit and Present Your Project:

Once you have completed the project you will submit your project here:

https://forms.gle/CQsSEGM3GeCrL43c8

Submit the following via the form:
Public GitHub Repo Link
Published Book Link for Github Pages or Vercel.
Include a demo video link (must be under 90 seconds). Judges will only watch the first 90 seconds. You can use NotebookLM or record your demo.
WhatsApp number (top submissions will be invited to present live)
Everyone is welcome to join the Zoom meeting to watch the presentations. Only invited participants will present their submissions. Meeting starts at 6:00 PM on Sunday, Nov 30:
The Course Details
Physical AI & Humanoid Robotics
Focus and Theme: AI Systems in the Physical World. Embodied Intelligence.
Goal: Bridging the gap between the digital brain and the physical body. Students apply their AI knowledge to control Humanoid Robots in simulated and real-world environments.
Quarter Overview
The future of AI extends beyond digital spaces into the physical world. This capstone quarter introduces Physical AI—AI systems that function in reality and comprehend physical laws. Students learn to design, simulate, and deploy humanoid robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.
Module 1: The Robotic Nervous System (ROS 2)
Focus: Middleware for robot control.
ROS 2 Nodes, Topics, and Services.
Bridging Python Agents to ROS controllers using rclpy.
Understanding URDF (Unified Robot Description Format) for humanoids.


Module 2: The Digital Twin (Gazebo & Unity)
Focus: Physics simulation and environment building.
Simulating physics, gravity, and collisions in Gazebo.
High-fidelity rendering and human-robot interaction in Unity.
Simulating sensors: LiDAR, Depth Cameras, and IMUs.


Module 3: The AI-Robot Brain (NVIDIA Isaac™)
Focus: Advanced perception and training.
NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation.
Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation.
Nav2: Path planning for bipedal humanoid movement.


Module 4: Vision-Language-Action (VLA)
Focus: The convergence of LLMs and Robotics.
Voice-to-Action: Using OpenAI Whisper for voice commands.
Cognitive Planning: Using LLMs to translate natural language ("Clean the room") into a sequence of ROS 2 actions.
Capstone Project: The Autonomous Humanoid. A final project where a simulated robot receives a voice command, plans a path, navigates obstacles, identifies an object using computer vision, and manipulates it

## Response snapshot

Created comprehensive constitution for Physical AI & Humanoid Robotics textbook project with 8 core principles:

1. **Educational-First Content Design**: Prioritize learning outcomes, progressive scaffolding from fundamentals to advanced topics, runnable code examples, and visual aids
2. **Modular Content Architecture**: Independent modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) with clear prerequisites, supporting personalization
3. **Interactive Learning Through RAG**: RAG chatbot as first-class learning companion with text-selection queries, context awareness, and citation
4. **Code-First Technical Validation**: All examples tested and executable in target environments (ROS 2 Humble/Iron, Gazebo, URDF validation)
5. **Accessibility and Personalization**: No hardware background assumed, user profile-based content adaptation, Urdu translation support
6. **Deployment and Performance Standards**: Production-ready deployment (Docusaurus, GitHub Pages/Vercel, <3s page load, <2s RAG response)
7. **Spec-Driven Development with Claude Code**: Spec-Kit Plus workflows, Claude Code subagents/skills for bonus points, ADRs for architectural decisions
8. **Version Control and Collaboration Hygiene**: Git best practices, clean commits, no secrets in repo, pinned dependencies

Additional sections:
- **Educational Content Guidelines**: Module structure, learning progression (Python-first, ROS 2 foundation before advanced modules)
- **Technical Architecture Standards**: Non-negotiable stack (Docusaurus, FastAPI, OpenAI, Neon Postgres, Qdrant), RAG architecture, auth architecture
- **Governance**: Constitution authority, amendment process, compliance review

Constitution version 1.0.0 ratified 2025-11-28.

## Outcome

- ✅ Impact: Constitution established with 8 core principles aligned to hackathon requirements (base functionality + all bonus point opportunities)
- 🧪 Tests: N/A (governance document)
- 📁 Files: Created .specify/memory/constitution.md with complete principles, guidelines, and governance
- 🔁 Next prompts: Create feature specifications for textbook chapters, RAG integration, authentication system; define subagents/skills for bonus points
- 🧠 Reflection: Constitution successfully balances educational quality, technical rigor, and hackathon scoring optimization. Clear guidance on modular content, RAG architecture, personalization, and Spec-Kit Plus workflow.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All placeholders filled, comprehensive principles covering hackathon scope
- Prompt variant (if applicable): N/A (initial constitution)
- Next experiment (smallest change to try): Test constitution compliance during first feature specification (e.g., Module 1: ROS 2 chapter spec)
