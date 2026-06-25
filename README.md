# Autonomous Warehouse Drone Training Platform

*A high-performance, Rust-based Reinforcement Learning (RL) platform designed to train autonomous warehouse drones in simulated environments.*

## 🚀 The Vision: No-Code Industrial Training

The gap between advanced robotics and real-world logistics is often limited by software engineering bottlenecks. Industries need to deploy autonomous drones safely, but hiring dedicated machine learning teams to train drones for every unique warehouse layout is simply not scalable.

This platform is being built with a long-term vision: A zero-code simulation-to-reality pipeline. By providing a robust, underlying RL architecture, warehouse managers will eventually be able to upload their floor plans (shelves, pickup zones, delivery points) and let the platform autonomously train a flight policy. The drones learn to navigate, avoid dynamic obstacles (like workers and forklifts), and optimize delivery routes in a completely risk-free simulation before the brain is ever deployed to physical hardware.

## 🏁 Phase 1: Foundational Intelligence (Completed)

Phase 1 focuses on building the mathematical and architectural backend from scratch in pure Rust.

* **Environment:** A custom 2D discrete grid world representing a simplified warehouse aisle.
* **Agent:** A Deep Q-Network (DQN) built using the `burn` deep learning framework.
* **Mechanics:** Implements $\epsilon$-greedy exploration, experience replay buffers, Bellman equation target syncing, and automated reward calculation.
* **Milestone:** The agent successfully learns to navigate from a starting position to a target coordinate while actively avoiding static walls and shelving units, optimizing for the shortest possible path.

## 🚧 Phase 2: Dynamic Environments & Sequential Tasks (Completed)

Phase 2 escalates the complexity by introducing moving elements and multi-step objectives, forcing the Neural Network to develop situational awareness and sub-policies.

* **Dynamic Obstacles:** Introduced a moving entity (representing a worker/forklift) that patrols a specific aisle. The drone must learn to time its movements to avoid catastrophic collisions.
* **Sequential Tasks (Pick & Drop):** The objective evolved from a simple single-destination pathfinding task to a multi-stage task: navigate to a Pickup zone, fetch a package, and then transport it to a Delivery zone.
* **Upgraded State Representation:** Expanded the neural network's input capacity from 2D to 5D (`[drone_x, drone_y, obstacle_x, obstacle_y, has_package]`) and increased the hidden layer depth to 128 neurons to process complex spatial-temporal relationships.
* **Milestone:** The agent successfully mastered the pick-and-drop sequence, consistently achieving the mathematical optimum path while dynamically dodging the moving obstacle.

## 🛸 Phase 3: Continuous Physics & Sensor Simulation (Completed)

Phase 3 bridges the gap between simulation and reality by abandoning the discrete grid and introducing aerospace physics and simulated hardware sensors.

* **Continuous Coordinate System:** Replaced rigid 90-degree movements with continuous physics, requiring the AI to manage thrust, velocity, acceleration, and aerodynamic drag.
* **Sensor Arrays (Lidar):** Removed absolute (x,y) coordinates from the drone's memory. The agent now relies on simulated raycasting to read distances to walls, mimicking real-world Lidar.
* **Domain Randomization:** The mass of the drone is randomly perturbed by ±20% every episode. This forces the Deep Q-Network to learn a robust, dynamic control policy based on its current velocity rather than memorizing static thrust timings.
* **Milestone:** The agent learned to fly optimized, curved aerodynamic paths—applying thrust to cut corners, coasting on its own momentum, and executing reverse-thrust braking to land smoothly in the delivery zone.

## 🧠 Phase 4: Sim-to-Real & Hardware Abstraction (Completed)

Phase 4 extracts the trained AI from the simulator and packages it for real-world or 3D engine deployment.

* **Weight Export:** The Deep Q-Network's parameters are serialized and compressed into a portable `.mpk` format (`drone_brain.mpk`).
* **Lightweight Inference Engine:** Created a forward-pass-only engine that strips away all Autodiff and training overhead. It loads the `.mpk` file and issues commands at a high frequency, making it perfectly sized for edge compute devices like a Raspberry Pi.
* **Hardware Abstraction Layer (HAL):** Implemented a universal `DroneController` trait. The AI no longer needs to know if it is operating inside a Rust simulator, a ROS2/Gazebo 3D environment, or on a physical drone over Wi-Fi. It simply ingests standardized `DroneSensors` data and outputs mechanical `Action` commands.

## 🛠️ Architecture

The project is structured as a Cargo Workspace to separate concerns:

* **`/physics`:** The continuous physics engine handling momentum, drag, and Lidar raycasting.
* **`/environment`:** The spatial logic, boundary enforcement, and reward state generation.
* **`/rl`:** The Deep Learning brain, neural networks, memory buffers, and gradient descent logic.
* **`/simulator`:** The HAL and execution engine that connects the AI to either simulated or physical hardware.

## 💻 How to Run

To run the lightweight Inference Engine using the pre-trained weights, ensure you have Rust installed and run:

```bash
cargo run -p simulator
```

## ⚖️ License & Intellectual Property

**Copyright (c) 2026. All Rights Reserved.**

This repository, its source code, its architectural design, and the underlying concepts represent proprietary intellectual property.

**Strictly Protected:** You may not use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this Software or its concepts in any capacity. No individual, corporation, or entity may utilize this project for personal, educational, or commercial purposes without explicit, prior written consent from the author.

If you are interested in discussing, licensing, or collaborating on this technology, please contact the author directly.