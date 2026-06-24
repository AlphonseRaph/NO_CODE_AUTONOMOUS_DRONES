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

## 🛠️ Architecture

The project is structured as a Cargo Workspace to separate concerns:

* **`/environment`:** The spatial logic, boundary enforcement, and reward state generation.

* **`/rl`:** The Deep Learning brain, neural networks, memory buffers, and gradient descent logic.

* **`/simulator`:** The execution loop that connects the brain to the simulated world.

## 💻 How to Run

To watch the agent train from scratch, ensure you have Rust installed and run the following command from the root directory:

```bash
cargo run -p simulator
```
##
⚖️ License & Intellectual Property
Copyright (c) 2026. All Rights Reserved.

This repository, its source code, its architectural design, and the underlying concepts represent proprietary intellectual property.

Strictly Protected: You may not use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this Software or its concepts in any capacity. No individual, corporation, or entity may utilize this project for personal, educational, or commercial purposes without explicit, prior written consent from the author.

If you are interested in discussing, licensing, or collaborating on this technology, please contact the author directly.