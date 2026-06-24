// simulator/src/main.rs

use environment::{Environment, ContinuousWarehouse}; // Updated!
use rl::agent::DqnAgent;
use rl::replay_buffer::Experience;

// Import Burn's Autodiff wrapper and the NdArray (CPU) backend
use burn::backend::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};

// Define our specific backend type: A CPU tensor backend with gradient tracking enabled
type Backend = Autodiff<NdArray>;

fn main() {
    println!("=== Autonomous Warehouse Drone: Training Phase ===");

    // 1. Initialize Device and Environment
    let device = NdArrayDevice::Cpu;
    let mut env = ContinuousWarehouse::new(); // Updated!

    // 2. Initialize the Agent (Replay Buffer capacity: 10,000)
    let mut agent = DqnAgent::<Backend>::new(&device, 10000);

    let num_episodes = 2500;
    let max_steps_per_episode = 300; // UPGRADED: Give the drone time to accelerate and brake!

    for episode in 1..=num_episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps_taken = 0;

        for _ in 0..max_steps_per_episode {
            steps_taken += 1;

            // A. AI Decides Action
            let action = agent.select_action(state, &device);

            // B. Environment Reacts
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;

            // C. Agent Remembers
            agent.memory.push(Experience {
                state,
                action,
                reward,
                next_state,
                done,
            });

            // D. Agent Learns (Backpropagation)
            agent.train_step(&device);

            state = next_state;

            if done {
                break;
            }
        }

        // 3. Episode Cleanup & Network Syncing
        agent.decay_epsilon();

        if episode % agent.hyperparams.target_update_freq == 0 {
            agent.update_target_network();
        }

        // 4. Log Progress every 10 episodes
        if episode % 10 == 0 {
            println!(
                "Episode {:>3} | Steps: {:>2} | Reward: {:>6.1} | Epsilon: {:.3}",
                episode, steps_taken, total_reward, agent.epsilon
            );
        }
    }

    println!("=== Training Complete! ===");
}
