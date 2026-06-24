// rl/src/agent.rs

// rl/src/agent.rs

use burn::optim::{Adam, AdamConfig, Optimizer}; 
use burn::optim::adaptor::OptimizerAdaptor;
use crate::network::{DqnModel, DqnModelConfig};
use crate::replay_buffer::{Experience, ReplayBuffer};
use burn::module::{Module, AutodiffModule}; 
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, ElementConversion, Int}; 
use environment::{Action, State};
use rand::Rng;

// ... (rest of your code)

pub struct DqnHyperparams {
    pub batch_size: usize,
    pub lr: f64,
    pub gamma: f32,             
    pub epsilon_start: f32,     
    pub epsilon_min: f32,       
    pub epsilon_decay: f32,     
    pub target_update_freq: usize, 
}

impl Default for DqnHyperparams {
    fn default() -> Self {
        Self {
            batch_size: 64,
            lr: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            target_update_freq: 10, 
        }
    }
}

pub struct DqnAgent<B: AutodiffBackend> {
    pub policy_net: DqnModel<B>,
    pub target_net: DqnModel<B::InnerBackend>, 
    
    // THE ULTIMATE FIX: Adam tracks momentum without gradients, so it needs the InnerBackend!
    pub optimizer: OptimizerAdaptor<Adam<B::InnerBackend>, DqnModel<B>, B>, 
    
    pub memory: ReplayBuffer,
    pub hyperparams: DqnHyperparams,
    pub epsilon: f32,
    pub step_count: usize,
}

impl<B: AutodiffBackend> DqnAgent<B> {
    
    pub fn new(device: &B::Device, capacity: usize) -> Self {
        let config = DqnModelConfig::new();
        let hyperparams = DqnHyperparams::default();
        
        let policy_net = config.init::<B>(device);
        let target_net = config.init::<B::InnerBackend>(device);
        
        let optim_config = AdamConfig::new();
        let optimizer = optim_config.init::<B, DqnModel<B>>(); 

        Self {
            policy_net,
            target_net,
            optimizer,
            memory: ReplayBuffer::new(capacity),
            hyperparams,
            epsilon: 1.0, 
            step_count: 0,
        }
    } 

    // 3. Epsilon-Greedy Action Selection
    pub fn select_action(&mut self, state: State, device: &B::Device) -> Action {
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f32>() < self.epsilon {
            return match rng.gen_range(0..4) {
                0 => Action::Up,
                1 => Action::Down,
                2 => Action::Left,
                _ => Action::Right,
            };
        }

        // SO CLEAN: Use the to_array() helper!
        let state_data = TensorData::from([state.to_array()]);
        let state_tensor = Tensor::<B, 2>::from_data(state_data, device);
        
        let q_values = self.policy_net.forward(state_tensor);
        let best_action_idx: i32 = q_values.argmax(1).into_scalar().elem();
        
        match best_action_idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }

    pub fn decay_epsilon(&mut self) {
        if self.epsilon > self.hyperparams.epsilon_min {
            self.epsilon *= self.hyperparams.epsilon_decay;
        }
    }

    pub fn update_target_network(&mut self) {
        self.target_net = self.policy_net.clone().valid(); 
    }

   // 6. The Deep Q-Learning Training Algorithm
  pub fn train_step(&mut self, device: &B::Device) {
        if self.memory.len() < self.hyperparams.batch_size { return; }

        let batch_size = self.hyperparams.batch_size;
        let batch = self.memory.sample(batch_size, &mut rand::thread_rng());

        let mut state_arr: Vec<f32> = Vec::new();
        let mut action_arr: Vec<i32> = Vec::new();
        let mut reward_arr: Vec<f32> = Vec::new();
        let mut next_state_arr: Vec<f32> = Vec::new();
        let mut done_arr: Vec<f32> = Vec::new();

        for exp in batch {
            // SO CLEAN: Append the whole 9-element array instantly!
            state_arr.extend_from_slice(&exp.state.to_array());
            next_state_arr.extend_from_slice(&exp.next_state.to_array());
            
            action_arr.push(match exp.action {
                Action::Up => 0,
                Action::Down => 1,
                Action::Left => 2,
                Action::Right => 3,
            });
            reward_arr.push(exp.reward);
            done_arr.push(if exp.done { 0.0 } else { 1.0 }); 
        }

        // Reshape to [batch_size, 9]
        let states = Tensor::<B, 1>::from_floats(state_arr.as_slice(), device).reshape([batch_size, 9]);
        let next_states = Tensor::<B::InnerBackend, 1>::from_floats(next_state_arr.as_slice(), device).reshape([batch_size, 9]);
        
        let actions = Tensor::<B, 1, Int>::from_ints(action_arr.as_slice(), device).reshape([batch_size, 1]);
        let rewards = Tensor::<B, 1>::from_floats(reward_arr.as_slice(), device).reshape([batch_size, 1]);
        let dones = Tensor::<B, 1>::from_floats(done_arr.as_slice(), device).reshape([batch_size, 1]);

        let current_q_all = self.policy_net.forward(states);
        let current_q = current_q_all.gather(1, actions); 

        let next_q_all = self.target_net.forward(next_states);
        let max_next_q = next_q_all.max_dim(1);
        let max_next_q_diff = Tensor::<B, 2>::from_inner(max_next_q);

        let target_q = rewards + (max_next_q_diff * dones) * self.hyperparams.gamma;
        let target_q = target_q.detach(); 

        let diff = current_q - target_q;
        let loss = (diff.clone() * diff).mean(); 

        let raw_grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(raw_grads, &self.policy_net);
        self.policy_net = self.optimizer.step(self.hyperparams.lr, self.policy_net.clone(), grads);
    }
} // <-- This is the ONLY brace that closes the impl block now!