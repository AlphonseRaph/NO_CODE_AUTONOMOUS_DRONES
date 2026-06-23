// rl/src/replay_buffer.rs

use environment::{Action, State};
use rand::seq::SliceRandom;
use rand::rngs::ThreadRng;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Experience {
    pub state: State,
    pub action: Action,
    pub reward: f32,
    pub next_state: State,
    pub done: bool,
}

pub struct ReplayBuffer {
    capacity: usize,
    buffer: VecDeque<Experience>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    // Push a new memory into the buffer. 
    // If full, the oldest memory is dropped.
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    // Grab a random batch of memories for training
    // Grab a random batch of memories for training
    pub fn sample(&self, batch_size: usize, rng: &mut ThreadRng) -> Vec<Experience> {
        // If we don't have enough data yet, return everything we have
        let actual_size = batch_size.min(self.buffer.len());
        
        // Generate a list of valid indices and shuffle them
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(rng);
        
        // Map the shuffled indices back into cloned Experiences
        indices.into_iter()
            .take(actual_size)
            .map(|i| self.buffer[i].clone())
            .collect()
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}