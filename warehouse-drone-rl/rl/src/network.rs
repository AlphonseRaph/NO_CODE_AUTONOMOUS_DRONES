// rl/src/network.rs

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

// 1. The Configuration for initializing the model
#[derive(Debug)]
pub struct DqnModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

// rl/src/network.rs (Update only the `new` function)

impl DqnModelConfig {
    pub fn new() -> Self {
        Self {
            input_size: 9,      // UPGRADED: 9 Sensor Inputs
            hidden_size: 256,   // UPGRADED: Deepened network for physics calculations
            output_size: 4,     // Actions remain the same: [Up, Down, Left, Right]
        }
    }
    
    // ... keep the rest of the file exactly as it is!
    
    pub fn init<B: Backend>(&self, device: &B::Device) -> DqnModel<B> {
        DqnModel {
            layer1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            layer2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            output: LinearConfig::new(self.hidden_size, self.output_size).init(device),
            relu: Relu::new(),
        }
    }
}

// 2. The Neural Network Model
#[derive(Module, Debug)]
pub struct DqnModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
    relu: Relu,
}

impl<B: Backend> DqnModel<B> {
    // 3. The Forward Pass (calculating Q-values from state)
    // The input tensor has shape [batch_size, 2]
    // The output tensor has shape [batch_size, 4]
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(state);
        let x = self.relu.forward(x);
        
        let x = self.layer2.forward(x);
        let x = self.relu.forward(x);
        
        // No activation on the final layer because Q-values can be negative (e.g., -50 for wall)
        self.output.forward(x) 
    }
}