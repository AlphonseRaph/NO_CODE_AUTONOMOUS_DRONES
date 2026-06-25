// 

// simulator/src/main.rs

pub mod hal;

use burn::record::{CompactRecorder, Recorder};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData, ElementConversion};
use burn::module::Module;

use environment::{ContinuousWarehouse, Environment, Action, State};
use rl::network::{DqnModel, DqnModelConfig};
use hal::{DroneController, DroneSensors};

// 1. INFERENCE BACKEND: No Autodiff required! 
type Backend = NdArray; 

// --- HARDWARE ABSTRACTION IMPLEMENTATION ---
// If you move to ROS2, you simply replace this struct with a `Ros2Hardware` struct
// that reads from `/scan` and publishes to `/cmd_vel`!
struct SimulatedHardware {
    env: ContinuousWarehouse,
    current_state: State,
}

impl SimulatedHardware {
    fn new() -> Self {
        let mut env = ContinuousWarehouse::new();
        let current_state = env.reset();
        Self { env, current_state }
    }
}

impl DroneController for SimulatedHardware {
    fn read_sensors(&self) -> DroneSensors {
        DroneSensors {
            distance_front: self.current_state.sens_up,
            distance_back: self.current_state.sens_down,
            distance_left: self.current_state.sens_left,
            distance_right: self.current_state.sens_right,
            velocity_x: self.current_state.vel_x,
            velocity_y: self.current_state.vel_y,
            payload_attached: self.current_state.has_package == 1.0,
        }
    }

    fn send_command(&mut self, action: Action) {
        let (new_state, _, _) = self.env.step(action);
        self.current_state = new_state;
    }

    fn get_target_vector(&self) -> (f32, f32) {
        (self.current_state.target_dx, self.current_state.target_dy)
    }
}

fn main() {
    println!("===========================================");
    println!("=== DRONE OS: Booting Inference Engine ===");
    println!("===========================================\n");

    let device = NdArrayDevice::Cpu;

    // 2. Load the trained "Brain"
    println!("[SYSTEM] Loading Neural Network weights from 'drone_brain.mpk'...");
    let record = CompactRecorder::new()
        .load("drone_brain".into(), &device) // FIXED: Added `&device` right here!
        .expect("Failed to load drone_brain! Check your file path.");

    // 3. Initialize the Model with the saved weights
    let model: DqnModel<Backend> = DqnModelConfig::new()
        .init(&device)
        .load_record(record);

    println!("[SYSTEM] Brain loaded successfully. Establishing hardware connection...");

    // 4. Connect to the Drone via HAL
    let mut drone = SimulatedHardware::new();

    println!("[SYSTEM] Connection established. Commencing autonomous flight.\n");
    
    // 5. The Real-Time Flight Loop (Operating at roughly 10Hz in real life)
    for step in 1..=150 {
        // A. Read sensors from hardware
        let sensors = drone.read_sensors();
        let (tx, ty) = drone.get_target_vector();

        // B. Package data exactly as the neural network expects (9 dimensions)
        let state_array = [
            sensors.distance_front, sensors.distance_back, 
            sensors.distance_left, sensors.distance_right,
            sensors.velocity_x, sensors.velocity_y,
            tx, ty,
            if sensors.payload_attached { 1.0 } else { 0.0 }
        ];

        let state_tensor = Tensor::<Backend, 2>::from_data(
            TensorData::from([state_array]), 
            &device
        );

        // C. Forward Pass (Pure AI prediction! No epsilon randomness)
        let q_values = model.forward(state_tensor);
        let best_action_idx: i32 = q_values.argmax(1).into_scalar().elem();

        let action = match best_action_idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        };

        // D. Execute hardware command
        drone.send_command(action);

        // E. Telemetry output
        let target_str = if sensors.payload_attached { "Delivery Zone" } else { "Pickup Shelf" };
        let speed = (sensors.velocity_x.powi(2) + sensors.velocity_y.powi(2)).sqrt();
        
        println!(
            "Flight Time T+{:>3} | Action: {:<5} | Target: {:<13} | Speed: {:.2} m/s", 
            step, format!("{:?}", action), target_str, speed
        );

        // In a real drone, we would pace the loop here to match physical hardware speeds:
        // std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("\n=== Autonomous Mission Complete ===");
}