// environment/src/lib.rs

use physics::{BoundingBox, DroneBody, Vector2D};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

// UPGRADED STATE: 9 Dimensions (Sensors + Physics + Objective)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub sens_up: f32,
    pub sens_down: f32,
    pub sens_left: f32,
    pub sens_right: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub target_dx: f32,
    pub target_dy: f32,
    pub has_package: f32,
}

impl State {
    // A helper function to make passing this into our Neural Network MUCH cleaner
    pub fn to_array(&self) -> [f32; 9] {
        [
            self.sens_up, self.sens_down, self.sens_left, self.sens_right,
            self.vel_x, self.vel_y,
            self.target_dx, self.target_dy,
            self.has_package,
        ]
    }
}

pub trait Environment {
    fn reset(&mut self) -> State;
    fn step(&mut self, action: Action) -> (State, f32, bool);
}

pub struct ContinuousWarehouse {
    pub drone: DroneBody,
    pub bounds: BoundingBox,
    pub start_pos: Vector2D,
    pub pickup_pos: Vector2D,
    pub delivery_pos: Vector2D,
    pub has_package: bool,
    pub target_radius: f32, // How close we need to get to consider the target "reached"
}

impl ContinuousWarehouse {
    pub fn new() -> Self {
        Self {
            drone: DroneBody::new(2.0, 2.0),
            // A realistic 20x10 meter warehouse bounding box
            bounds: BoundingBox { min_x: 0.0, max_x: 20.0, min_y: 0.0, max_y: 10.0 },
            start_pos: Vector2D::new(2.0, 2.0),
            pickup_pos: Vector2D::new(10.0, 8.0),
            delivery_pos: Vector2D::new(18.0, 2.0),
            has_package: false,
            target_radius: 1.0, // Must get within 1 meter of the target
        }
    }

    fn get_state(&self) -> State {
        let sensors = self.drone.read_sensors(&self.bounds);
        let target = if self.has_package { &self.delivery_pos } else { &self.pickup_pos };

        // Calculate direction to current target
        let dx = target.x - self.drone.position.x;
        let dy = target.y - self.drone.position.y;
        let dist = (dx * dx + dy * dy).sqrt().max(0.001); // Prevent division by zero

        State {
            sens_up: sensors.dist_up,
            sens_down: sensors.dist_down,
            sens_left: sensors.dist_left,
            sens_right: sensors.dist_right,
            vel_x: self.drone.velocity.x,
            vel_y: self.drone.velocity.y,
            target_dx: dx / dist, // Normalized direction
            target_dy: dy / dist, // Normalized direction
            has_package: if self.has_package { 1.0 } else { 0.0 },
        }
    }
}

impl Environment for ContinuousWarehouse {
    fn reset(&mut self) -> State {
        let mut rng = rand::thread_rng();
        
        self.drone = DroneBody::new(self.start_pos.x, self.start_pos.y);
        
        // DOMAIN RANDOMIZATION: Randomize the drone's mass by +/- 20% every episode!
        self.drone.mass = rng.gen_range(0.8..1.2); 
        
        self.has_package = false;
        self.get_state()
    }

    fn step(&mut self, action: Action) -> (State, f32, bool) {
        let thrust = 0.5;
        let mut force_x = 0.0;
        let mut force_y = 0.0;

        // Apply continuous thrust
        match action {
            Action::Up => force_y -= thrust,
            Action::Down => force_y += thrust,
            Action::Left => force_x -= thrust,
            Action::Right => force_x += thrust,
        }

        self.drone.apply_force(force_x, force_y);

        // Collision Check using Physics bounds
        if self.drone.is_colliding(&self.bounds) {
            return (self.get_state(), -100.0, true);
        }

        let mut reward = -1.0; 

        // Target Proximity Check (Euclidean Distance)
        let target = if self.has_package { &self.delivery_pos } else { &self.pickup_pos };
        let dx = target.x - self.drone.position.x;
        let dy = target.y - self.drone.position.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < self.target_radius {
            if !self.has_package {
                self.has_package = true;
                reward += 50.0;
                println!("    [Sim Log] Package Picked Up! Target is now Delivery.");
            } else {
                return (self.get_state(), 100.0, true);
            }
        }

        (self.get_state(), reward, false)
    }
}