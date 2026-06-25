// simulator/src/hal.rs

use environment::Action;

/// The sensor data that both a REAL drone and a ROS2 drone must provide to our AI
pub struct DroneSensors {
    pub distance_front: f32,
    pub distance_back: f32,
    pub distance_left: f32,
    pub distance_right: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub payload_attached: bool,
}

/// The universal contract for controlling a drone.
/// We can implement this for a ROS2 Node, a physical DJI drone, or a Test Simulator.
pub trait DroneController {
    /// Reads the sensors (Lidar/Odometry) from the hardware/ROS2
    fn read_sensors(&self) -> DroneSensors;

    /// Sends a velocity command to the hardware/ROS2 (e.g., publishing cmd_vel)
    fn send_command(&mut self, action: Action);

    /// Helper to get the vector to the current target (Pickup or Delivery)
    fn get_target_vector(&self) -> (f32, f32);
}
