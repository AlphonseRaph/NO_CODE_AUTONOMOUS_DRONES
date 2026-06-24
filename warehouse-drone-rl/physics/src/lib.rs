// physics/src/lib.rs

#[derive(Debug, Clone, Copy)]
pub struct Vector2D {
    pub x: f32,
    pub y: f32,
}

impl Vector2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

pub struct DroneBody {
    pub position: Vector2D,
    pub velocity: Vector2D,
    pub mass: f32,
    pub max_speed: f32,
    pub drag: f32, // Air resistance/friction
}

impl DroneBody {
    pub fn new(start_x: f32, start_y: f32) -> Self {
        Self {
            position: Vector2D::new(start_x, start_y),
            velocity: Vector2D::new(0.0, 0.0),
            mass: 1.0,
            max_speed: 2.0,
            drag: 0.85, // Retains 85% of velocity each step (simulates air resistance)
        }
    }

    // Apply thrust to change velocity, then update position
    pub fn apply_force(&mut self, force_x: f32, force_y: f32) {
        // F = ma -> a = F/m
        let accel_x = force_x / self.mass;
        let accel_y = force_y / self.mass;

        self.velocity.x += accel_x;
        self.velocity.y += accel_y;

        // Apply drag (friction)
        self.velocity.x *= self.drag;
        self.velocity.y *= self.drag;

        // Clamp to max speed
        self.velocity.x = self.velocity.x.clamp(-self.max_speed, self.max_speed);
        self.velocity.y = self.velocity.y.clamp(-self.max_speed, self.max_speed);

        // Update position based on velocity
        self.position.x += self.velocity.x;
        self.position.y += self.velocity.y;
    }
}

// Add to the bottom of physics/src/lib.rs

pub struct BoundingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

pub struct SensorReadings {
    pub dist_up: f32,
    pub dist_down: f32,
    pub dist_left: f32,
    pub dist_right: f32,
}

impl DroneBody {
    // Simulates raycasting to find the distance to the nearest warehouse boundary
    pub fn read_sensors(&self, bounds: &BoundingBox) -> SensorReadings {
        SensorReadings {
            dist_up: self.position.y - bounds.min_y,
            dist_down: bounds.max_y - self.position.y,
            dist_left: self.position.x - bounds.min_x,
            dist_right: bounds.max_x - self.position.x,
        }
    }

    // A helper to check if the drone has crashed into the wall
    pub fn is_colliding(&self, bounds: &BoundingBox) -> bool {
        self.position.x <= bounds.min_x
            || self.position.x >= bounds.max_x
            || self.position.y <= bounds.min_y
            || self.position.y >= bounds.max_y
    }
}