// environment/src/lib.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

// UPGRADED STATE: 5 Dimensions (Drone Pos + Worker Pos + Package Flag)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub x: f32,
    pub y: f32,
    pub worker_x: f32,
    pub worker_y: f32,
    pub has_package: f32, // 0.0 = No, 1.0 = Yes
}

pub trait Environment {
    fn reset(&mut self) -> State;
    fn step(&mut self, action: Action) -> (State, f32, bool);
}

#[derive(Debug, Clone)]
pub struct Worker {
    pub x: usize,
    pub y: usize,
    pub min_y: usize,
    pub max_y: usize,
    pub moving_down: bool,
}

impl Worker {
    pub fn move_step(&mut self) {
        if self.moving_down {
            self.y += 1;
            if self.y >= self.max_y {
                self.moving_down = false;
            }
        } else {
            self.y -= 1;
            if self.y <= self.min_y {
                self.moving_down = true;
            }
        }
    }
}

pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub drone_pos: (usize, usize),
    pub start_pos: (usize, usize),
    pub pickup_pos: (usize, usize),    // P Location
    pub delivery_pos: (usize, usize),  // D Location (Old Goal)
    pub obstacles: Vec<(usize, usize)>,
    pub worker: Worker,
    pub has_package: bool,             // Current drone cargo status
}

impl GridWorld {
    pub fn new() -> Self {
        let width = 20;
        let height = 7;
        let start_pos = (1, 1);
        let pickup_pos = (5, 3);     // Item location
        let delivery_pos = (13, 5);  // Final dropping zone

        let mut obstacles = Vec::new();
        
        // Outer boundaries
        for x in 0..width {
            obstacles.push((x, 0));
            obstacles.push((x, height - 1));
        }
        for y in 0..height {
            obstacles.push((0, y));
            obstacles.push((width - 1, y));
        }
        
        // The static shelf
        obstacles.push((14, 2));
        obstacles.push((14, 3));
        obstacles.push((14, 4));

        let worker = Worker {
            x: 8,
            y: 1,
            min_y: 1,
            max_y: 5,
            moving_down: true,
        };

        Self {
            width,
            height,
            drone_pos: start_pos,
            start_pos,
            pickup_pos,
            delivery_pos,
            obstacles,
            worker,
            has_package: false,
        }
    }

    fn get_state(&self) -> State {
        State {
            x: self.drone_pos.0 as f32,
            y: self.drone_pos.1 as f32,
            worker_x: self.worker.x as f32,
            worker_y: self.worker.y as f32,
            has_package: if self.has_package { 1.0 } else { 0.0 },
        }
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> State {
        self.drone_pos = self.start_pos;
        self.worker.y = self.worker.min_y;
        self.worker.moving_down = true;
        self.has_package = false; // Reset payload status
        
        self.get_state()
    }

    fn step(&mut self, action: Action) -> (State, f32, bool) {
        let (mut nx, mut ny) = self.drone_pos;

        match action {
            Action::Up => { if ny > 0 { ny -= 1 } },
            Action::Down => { ny += 1 },
            Action::Left => { if nx > 0 { nx -= 1 } },
            Action::Right => { nx += 1 },
        }

        self.worker.move_step();

        // 1. Structural Obstacle Collision
        if self.obstacles.contains(&(nx, ny)) {
            return (self.get_state(), -50.0, true);
        }

        // 2. Worker Collision
        if nx == self.worker.x && ny == self.worker.y {
            return (self.get_state(), -100.0, true);
        }

        self.drone_pos = (nx, ny);
        let mut total_step_reward = -1.0;

        // 3. Sequential Task Evaluation
        if !self.has_package {
            // Task A: Go to pickup location
            if self.drone_pos == self.pickup_pos {
                self.has_package = true;
                total_step_reward += 50.0; // Intermediate reward bonus for picking up package
                println!("    [Sim Log] Package Picked Up! Headed to delivery...");
            }
        } else {
            // Task B: Go to delivery location
            if self.drone_pos == self.delivery_pos {
                return (self.get_state(), 100.0, true); // Ultimate delivery goal reached!
            }
        }

        (self.get_state(), total_step_reward, false)
    }
}