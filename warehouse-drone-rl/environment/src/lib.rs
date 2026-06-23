// environment/src/lib.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

// We return f32 because Neural Networks expect continuous float tensors as input
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub x: f32,
    pub y: f32,
}

pub trait Environment {
    fn reset(&mut self) -> State;
    fn step(&mut self, action: Action) -> (State, f32, bool);
}

// Our concrete warehouse environment
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub drone_pos: (usize, usize),
    pub start_pos: (usize, usize),
    pub goal_pos: (usize, usize),
    pub obstacles: Vec<(usize, usize)>,
}

impl GridWorld {
    pub fn new() -> Self {
        // Warehouse dimensions
        let width = 20;
        let height = 7;
        
        // S = Start, G = Goal
        let start_pos = (1, 1);
        let goal_pos = (13, 5);

        let mut obstacles = Vec::new();
        
        // 1. Add outer boundary walls (#)
        for x in 0..width {
            obstacles.push((x, 0));
            obstacles.push((x, height - 1));
        }
        for y in 0..height {
            obstacles.push((0, y));
            obstacles.push((width - 1, y));
        }
        
        // 2. Add the inner shelf block (#)
        obstacles.push((14, 2));
        obstacles.push((14, 3));
        obstacles.push((14, 4));

        Self {
            width,
            height,
            drone_pos: start_pos,
            start_pos,
            goal_pos,
            obstacles,
        }
    }

    // Helper to format the internal usize coordinates into the f32 State struct
    fn get_state(&self) -> State {
        State {
            x: self.drone_pos.0 as f32,
            y: self.drone_pos.1 as f32,
        }
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> State {
        self.drone_pos = self.start_pos;
        self.get_state()
    }

    fn step(&mut self, action: Action) -> (State, f32, bool) {
        let (mut nx, mut ny) = self.drone_pos;

        // Calculate intended next position
        match action {
            Action::Up => { if ny > 0 { ny -= 1 } },
            Action::Down => { ny += 1 },
            Action::Left => { if nx > 0 { nx -= 1 } },
            Action::Right => { nx += 1 },
        }

        // Apply Reward Function Rules:
        
        // Rule 1: Did we hit a wall or shelf?
        if self.obstacles.contains(&(nx, ny)) {
            // Hit obstacle (-50 reward, end episode). Drone does not move.
            return (self.get_state(), -50.0, true);
        }

        // Safely move the drone
        self.drone_pos = (nx, ny);

        // Rule 2: Did we reach the goal?
        if self.drone_pos == self.goal_pos {
            // Reached goal (+100 reward, end episode).
            return (self.get_state(), 100.0, true);
        }

        // Rule 3: Normal movement step cost
        // (-1 reward, episode continues)
        (self.get_state(), -1.0, false)
    }
}