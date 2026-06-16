import sys
import math
import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QPainter, QPixmap, QPen, QColor
from parking_render import render_parking_scene

class ParkingEnv(gym.Env, QWidget):
    target_theta = -math.pi / 2
    start_x = 475.0
    start_y = 200.0
    # Parked car coordinates
    parked_car_x = 630.0
    # ist was 130 and 700 originally
    parked_car_ys = (160, 670)
    parked_car_width = 100
    parked_car_height = 200
    # Target parking spot center
    target_x = 630.0
    target_y = 415.0

    def __init__(
        self,
        render_mode=None,
        collision_penalty=-20.0,
        distance_penalty_weight=0.5,
        alignment_penalty_weight=1.0,
        baseline_time_penalty=-0.01,
        standing_reward_weight=0.2,
        delta_reward_weight=2.0,
        success_angle_threshold=0.02,
        success_distance_threshold=10.0,
        success_reward=150.0,
        enable_training_spawns=False,
    ):
        super(ParkingEnv, self).__init__()
        QWidget.__init__(self)

        self.render_mode = render_mode
        self.collision_penalty = float(collision_penalty)
        self.distance_penalty_weight = float(distance_penalty_weight)
        self.alignment_penalty_weight = float(alignment_penalty_weight)
        self.baseline_time_penalty = float(baseline_time_penalty)
        self.standing_reward_weight = float(standing_reward_weight)
        self.delta_reward_weight = float(delta_reward_weight)
        self.success_angle_threshold = float(success_angle_threshold)
        self.success_distance_threshold = float(success_distance_threshold)
        self.success_reward = float(success_reward)
        self.enable_training_spawns = bool(enable_training_spawns)
        self.setFixedSize(800, 800)
        self.prev_action = 0
        self.show_car_points_debug = False
        self.show_score_overlay = True
        self.show_front_wheels = True
        self.show_lidar = True
        self.show_player_center_dot = False
        self.traffic_visible = True
        self.manual_keys = {"up": False, "down": False, "left": False, "right": False}
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # RL Spaces
        self.action_space = spaces.Discrete(9)
        # Observation = 9 scalar features + 12 lidar readings.
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)

        # Load Assets (Assuming same directory)
        self._car_pixmap = QPixmap("red_car.png")
        self._tree_pixmap = QPixmap("tree.png")
        self._parked_car_pixmap = QPixmap("yellow_car.png")

        self._build_car_points_grid()

        self.reset()

    def _build_car_points_grid(self):
        """Builds a fixed sampling grid in the car's local coordinate space."""
        self.grid_rows = 5
        self.grid_cols = 3
        self.car_points = []
        # Create a grid of points in the car's local coordinate system (-50 to 50, -100 to 100)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Evenly space points across the car's width and height
                lx = -50 + (j * 100 / (self.grid_cols - 1))
                ly = -100 + (i * 200 / (self.grid_rows - 1))
                self.car_points.append((lx, ly))
        self.total_points = len(self.car_points)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Per-reset override: options={"training_spawns": True/False}
        # Falls back to the instance default when not provided.
        training_spawns_enabled = self.enable_training_spawns
        # Default spawn on the right road lane.
        self.car_x = float(self.np_random.uniform(self.start_x - 25.0, self.start_x + 25.0))
        self.car_y = self.start_y
        self.car_theta = self.target_theta

        # Training-only random spawn variants.
        if training_spawns_enabled:
            # 40% of episodes: start at the left border of the parking lane,
            # with y between the two parked cars and a slight tilt from vertical.
            if self.np_random.random() < 0.4:
                self._spawn_at_parking_lane_border()

            # 10% of episodes: start in the parking lane with a random position and angle.
            if self.np_random.random() < 0.1:
                self._spawn_in_parking_lane()

        # 2. State resets
        self.car_delta = 0.0
        self.L = 60.0
        self.prev_coverage = 0.0
        self.max_coverage_level = 0
        self.prev_action = 0  # To track "twitching"
        self.steps = 0
        self.prev_coverage = self.compute_coverage()
        
        self.traffic_x = 325.0
        self.traffic_y = 800.0


        if self.render_mode == "human":
            self.show()
            
        return self._get_obs(), {}

    def _spawn_at_parking_lane_border(self):
        # Try multiple randomized poses and keep the first collision-free one.
        spawn_x = 550.0
        y1 = self.parked_car_ys[0] + self.parked_car_height / 2 + 40.0
        y2 = self.parked_car_ys[1] - self.parked_car_height / 2 - 40.0
        max_attempts = 25
        for _ in range(max_attempts):
            self.car_x = spawn_x
            self.car_y = float(self.np_random.uniform(y1, y2))
            theta_offset = math.radians(float(self.np_random.uniform(-12.0, 12.0)))
            self.car_theta = self.target_theta + theta_offset

            if not self._check_collisions():
                return

        # Fallback if all sampled poses collide.
        self.car_x = self.start_x
        self.car_y = self.start_y
        self.car_theta = self.target_theta

    def _spawn_in_parking_lane(self):
        # Try multiple randomized poses and keep the first collision-free one.
        x1 = 585.0
        x2 = 675.0
        y1 = self.parked_car_ys[0] + self.parked_car_height / 2 + 35.0
        y2 = self.parked_car_ys[1] - self.parked_car_height / 2 - 35.0
        min_perturbation = math.radians(3.0)
        max_perturbation = math.radians(15.0)

        max_attempts = 25
        for _ in range(max_attempts):
            self.car_x = float(self.np_random.uniform(x1, x2))
            self.car_y = float(self.np_random.uniform(y1, y2))

            # Keep a minimum angular offset so the spawn is never perfectly straight.
            magnitude = self.np_random.uniform(min_perturbation, max_perturbation)
            sign = self.np_random.choice([-1, 1])
            theta_offset = magnitude * sign
            self.car_theta = self.target_theta + theta_offset

            if not self._check_collisions():
                return

        # Fallback if all sampled poses collide.
        self.car_x = self.start_x
        self.car_y = self.start_y
        self.car_theta = self.target_theta

    def set_training_spawns_enabled(self, enabled):
        self.enable_training_spawns = bool(enabled)

    def set_training_spawns_disabled(self):
        self.enable_training_spawns = False

    def _place_random_in_lane_collision_free(self, max_attempts=50):
        x1 = 585.0
        x2 = 675.0
        y1 = self.parked_car_ys[0] + self.parked_car_height / 2 + 35.0
        y2 = self.parked_car_ys[1] - self.parked_car_height / 2 - 35.0

        for _ in range(max_attempts):
            self.car_x = float(self.np_random.uniform(x1, x2))
            self.car_y = float(self.np_random.uniform(y1, y2))
            theta_offset = math.radians(float(self.np_random.uniform(-15.0, 15.0)))
            self.car_theta = self.target_theta + theta_offset
            self.car_delta = 0.0
            self.prev_v = 0.0

            if not self._check_collisions():
                return True

        self.car_x = self.start_x
        self.car_y = self.start_y
        self.car_theta = self.target_theta
        self.car_delta = 0.0
        self.prev_v = 0.0
        return False

    def _apply_manual_control(self, dt):
        v = 0.0

        if self.manual_keys["up"] and not self.manual_keys["down"]:
            v = 100.0
        elif self.manual_keys["down"] and not self.manual_keys["up"]:
            v = -100.0

        steer_dir = 0
        if self.manual_keys["left"] and not self.manual_keys["right"]:
            steer_dir = -1
        elif self.manual_keys["right"] and not self.manual_keys["left"]:
            steer_dir = 1

        self.prev_v = v
        self.car_x += v * math.cos(self.car_theta) * dt
        self.car_y += v * math.sin(self.car_theta) * dt
        self.car_theta += (v / self.L) * math.tan(self.car_delta) * dt
        self.car_delta = max(
            -math.radians(30),
            min(math.radians(30), self.car_delta + steer_dir * math.radians(45) * dt),
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            self.manual_keys["up"] = True
        if event.key() == Qt.Key.Key_Down:
            self.manual_keys["down"] = True
        if event.key() == Qt.Key.Key_Left:
            self.manual_keys["left"] = True
        if event.key() == Qt.Key.Key_Right:
            self.manual_keys["right"] = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            self.manual_keys["up"] = False
        if event.key() == Qt.Key.Key_Down:
            self.manual_keys["down"] = False
        if event.key() == Qt.Key.Key_Left:
            self.manual_keys["left"] = False
        if event.key() == Qt.Key.Key_Right:
            self.manual_keys["right"] = False
        super().keyReleaseEvent(event)

    def step(self, action):
        self.steps += 1
        dt = 0.016

        # 1. Physics Update
        if action < 3:
            v = 100.0     # Forward
        elif action < 6:
            v = -100.0    # Reverse
        else:
            v = 0.0       # Stationary Steering / Full Brake

        self.prev_v = v
        # Steering: -1 = left, 0 = straight, +1 = right (works for all 9 actions)
        steer_dir = (action % 3) - 1

        self.car_x += v * math.cos(self.car_theta) * dt
        self.car_y += v * math.sin(self.car_theta) * dt
        self.car_theta += (v / self.L) * math.tan(self.car_delta) * dt

        # Steering angle update (clamped to ±30°)
        self.car_delta = max(-math.radians(30), min(math.radians(30),
                           self.car_delta + steer_dir * math.radians(45) * dt))

        # 2. Traffic Update
        self.traffic_y -= 80.0 * dt
        if self.traffic_y < -150: self.traffic_y = 950.0

        # 3. Reward Calculation
        coverage = self.compute_coverage()
        reward = self.baseline_time_penalty  # Baseline time penalty

        if coverage > 0:
            # Existence reward: incentivises staying in the spot each step
            standing_reward = coverage * self.standing_reward_weight
            # Progress reward: small bonus for increasing coverage
            delta_reward = (coverage - self.prev_coverage) * self.delta_reward_weight
            reward += standing_reward + delta_reward

        self.prev_coverage = coverage

        if coverage > 0.95:
            # Alignment penalty: penalises crooked parking once mostly inside
            angle_diff = math.atan2(math.sin(self.car_theta - self.target_theta),
                            math.cos(self.car_theta - self.target_theta))
            alignment_penalty = (
                -(abs(angle_diff) / (math.pi / 2))
                * self.alignment_penalty_weight
                * coverage
            )
            reward += alignment_penalty

        if coverage < 0.1 or coverage > 0.95:
            # Distance penalty: guides the car toward the target when far from the spot
            dist = math.hypot(self.car_x - self.target_x, self.car_y - self.target_y)
            reward -= (dist / 800.0) * self.distance_penalty_weight

        # 4. Terminal Conditions
        collision = self._check_collisions()
        success = self._check_success()
        
        terminated = False
        if success:
            print("Success! Car parked correctly.")
            reward = self.success_reward  # Configurable success reward
            terminated = True
        elif collision:
            reward = -self.collision_penalty  # Configurable collision penalty
            terminated = True

        truncated = self.steps >= 500

        if self.render_mode == "human":
            self.update()
            QApplication.processEvents()

        return self._get_obs(), reward, terminated, truncated, {}
    def _get_obs(self):
        # 1. Constants and Normalization
        scale = 800.0

        # 2. Build the Core Vector
        obs = np.array([
            # 1-2: Absolute Position
            self.car_x / scale, 
            self.car_y / scale,
            
            # 3-4: Orientation (Sin/Cos prevents 0/2π jump)
            math.sin(self.car_theta),
            math.cos(self.car_theta),
            
            # 5-6: Relative to Target Spot
            (self.target_x - self.car_x) / scale,
            (self.target_y - self.car_y) / scale,
            
            # 7: Distance to Right Curb (Pavement)
            (710.0 - self.car_x) / scale,

            # 8: Internal Steering Angle (Normalized -1 to 1)
            self.car_delta / math.radians(30),

            # 9: Internal Velocity (Normalized -1 to 1)
            # Using self.prev_v from the step function
            getattr(self, 'prev_v', 0.0) / 100.0,
            
        ], dtype=np.float32)
        
        # 3. Append LIDAR (12 rays)
        # Total size: 9 + 12 = 21
        lidar_data = self.get_lidar_readings()
        return np.concatenate([obs, lidar_data]).astype(np.float32)
        
    
    def get_lidar_readings(self):
        """
        Casts 12 rays from the car's rectangle boundary to detect obstacles.
        Measures distance from the car's edge to obstacles (walls, parked cars).
        Returns normalized distances [0, 1].
        """
        num_rays = 12
        max_range = 300.0  # Pixels
        readings = []
        
        # 12 directions relative to the car's heading
        angles = [i * (2 * math.pi / num_rays) for i in range(num_rays)]
        
        # Obstacle definitions (matching your collision logic)
        parked_cars = [
            self._get_rect(self.parked_car_x, py, self.parked_car_width, self.parked_car_height)
            for py in self.parked_car_ys
        ]
        
        for ray_angle in angles:
            # Absolute angle of the ray
            world_angle = self.car_theta + ray_angle
            ray_start_x, ray_start_y = self._get_lidar_ray_start(world_angle)
            distance = self._get_lidar_hit_distance(
                ray_start_x,
                ray_start_y,
                world_angle,
                max_range,
                parked_cars,
            )
            
            # Normalize: 1.0 = Nothing detected, 0.0 = Right in front
            readings.append(distance / max_range)
            
        return readings
    

    def _check_collisions(self):
        player_poly = self._get_corners(self.car_x, self.car_y, self.car_theta)
        
        # --- NEW: Boundary Checks ---
        # Road starts at 250, Parking Lane ends at 710
        for corner in player_poly:
            if corner[0] < 250 or corner[0] > 710:
                return True
            if corner[1] < 0 or corner[1] > 800:
                return True

        # --- Existing: Parked Cars & Traffic ---
        for py in self.parked_car_ys:
            if self._poly_intersect(player_poly, self._get_rect(self.parked_car_x, py, self.parked_car_width, self.parked_car_height)): 
                return True
        
#        if self.traffic_visible:
#            if self._poly_intersect(player_poly, self._get_rect(325, self.traffic_y, 100, 200)): 
#                return True
                
        return False    
    
    def compute_target_coverage(self):
        # Target Box Boundaries: center ± half car dimensions
        t_min_x, t_max_x = self.target_x - 50.0, self.target_x + 50.0
        t_min_y, t_max_y = self.target_y - 100.0, self.target_y + 100.0
        
        cx, cy, alpha = self.car_x, self.car_y, self.car_theta + math.pi / 2
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        
        points_in_target = 0
        for lx, ly in self.car_points:
            wx = cx + lx * cos_a - ly * sin_a
            wy = cy + lx * sin_a + ly * cos_a
            
            if t_min_x <= wx <= t_max_x and t_min_y <= wy <= t_max_y:
                points_in_target += 1
                    
        return points_in_target / self.total_points

    def compute_coverage(self):
        """
        Calculates the percentage of the car's area currently 
        inside the parking lane using a sampling grid.
        """
        # 1. Define lane boundaries
        lane_min_x, lane_max_x = 550.0, 710.0
        
        # 2. Get current car pose
        cx, cy, theta = self.car_x, self.car_y, self.car_theta
        
        # 3. Calculate rotation angle (aligned with your _get_corners logic)
        # Since theta = -pi/2 is North, we offset to align the grid properly.
        alpha = theta + math.pi / 2
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        
        points_in_lane = 0
        
        # 4. Transform each grid point and check X-boundaries
        for lx, ly in self.car_points:
            # We only need the World X coordinate for a vertical parking lane
            # Formula: wx = cx + lx*cos(a) - ly*sin(a)
            wx = cx + lx * cos_a - ly * sin_a
            
            if lane_min_x <= wx <= lane_max_x:
                points_in_lane += 1
                
        # 5. Return coverage ratio [0.0, 1.0]
        return points_in_lane / self.total_points
    def _check_success(self):
        # 1. Get the current corners of the car
        player_corners = self._get_corners(self.car_x, self.car_y, self.car_theta)
        
        # 2. Define the parking lane boundaries
        lane_min_x = 550.0
        lane_max_x = 710.0
        
        # 3. Check if ALL corners are inside the X-boundaries of the lane
        # We use a small 5-pixel buffer to ensure the car isn't "pixel-perfect" on the line
        in_lane = all(lane_min_x + 5 <= corner[0] <= lane_max_x - 5 for corner in player_corners)
        
        # 4. Check orientation (more or less straight)
        # Using atan2 to handle the -pi to pi wrap-around correctly
        angle_diff = math.atan2(math.sin(self.car_theta - self.target_theta), 
                    math.cos(self.car_theta - self.target_theta))
#.      this was 0.05        
        # 0.05 radians is roughly 2.87 degrees of tolerance
#        print(f"Angle Diff for Success Check: {math.degrees(angle_diff):.2f}°")
        is_straight = abs(angle_diff) < self.success_angle_threshold

#       distance was 20
        dist = math.hypot(self.car_x - self.target_x, self.car_y - self.target_y)
        is_close_enough = dist < self.success_distance_threshold  # Within threshold pixels of the center
        
        # 5. Success if it's in the lane, straight, AND not currently colliding
        # (Though collision is usually handled in the step function first)
        return in_lane and is_straight and is_close_enough

    # --- SAT Collision Helpers ---
    def _get_corners(self, cx, cy, theta):
        alpha = theta + math.pi / 2
        hw, hh, cos_a, sin_a = 50, 100, math.cos(alpha), math.sin(alpha)
        return [(cx + lx*cos_a - ly*sin_a, cy + lx*sin_a + ly*cos_a) 
                for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]

    def _get_lidar_ray_start(self, world_angle):
        direction_x = math.cos(world_angle)
        direction_y = math.sin(world_angle)

        alpha = self.car_theta + math.pi / 2
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        # Rotate ray direction into the car's local rectangle space.
        local_dx = direction_x * cos_a + direction_y * sin_a
        local_dy = -direction_x * sin_a + direction_y * cos_a

        half_width = 50.0
        half_height = 100.0
        candidates = []
        epsilon = 1e-9

        if abs(local_dx) > epsilon:
            edge_distance = half_width / abs(local_dx)
            hit_y = edge_distance * local_dy
            if abs(hit_y) <= half_height + epsilon:
                candidates.append(edge_distance)

        if abs(local_dy) > epsilon:
            edge_distance = half_height / abs(local_dy)
            hit_x = edge_distance * local_dx
            if abs(hit_x) <= half_width + epsilon:
                candidates.append(edge_distance)

        ray_offset = min(candidates) if candidates else 0.0
        return (
            self.car_x + ray_offset * direction_x,
            self.car_y + ray_offset * direction_y,
        )

    def _get_lidar_hit_distance(self, start_x, start_y, world_angle, max_range, parked_cars):
        direction_x = math.cos(world_angle)
        direction_y = math.sin(world_angle)
        hit_distances = []

        wall_distance = self._ray_aabb_distance(
            start_x,
            start_y,
            direction_x,
            direction_y,
            250.0,
            710.0,
            0.0,
            800.0,
            use_exit_if_inside=True,
        )
        if wall_distance is not None:
            hit_distances.append(wall_distance)

        for rect in parked_cars:
            parked_distance = self._ray_aabb_distance(
                start_x,
                start_y,
                direction_x,
                direction_y,
                rect[0][0],
                rect[2][0],
                rect[0][1],
                rect[2][1],
            )
            if parked_distance is not None:
                hit_distances.append(parked_distance)

        if self.traffic_visible:
            traffic_distance = self._ray_aabb_distance(
                start_x,
                start_y,
                direction_x,
                direction_y,
                self.traffic_x - 50.0,
                self.traffic_x + 50.0,
                self.traffic_y - 100.0,
                self.traffic_y + 100.0,
            )
            if traffic_distance is not None:
                hit_distances.append(traffic_distance)

        if not hit_distances:
            return max_range

        return min(max_range, min(hit_distances))

    def _ray_aabb_distance(self, start_x, start_y, dir_x, dir_y, min_x, max_x, min_y, max_y, use_exit_if_inside=False):
        epsilon = 1e-9
        t_min = -math.inf
        t_max = math.inf

        if abs(dir_x) < epsilon:
            if start_x < min_x or start_x > max_x:
                return None
        else:
            tx1 = (min_x - start_x) / dir_x
            tx2 = (max_x - start_x) / dir_x
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))

        if abs(dir_y) < epsilon:
            if start_y < min_y or start_y > max_y:
                return None
        else:
            ty1 = (min_y - start_y) / dir_y
            ty2 = (max_y - start_y) / dir_y
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))

        if t_max < max(t_min, 0.0):
            return None

        if use_exit_if_inside and t_min <= epsilon <= t_max:
            return max(0.0, t_max)

        if t_min >= epsilon:
            return t_min

        return None

    def _get_rect(self, cx, cy, w, h):
        hw, hh = w/2, h/2
        return [(cx-hw, cy-hh), (cx+hw, cy-hh), (cx+hw, cy+hh), (cx-hw, cy+hh)]

    def _poly_intersect(self, poly_a, poly_b):
        for poly in (poly_a, poly_b):
            for i in range(len(poly)):
                p1, p2 = poly[i], poly[(i+1)%len(poly)]
                axis = (-(p2[1]-p1[1]), p2[0]-p1[0])
                norm = math.hypot(*axis)
                if norm == 0: continue
                axis = (axis[0]/norm, axis[1]/norm)
                min_a, max_a = self._project(poly_a, axis)
                min_b, max_b = self._project(poly_b, axis)
                if max_a < min_b or max_b < min_a: return False
        return True

    def _project(self, poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        render_parking_scene(painter, self)

    def calculate_final_score(self):
        # 1. Hard Failure: Collision gets an immediate zero
        if self._check_collisions():
            return 0.0

        # 3. Gather current metrics (Identical to your _check_success tracking)
        player_corners = self._get_corners(self.car_x, self.car_y, self.car_theta)
        
        angle_diff = math.atan2(math.sin(self.car_theta - self.target_theta), 
                                math.cos(self.car_theta - self.target_theta))
        
        dist = math.hypot(self.car_x - self.target_x, self.car_y - self.target_y)

        # 4. Define "Too Far Away" Horizons (The 0-point boundaries)
        MAX_DIST = 300.0          # 0 points if further than 300 pixels away
        MAX_ANGLE = math.pi / 2   # 0 points if perpendicular (90°) or worse
        MAX_LANE_ERR = 100.0      # 0 points if sticking out > 100 pixels into the road
        
        # 5. Compute Proximity Components (Continuous scale from 0.0 to 1.0)
        s_dist = max(0.0, 1.0 - (dist / MAX_DIST))
        s_align = max(0.0, 1.0 - (abs(angle_diff) / MAX_ANGLE))
        
        # Measure lane boundary violations using your success coordinates [555.0, 705.0]
        lane_min_x = 550.0 + 5
        lane_max_x = 710.0 - 5
        
        left_violation = max([lane_min_x - corner[0] for corner in player_corners] + [0.0])
        right_violation = max([corner[0] - lane_max_x for corner in player_corners] + [0.0])
        total_lane_err = left_violation + right_violation
        s_lane = max(0.0, 1.0 - (total_lane_err / MAX_LANE_ERR))
        
        # 6. Final Multiplicative Assembly
        # If the car is perfectly placed, components approach 1.0 -> Score approaches 100
        # If any single criteria is completely missed, the whole score drops to 0.0
        final_score = s_dist * s_align * s_lane * 100.0
        print(f"s_dist={s_dist:.4f}, s_align={s_align:.4f}, s_lane={s_lane:.4f}")
        return float(final_score)

    
    
    def test_score_system(self, num_trials=10, out_dir="score_test"):
        """
        Runs ``num_trials`` automated parking trials, scores each final pose,
        and saves a PNG snapshot of the widget for every trial under ``out_dir``.

        Each trial:
          1. Spawns the car randomly inside the lane (collision-free).
          2. Runs up to 500 steps using random actions.
          3. Stops early on success or collision.
          4. Captures the widget as a PNG and records the score.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        num_trials = int(num_trials)
        print(f"Running {num_trials} scoring trials → snapshots saved to '{out_dir}/'")
        print(f"{'Trial':>5} | {'X':>6} | {'Y':>6} | {'Angle':>7} | {'Outcome':<10} | {'Score':>6}")
        print("-" * 58)

        self.show()
        self.activateWindow()
        self.raise_()

        prev_show_lidar = self.show_lidar
        prev_show_front_wheels = self.show_front_wheels
        prev_show_player_center_dot = self.show_player_center_dot
        self.show_lidar = False
        self.show_front_wheels = False
        self.show_player_center_dot = False

        scores = []
        for trial in range(1, num_trials + 1):
            # --- Spawn ---
            placed = self._place_random_in_lane_collision_free()
            if not placed:
                print(f"Trial {trial:2d}: spawn failed, skipping.")
                continue

            self.steps = 0
            self.prev_coverage = self.compute_coverage()

            outcome = "truncated"
            for _ in range(500):
                action = self.action_space.sample()
                _, _, terminated, truncated, _ = self.step(action)
                QApplication.processEvents()
                if terminated:
                    outcome = "success" if self._check_success() else "collision"
                    break
                if truncated:
                    break

            final_score = self.calculate_final_score()
            scores.append(final_score)

            angle_diff = math.atan2(
                math.sin(self.car_theta - self.target_theta),
                math.cos(self.car_theta - self.target_theta),
            )

            # --- Snapshot (cropped to parking area) ---
            self.update()
            QApplication.processEvents()
            snapshot_path = os.path.join(out_dir, f"trial_{trial:02d}_score{final_score:.1f}.png")
            pad = 60
            crop_x = max(0, int(550 - pad))
            crop_y = max(0, int(self.parked_car_ys[0] + self.parked_car_height / 2 - pad))
            crop_w = min(800, int(710 + pad)) - crop_x
            crop_h = min(800, int(self.parked_car_ys[1] - self.parked_car_height / 2 + pad)) - crop_y
            from PyQt6.QtCore import QRect
            pixmap = self.grab(QRect(crop_x, crop_y, crop_w, crop_h))
            pixmap.save(snapshot_path)

            print(
                f"{trial:5d} | {self.car_x:6.1f} | {self.car_y:6.1f} | "
                f"{math.degrees(angle_diff):6.1f}° | {outcome:<10} | {final_score:6.1f}"
            )

        if scores:
            print("-" * 58)
            print(f"  avg={sum(scores)/len(scores):.1f}  min={min(scores):.1f}  max={max(scores):.1f}")
        self.show_lidar = prev_show_lidar
        self.show_front_wheels = prev_show_front_wheels
        self.show_player_center_dot = prev_show_player_center_dot
        print(f"Snapshots saved to '{os.path.abspath(out_dir)}/'.")

    def test_run(self):
        self.enable_training_spawns = True
        """Runs one random episode with visualization."""
        obs, _ = self.reset()
        for _ in range(500):
            action = self.action_space.sample()
            obs, reward, term, trunc, _ = self.step(action)
            if term or trunc: break
        print("Test Run Complete.")

    def test_collision(self):
        """
        Deterministically drives the car into a parked car 
        to verify that the SAT collision logic triggers correctly.
        """
        print("--- Starting Collision Test ---")
        self.reset()
        
        # Position the car in the right lane, slightly below the top parked car
        self.car_x = 475.0
        self.car_y = 350.0
        self.car_theta = -math.pi / 2  # Facing North
        
        # Action 2: Forward-Right (This creates an arc toward the top parked car at y=130)
        action = 0 
        
        for i in range(500):
            obs, reward, terminated, truncated, _ = self.step(action)
            
            # We want to see the collision happen
            if terminated:
                print(f"Collision Detected! Step: {i} | Pos: ({int(self.car_x)}, {int(self.car_y)})")
                return True
                
        print("Test Failed: Car did not detect a collision.")
        return False   
    def test_lidar(self):
        """
        Spins the car in a tight spot to verify that the 12 LIDAR rays 
        correctly detect and shorten when hitting obstacles.
        """
        print("--- Starting LIDAR Visual Test ---")
        self.reset()
        
        # Position the car near the parking spot and the curb
        self.car_x = 600.0
        self.car_y = 415.0
        
        # Slow rotation to watch the rays
        steps = 360
        dt = 0.016
        for i in range(steps):
            # Rotate by 1 degree per frame
            self.car_theta += math.radians(1)
            
            # Update traffic car position
            self.traffic_y -= 80.0 * dt
            if self.traffic_y < -150:
                self.traffic_y = 950.0
            
            if self.render_mode == "human":
                self.update()
                QApplication.processEvents()
                import time
                time.sleep(0.01)
                
            if i % 90 == 0:
                print(f"Testing Rotation... {int(i/360*100)}% complete")

        print("LIDAR Test Complete.")
        return True


    def test_coverage(self):
        """
        Slowly slides a TILTED car into the lane to verify 
        smooth partial coverage transitions.
        """
        print("--- Starting Tilted Coverage Test ---")
        self.reset()
        
        # 1. Set Initial Pose
        self.car_x = 450.0
        self.car_y = 415.0
        # Tilt the car by ~0.2 radians (approx 11 degrees)
        self.car_theta = -math.pi / 2 + 0.2 
        
        # 2. Slow Movement Loop
        # We increase the number of steps and decrease the increment for "slow motion"
        total_steps = 400
        for i in range(total_steps):
            self.car_x += 0.5  # Half-pixel increments

            # Traffic update
            self.traffic_y -= 80.0 * 0.016
            if self.traffic_y < -150:
                self.traffic_y = 950.0

            # 3. Calculate Coverage
            coverage = self.compute_coverage()
            
            # Print every 20 steps to observe the ramp-up
            if i % 20 == 0:
                print(f"Step: {i:03} | X: {self.car_x:.1f} | Coverage: {coverage*100:5.1f}%")
            
            # 4. Visualization
            if self.render_mode == "human":
                self.update()
                QApplication.processEvents()
                import time
                time.sleep(0.01) # Small delay for smooth viewing

            # Stop if we hit 100% or go too far
            if coverage >= 1.0 and i > 100:
                print(f"--- Full Coverage Reached at Step {i} ---")
                time.sleep(1.0)
                break
                
        return True

    def test_success(self):
        """
        Directly manipulates the car's position to test the success 
        condition boundaries without using steering physics.
        """
        print("--- Starting Success Condition Test ---")
        self.reset()
        
        # Start in the middle of the road, facing North
        self.car_x = 475.0
        self.car_y = 415.0
        self.car_theta = -math.pi / 2 
        
        # Slide laterally from X=475 to X=700
        for i in range(150):
            self.car_x += 1.5  # Move right 1.5 pixels per frame
            
            # Manually trigger the success check
            success = self._check_success()
            
            if self.render_mode == "human":
                self.update()
                QApplication.processEvents()
                # Tiny sleep so we can actually see the slide
                import time
                time.sleep(0.01)

            if success:
                print(f"Success Triggered! X: {self.car_x:.1f}, Y: {self.car_y:.1f}")
                # We stay in the loop for a few more frames to see it "parked"
                time.sleep(1)
                return True
                
        print("Test Failed: Car crossed the lane without triggering success.")
        return False

    def test_car_points_grid(self, duration_sec=3.0):
        """Rebuilds and visualizes the local sampling grid at the reference start pose."""
        print("--- Starting Car Points Grid Test ---")

        # Rebuild points to explicitly validate _build_car_points_grid output.
        self._build_car_points_grid()

        # Place the car at the reference pose.
        self.car_x = self.start_x
        self.car_y = self.start_y
        self.car_theta = self.target_theta
        self.car_delta = 0.0

        print(f"Reference pose: ({self.car_x:.1f}, {self.car_y:.1f}) | theta: {math.degrees(self.car_theta):.1f}°")
        print(f"Grid points: {self.total_points} ({self.grid_rows}x{self.grid_cols})")

        # Show local points on top of the player car.
        self.show_car_points_debug = True
        if self.render_mode == "human":
            self.update()
            QApplication.processEvents()
            import time
            time.sleep(duration_sec)
            self.show_car_points_debug = False
            self.update()
            QApplication.processEvents()
        else:
            print("Set render_mode='human' to visualize the grid overlay.")

        print("Car Points Grid Test Complete.")
        return True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    env = ParkingEnv(render_mode="human")
#    env.test_score_system()
#    env.test_car_points_grid(duration_sec=10.0)
#    env.test_collision()
#    env.test_run()
    env.test_lidar()
#    env.test_coverage()
#    env.test_success()
    sys.exit(app.exec())