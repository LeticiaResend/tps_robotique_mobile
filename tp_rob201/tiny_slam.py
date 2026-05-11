""" A simple robotics navigation code including SLAM, exploration, planning"""

import numpy as np
from occupancy_grid import OccupancyGrid
import math


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
 
        # Step 1: keep only valid hits (below max range)
        valid = distances < lidar.max_range
        distances = distances[valid]
        angles = angles[valid]
 
        # Step 2: hit positions in world frame
        x = pose[0] + distances * np.cos(angles + pose[2])
        y = pose[1] + distances * np.sin(angles + pose[2])
 
        # Step 3: convert to map indices, filter out-of-bounds
        x_idx, y_idx = self.grid.conv_world_to_map(x, y)
        in_bounds = (
            (x_idx >= 0) & (x_idx < self.grid.x_max_map) &
            (y_idx >= 0) & (y_idx < self.grid.y_max_map)
        )
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
 
        # Step 4: sum occupancy values
        return np.sum(self.grid.occupancy_map[x_idx, y_idx])

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
 
        x_ref, y_ref, theta_ref = odom_pose_ref
        x0, y0, theta0 = odom_pose
 
        # Distance and angle of the robot in the odom frame
        d = math.sqrt(x0**2 + y0**2)
        alpha = math.atan2(y0, x0)
 
        x = x_ref + d * math.cos(theta_ref + alpha)
        y = y_ref + d * math.sin(theta_ref + alpha)
        theta = theta_ref + theta0
 
        return [x, y, theta]

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        N = 100                              # number of random trials
        sigma = np.array([0.05, 0.05, 0.001])   # [x_std, y_std, theta_std]
 
        current_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref)
        best_score = self._score(lidar, current_pose)
        best_ref = np.array(self.odom_pose_ref)
 
        for _ in range(N):
            candidate_ref = best_ref + np.random.normal(0, sigma)
            candidate_pose = self.get_corrected_pose(raw_odom_pose, candidate_ref)
            score = self._score(lidar, candidate_pose)
 
            if score > best_score:
                best_score = score
                best_ref = candidate_ref
 
        self.odom_pose_ref = best_ref
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        x_r, y_r, theta_r = pose

        distances = np.asarray(lidar.get_sensor_values(), dtype=float)
        angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        # Log-odds increments: weak evidence for free space, stronger for obstacle.
        free_update = -0.15
        occ_update = 3
        map_min = -20.0
        map_max = 20.0

        # If available, use lidar max range to avoid adding false obstacles at range limit.
        lidar_max_range = getattr(lidar, "max_range", np.inf)

        for distance, alpha in zip(distances, angles):
            if not np.isfinite(distance) or distance <= 0:
                continue

            ray_angle = theta_r + alpha

            # Free cells along the beam (stop one cell before measured obstacle when possible).
            free_distance = max(0.0, distance - self.grid.resolution)
            x_free = x_r + free_distance * np.cos(ray_angle)
            y_free = y_r + free_distance * np.sin(ray_angle)
            self.grid.add_value_along_line(x_r, y_r, x_free, y_free, free_update)

            # Occupied cell at the end point only when a real hit is likely observed.
            if distance < 0.99 * lidar_max_range:
                x_hit = x_r + distance * np.cos(ray_angle)
                y_hit = y_r + distance * np.sin(ray_angle)
                self.grid.add_map_points(np.array([x_hit]), np.array([y_hit]), occ_update)

        # Saturation avoids numerical divergence after many updates.
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, map_min, map_max)
