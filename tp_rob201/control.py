""" A set of robotics control functions """

import random
from turtle import distance
import numpy as np

last_turn_direction = None
stored_poses = []
wall_following = False
wall_following_direction = None



def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()

    front_mask = np.abs(ray_angles) < np.deg2rad(45)

    front_distances = laser_dist[front_mask]

    min_front_distance = np.min(front_distances)

    obstacle_threshold = 25.0  

    if min_front_distance > obstacle_threshold:
        speed = 0.7
        rotation_speed = 0.0
    else:
        speed = 0.0
        rotation_speed = 0.5

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command

def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    """
    current_x     = current_pose[0]
    current_y     = current_pose[1]
    current_theta = current_pose[2]

    goal_x = goal_pose[0]
    goal_y = goal_pose[1]
 
    K_goal = 1.0
    
    # Vector from current position to goal
    dx = goal_x - current_x
    dy = goal_y - current_y
    # Distance Euclidienne to the goal
    distance_to_goal = np.sqrt(dx**2 + dy**2)
    # If we are close enough to the goal, stop
    if distance_to_goal < 10.0:
        return {"forward": 0.0, "rotation": 0.0}
    # Attractive potential gradient
    grad_att_x = (K_goal / distance_to_goal) * dx
    grad_att_y = (K_goal / distance_to_goal) * dy
   
 
    K_obs  = 300.0
    d_safe = 100.0
    d_rep_disable = 30.0   # distance goal en dessous de laquelle on désactive
 
    grad_rep_x = 0.0
    grad_rep_y = 0.0
 
    if distance_to_goal > d_rep_disable:
        laser_dist   = lidar.get_sensor_values()
        laser_angles = lidar.get_ray_angles()
 
        for dist, angle in zip(laser_dist, laser_angles):
            if dist < d_safe:
                obs_x = np.cos(angle) * dist
                obs_y = np.sin(angle) * dist
                coeff  = K_obs * ((1.0 / dist) - (1.0 / d_safe)) / (dist**3)
                grad_rep_x +=  coeff * obs_x
                grad_rep_y +=  coeff * obs_y
 
    grad_x = grad_att_x + grad_rep_x
    grad_y = grad_att_y + grad_rep_y
 
    desired_angle = np.arctan2(grad_y, grad_x)
 
    angle_error = desired_angle - current_theta
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
 
    rotation_speed = 1.0 * angle_error
 
    forward_speed = max(0.0, np.cos(angle_error))
    forward_speed *= min(distance_to_goal / 100.0, 1.0)
 
    forward_speed  = float(np.clip(forward_speed,  -0.3, 0.3))
    rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
 
    return {"forward": forward_speed, "rotation": rotation_speed}