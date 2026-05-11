"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.localise_counter = 0

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp4()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        return reactive_obst_avoid(self.lidar())

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-450, -100, 0] # [-760, -30, 0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command
    
    def control_tp3(self):
        """
        Build the occupancy map using raw odometry pose.
        No localisation correction — the map will drift, which is expected
        """

        raw_pose   = self.odometer_values()
        lidar_data = self.lidar()
 
        # Update map directly with raw odometry (no correction)
        self.tiny_slam.update_map(lidar_data, raw_pose)
        
        self.occupancy_grid.display_cv(raw_pose)
 
        return self.control_tp2()
        

    def control_tp4(self):
        """
        Full SLAM loop: localise first, then update the map with the
        corrected pose. The map should no longer drift.

        """
        raw_pose   = self.odometer_values()
        lidar_data = self.lidar()
 
        if self.counter > 20:
            self.tiny_slam.localise(lidar_data, raw_pose)
            
        # --- Corrected pose: odom pose expressed in world frame ---
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_pose)
 
        # --- Map update with corrected pose ---
        self.tiny_slam.update_map(lidar_data, self.corrected_pose)
 
        self.occupancy_grid.display_cv(self.corrected_pose)

        self.counter += 1
 
        return self.control_tp2()
