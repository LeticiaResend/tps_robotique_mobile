"""
Planner class
Implementation of A*
"""

import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.occupancy_grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        """
        Returns safe 8-neighbors for A*
        """

        neighbors = []

        x, y = current_cell

        grid_data = self.occupancy_grid.occupancy_map

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:

                # skip current cell
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                # map bounds
                if not (0 <= nx < grid_data.shape[0] and 0 <= ny < grid_data.shape[1]):
                    continue

                # safety margin around obstacles
                safe = True

                for sx in range(-9, 10):
                    for sy in range(-9, 10):

                        check_x = nx + sx
                        check_y = ny + sy

                        if (0 <= check_x < grid_data.shape[0] and
                            0 <= check_y < grid_data.shape[1]):

                            # occupied cell detected nearby
                            if grid_data[check_x, check_y] > 0:
                                safe = False
                                break

                    if not safe:
                        break

                if safe:
                    neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, cell_1, cell_2):

        return np.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)

    def plan(self, start, goal):
      
        start_node = self.occupancy_grid.conv_world_to_map(start[0], start[1])
        goal_node = self.occupancy_grid.conv_world_to_map(goal[0], goal[1])

        # openSet := {start}
        openSet = [start_node]
        cameFrom = {}
        
        # Costs maps
        gScore = {start_node: 0}
        fScore = {start_node: self.heuristic(start_node, goal_node)}

        while openSet:
            # current := node in openSet with lowest fScore
            current = min(openSet, key=lambda node: fScore.get(node, float('inf')))

            if current == goal_node:
                return self.reconstruct_path(cameFrom, current)

            openSet.remove(current)

            for neighbor in self.get_neighbors(current):
                # distance d(current, neighbor)
                d = self.heuristic(current, neighbor)
                tentative_gScore = gScore[current] + d

                if tentative_gScore < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + self.heuristic(neighbor, goal_node)
                    if neighbor not in openSet:
                        openSet.append(neighbor)
        print("No path found")
        return None

    def reconstruct_path(self, cameFrom, current):
      
        total_path = [current]
        while current in cameFrom:
            current = cameFrom[current]
            total_path.insert(0, current) # prepend
        
        # Converte para coordenadas de mundo para o robô usar
        world_path = []
        for cell in total_path:
            xw, yw = self.occupancy_grid.conv_map_to_world(cell[0], cell[1])
            world_path.append([xw, yw])
        return world_path