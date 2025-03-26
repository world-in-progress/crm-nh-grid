import redis
import numpy as np
import pandas as pd

class ICRM:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with open data organized by `crm-nh-grid<https://github.com/world-in-progress/crm-nh-grid>`_ and published by `Noodle <https://github.com/world-in-progress/noodle>`_. 

    Attributes of Grid
    ---
    - level (int8): the level of the grid
    - type (int8): the type of the grid, default to 0
    - subdivided (bool), the subdivision status of the grid
    - deleted (bool): the deletion status of the grid, default to False
    - elevation (float64): the elevation of the grid, default to -9999.0
    - global_id (int32): the global id within the bounding box that subdivided by grids all in the level of this grid
    - local_id (int32): the local id within the parent grid that subdivided by child grids all in the level of this grid
    - min_x (float64): the min x coordinate of the grid
    - min_y (float64): the min y coordinate of the grid
    - max_x (float64): the max x coordinate of the grid
    - max_y (float64): the max y coordinate of the grid
    """
    
    def __init__(self, redis_host: str, redis_port: int):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        
    @staticmethod
    def initialize(redis_host: str, redis_port: str, epsg: int, bounds: list, first_size: list[float], subdivide_rules: list[list[int]]):
        """Method to initialize CRM

        Args:
            redis_host (str): host name of redis service
            redis_port (str): port of redis service
            epsg (int): epsg code of the grid
            bounds (list): bounding box of the grid (organized as [min_x, min_y, max_x, max_y])
            first_size (list[float]): [width, height] of the first level grid
            subdivide_rules (list[list[int]]): list of subdivision rules per level
        """
        pass
    
    def get_local_ids(self, level: int, global_ids: np.ndarray) -> np.ndarray:
        """Method to calculate local_ids for provided grids having same level
        
        Args:
            level (int): level of provided grids
            global_ids (np.ndarray): global_ids of provided grids
        
        Returns:
            local_ids (np.ndarray): local_ids of provided grids
        """
        pass
    
    def get_coordinates(self, level: int, global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Method to calculate coordinates for provided grids having same level
        
        Args:
            level (int): level of provided grids
            global_ids (np.ndarray): global_ids of provided grids

        Returns:
            coordinates (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): coordinates of provided grids, orgnized by tuple of (min_xs, min_ys, max_xs, max_ys)
        """
        pass
    
    def get_grid_infos(self, level: int, global_ids: np.ndarray) -> pd.DataFrame:
        """Method to get all attributes for provided grids having same level

        Args:
            level (int): level of provided grids
            global_ids (np.ndarray): global_ids of provided grids

        Returns:
            grid_infos (pd.DataFrame): grid infos orgnized by dataFrame, the order of which is: level, global_id, local_id, type, elevation, deleted, show, tl_x, tl_y, br_x, br_y
        """
        pass
