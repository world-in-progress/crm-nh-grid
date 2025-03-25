import numpy as np
import pandas as pd

class INode:
    def __init__(self, level: int, global_id: int, local_id: int, type: int, elevation: float, deleted: bool, subdivided: bool, tl_x: float, tl_y: float, br_x: float, br_y: float):
        
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.br_x = br_x
        self.br_y = br_y
        self.type = type
        self.level = level
        self.deleted = deleted
        self.local_id = local_id
        self.global_id = global_id
        self.elevation = elevation
        self.subdivided = subdivided

class ICRM:
    # Method to calculate local_id
    def get_local_ids(self, level: int, global_ids: np.ndarray) -> np.ndarray:
        pass
    
    # Method to calculate coordinates for a given level and global_ids
    # Return tuple (tl_x, tl_y, br_x, br_y)
    def get_coordinates(self, level: int, global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    # Method to get grid info for multiple grids in a given level
    # Column order of returned dataFrame is as follows: 
    # 'level', 'global_id', 'local_id', 'type', 'elevation', 'deleted', 'subdivided', 'tl_x', 'tl_y', 'br_x', 'br_y'
    def get_grid_infos(self, level: int, global_ids: np.ndarray) -> pd.DataFrame:
        pass