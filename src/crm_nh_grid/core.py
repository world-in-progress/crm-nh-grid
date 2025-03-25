# Grid.py is the core resource model for a runtime memory resource that can be shared.
# Each grid has following attributes:
# - level: int (stored), the level of the grid
# - type: int (stored), the type of the grid, default to 0
# - subdivided: bool (stored), the subdivision status of the grid
# - deleted: bool (stored), the deletion status of the grid, default to False
# - elevation: float (stored), the elevation of the grid, default to -9999
# - global_id: int (stored), the global id within the bounding box that subdivided by grids all in the level of this grid
# - local_id: int (calclated), the local id within the parent grid that subdivided by child grids all in the level of this grid
# - tl_x: int (calculated), the top left x coordinate of the grid
# - tl_y: int (calculated), the top left y coordinate of the grid
# - br_x: int (calculated), the bottom right x coordinate of the grid
# - br_y: int (calculated), the bottom right y coordinate of the grid

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .interface.icrm import ICRM

class GridRecorder(ICRM):
    def __init__(self, epsg: int, bounds: list, first_size: list[float], subdivide_rules: list[list[int]]):
        super().__init__()
        self.epsg = epsg
        self.bounds = bounds # [min_x, min_y, max_x, max_y]
        self.first_size = first_size # [width, height] of the first level grid
        self.subdivide_rules = subdivide_rules # list of subdivision rules per level
        self.grid_table_name = "./grids.parquet"
        
        # Calculate level infos
        self.level_info: list[dict[str, int]] = [{
            'width': 1,
            'height': 1
        }]
        for level, rule in enumerate(subdivide_rules):
            sub_width: int = self.level_info[level - 1]['width'] * rule[0]
            sub_height: int = self.level_info[level - 1]['height'] * rule[1]
            self.level_info.append({
                'width': sub_width,
                'height': sub_height
            })
            
        # Try to find parquet file of grids
        # Init table if not exists
        if not os.path.exists(self.grid_table_name):
            data = {
                'type': [], 'level': [], 'elevation': [], 'global_id': [], 'deleted': [], 'subdivided': []
            }
        
            # Init all grids in all levels
            for level in range(len(self.level_info)):
                total_width = self.level_info[level]['width']
                total_height = self.level_info[level]['height']
                num_grids = total_width * total_height
                
                data['type'].extend(np.zeros(num_grids, dtype=np.int8))
                data['level'].extend(np.full(num_grids, level, dtype=np.int8))
                data['global_id'].extend(np.arange(num_grids, dtype=np.int32))
                data['deleted'].extend(np.full(num_grids, False, dtype=np.bool))
                data['elevation'].extend(np.full(num_grids, -9999.0, dtype=np.float32))
                data['subdivided'].extend(np.full(num_grids, True if level == 0 else False, dtype=np.bool))
            
            schema = pa.schema([
                ('type', pa.int8()), ('level', pa.int8()), ('deleted', pa.bool_()), 
                ('global_id', pa.int32()), ('subdivided', pa.bool_()), ('elevation', pa.float32())
            ])
            grid_table = pa.table([
                pa.array(data['type']), pa.array(data['level']), pa.array(data['deleted']), 
                pa.array(data['global_id']), pa.array(data['subdivided']), pa.array(data['elevation'])
            ], schema=schema)
            os.makedirs(os.path.dirname(self.grid_table_name), exist_ok=True)
            pq.write_table(grid_table, self.grid_table_name)
            
    def get_local_ids(self, level: int, global_ids: np.ndarray) -> np.ndarray:
        if level == 0:
            return global_ids
        total_width = self.level_info[level]['width']
        sub_width = self.subdivide_rules[level - 1][0]
        sub_height = self.subdivide_rules[level - 1][1]
        local_x = global_ids % total_width
        local_y = global_ids // total_width
        return (((local_y % sub_height) * sub_width) + (local_x % sub_width))
    
    def get_coordinates(self, level: int, global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_width = self.first_size[0] / self.level_info[level]['width']
        grid_height = self.first_size[1] / self.level_info[level]['height']
        total_width = self.level_info[level]['width']

        # Vectorized calculation of x and y indices
        x = global_ids % total_width
        y = global_ids // total_width

        # Vectorized coordinate computation
        tl_x = self.bounds[0] + x * grid_width
        tl_y = self.bounds[1] + y * grid_height
        br_x = tl_x + grid_width
        br_y = tl_y + grid_height

        return (tl_x, tl_y, br_x, br_y)
    
    def get_grid_infos(self, level: int, global_ids: np.ndarray) -> pd.DataFrame:
        
        # Filter table for the given level and global_ids
        filters = [('level', '=', level), ('global_id', 'in', global_ids)]
        table = pq.read_table(self.grid_table_name, filters=[filters], use_threads=True)
        rows = table.to_pandas(
            use_threads=True, integer_object_nulls=False
        )

        # Calculate computed attributes
        global_ids_array = rows['global_id'].to_numpy()
        local_ids = self.get_local_ids(level, global_ids_array)
        tl_x, tl_y, br_x, br_y = self.get_coordinates(level, global_ids_array)
        rows['local_id'] = local_ids
        rows['tl_x'] = tl_x
        rows['tl_y'] = tl_y
        rows['br_x'] = br_x
        rows['br_y'] = br_y
        
        column_order = ['level', 'global_id', 'local_id', 'type', 'elevation', 
                        'deleted', 'subdivided', 'tl_x', 'tl_y', 'br_x', 'br_y']
        return rows[column_order]
    