import numpy as np
from crm_nh_grid import CRM
    
redis_host: str = 'localhost'
redis_port: int = 6379
persistence_path : str = './'
epsg: int = 2326
first_size: list[float] = [64, 64]
bounds: list[float] = [808357.5, 824117.5, 838949.5, 843957.5]
subdivide_rules: list[list[int]] = [
    #    64x64,  32x32,  16x16,    8x8,    4x4,    2x2,    1x1
    [478, 310], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1, 1]
]

if __name__ == '__main__':

    CRM.initialize(redis_host, redis_port, epsg, bounds, first_size, subdivide_rules)
    
    gr = CRM(redis_host, redis_port)
    
    parent = gr.get_grid_info(1, 0)
    print('=== Parent ===')
    print(parent, '\n')
    
    child_keys = gr.subdivide_grids(np.array([1]), np.array([0]))
    for key in child_keys:
        print(key, '\n')
        level, global_id = map(int, key.split('-'))
        child = gr.get_grid_info(level, global_id)
        print(child)
        
    parent = gr.get_grid_info(1, 0)
    print('=== Parent ===')
    print(parent) 
    