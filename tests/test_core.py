import os
import pytest
import numpy as np
import pandas as pd
from crm_nh_grid.crm import CRM

# mmp_path: str = '/dev/shm'
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

CRM.initialize(redis_host, redis_port, epsg, bounds, first_size, subdivide_rules)

def test_grid_recorder_init():
    gr = CRM(redis_host, redis_port)

    assert gr.epsg == epsg
    assert gr.bounds == bounds
    assert gr.first_size == first_size
    assert len(gr.level_info) == 8
    assert gr.level_info[0] == {'width': 1, 'height': 1}
    assert gr.level_info[1] == {'width': 478, 'height': 310}

def test_get_grid_info():
    gr = CRM(redis_host, redis_port)

    global_ids = np.array([0, 1, 478])
    df = gr.get_grid_infos(1, global_ids)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    
    expected_columns = ['level', 'global_id', 'local_id', 'type', 'elevation', 
                        'deleted', 'shown', 'min_x', 'min_y', 'max_x', 'max_y']
    assert list(df.columns) == expected_columns
    
    # assert np.array_equal(df['global_id'].values, global_ids)
    # assert df['level'].iloc[0] == 1
    # assert df['min_x'].iloc[0] == bounds[0]
    # assert df['elevation'].iloc[0] == -9999.0

def test_get_coordinates():
    gr = CRM(redis_host, redis_port)
    global_ids = np.array([0, 1, 477])
    min_xs, min_ys, max_xs, max_ys = gr.get_coordinates(1, global_ids)

    grid_width = (bounds[2] - bounds[0]) / 478
    grid_height = (bounds[3] - bounds[1]) / 310
    assert min_xs[0] == bounds[0]
    assert max_xs[0] == bounds[0] + grid_width
    assert min_ys[0] == bounds[1]
    assert max_ys[0] == bounds[1] + grid_height
    
@pytest.mark.parametrize("size", [10, 100, 1000, 10000, 100000])
def test_benchmark_get_grid_info(benchmark, size):
    gr = CRM(redis_host, redis_port)
    test_level = 3
    
    global_ids = np.random.randint(
        0, gr.level_info[test_level]['width'] * gr.level_info[test_level]['height'],
        size=size, dtype=np.int32
    )
    
    benchmark(lambda: gr.get_grid_infos(test_level, global_ids))
    