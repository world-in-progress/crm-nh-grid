import pytest
import numpy as np
import pandas as pd
from crm_nh_grid.core import GridRecorder

epsg: int = 2326
first_size: list[float] = [64, 64]
bounds: list[float] = [808357.5, 824117.5, 838949.5, 843957.5]
subdivide_rules: list[list[int]] = [
    #    64x64,  32x32,  16x16,    8x8,    4x4,    2x2,    1x1
    [478, 310], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1, 1]
]

def test_grid_recorder_init():
    gr = GridRecorder(epsg, bounds, first_size, subdivide_rules)

    assert gr.epsg == epsg
    assert gr.bounds == bounds
    assert gr.first_size == first_size
    assert len(gr.level_info) == 8
    assert gr.level_info[0] == {'width': 1, 'height': 1}
    assert gr.level_info[1] == {'width': 478, 'height': 310}

def test_get_grid_info():
    gr = GridRecorder(epsg, bounds, first_size, subdivide_rules)

    global_ids = np.array([0, 1, 478])
    df = gr.get_grid_infos(1, global_ids)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    
    expected_columns = ['level', 'global_id', 'local_id', 'type', 'elevation', 
                        'deleted', 'subdivided', 'tl_x', 'tl_y', 'br_x', 'br_y']
    assert list(df.columns) == expected_columns
    
    assert np.array_equal(df['global_id'].values, global_ids)
    assert df['level'].iloc[0] == 1
    assert df['tl_x'].iloc[0] == bounds[0]
    assert df['elevation'].iloc[0] == -9999.0

def test_get_coordinates():
    gr = GridRecorder(epsg, bounds, first_size, subdivide_rules)
    global_ids = np.array([0, 1, 477])
    tl_x, tl_y, br_x, br_y = gr.get_coordinates(1, global_ids)

    grid_width = first_size[0] / 478
    assert tl_x[0] == bounds[0]
    assert tl_x[1] == bounds[0] + grid_width
    assert br_y[0] == bounds[1] + first_size[1] / 310
    
@pytest.mark.parametrize("size", [10, 100, 1000, 10000, 100000])
def test_benchmark_get_grid_info(benchmark, size):
    gr = GridRecorder(epsg, bounds, first_size, subdivide_rules)
    test_level = 3
    
    global_ids = np.random.randint(
        0, gr.level_info[test_level]['width'] * gr.level_info[test_level]['height'],
        size=size, dtype=np.int32
    )
    
    benchmark(lambda: gr.get_grid_infos(test_level, global_ids))
    