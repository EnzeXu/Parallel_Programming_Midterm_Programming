from config.alltests import TestSettings
import numpy as np
import pandas as pd

np_rng = np.random.default_rng(seed=20)
data_folder = 'data/'
    
for funcname, cfg in TestSettings.items():
    print(funcname)
    x_num = cfg['common']['x_num']
    y_num = cfg['common'].get('y_num', 1)
    x_range = cfg['common']['x_range']
    data_num = cfg['data_num']
    target_func = cfg['target_func']
    data_x = np.ndarray((data_num, x_num), dtype=np.float32)
    for vid in range(x_num):
        data_x[:, vid] = np_rng.uniform(*x_range[f"x{vid}"], data_num)
    data_y = target_func(data_x)
    print("data_num:", data_num)
    print(data_x[:3, ...])
    print(data_y[:3, ...])
    np.save(f"data/{funcname}_x.npy", data_x)
    np.save(f"data/{funcname}_y.npy", data_y)
    
    train_num = int(data_num * 0.8)
    train_sample = np.hstack([data_x, data_y.reshape(len(data_y), 1)])[:train_num, ...]
    pd.DataFrame(train_sample).to_csv('data/' + funcname + '_train.csv', index=False, header=False)
    
