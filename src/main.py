# import sys
import numpy as np
import torch
from array import array
import sys

from srnn.mlnn.trainer import Trainer
from svsr.spl.train import run_spl
from svsr.pesr.solver import run_pesr
from mksr.solver import MKSR
from config.test0422 import TestSettings
from config import dynamic

if __name__ == "__main__":
    # step 0 : load the config
    try:
        func_name = sys.argv[1]
        cfg = TestSettings[func_name]
    except:
        print("please specify test name.\n"
              "e.g. python3 main.py test2\n"
              "e.g. make run test=test2\n")
        exit(0)

    # step 1 : generate data
    x_num = cfg['common']['x_num']
    y_num = cfg['common'].get('y_num', 1)
    x_range = cfg['common']['x_range']
    np_rng = np.random.default_rng(seed=2)
    if func_name == "toggle":
        data_x, data_y = dynamic.generate_dynamcis_data(
            'Toggle',
            dynamic_kwargs={'x_range': list(x_range.values()), 'params': cfg['params'], 'dt': cfg['dt']},
            traj_num=cfg['traj_num'],
            traj_points=cfg['traj_points'],
        )
        data_x = data_x.numpy()
        data_y = data_y.numpy()
        # print(data_x.shape)
        # print(data_y.shape)
    else:
        data_num = cfg['data_num']
        target_func = cfg['target_func']
        data_x = np.ndarray((data_num, x_num), dtype=np.float32)
        for vid in range(x_num):
            data_x[:, vid] = np_rng.uniform(*x_range[f"x{vid}"], data_num)
        data_y = target_func(data_x)

    # print(f"data_y : {data_y}")

    # step 2 : train the neuro-network and get a eval function
    trainer = Trainer(
        func_name=func_name,
        data_x=data_x,
        data_y=data_y,
        **cfg['common'],
        **cfg['srnn_config'])
    trainer.run()

    eqs = {}
    for y_id in range(y_num):
        neuro_eval = trainer.get_eval(y_id)
        svsr_method = run_spl
        

        # def neuro_eval(x):
        #     def target_func(x):
        #         return 4 / (1 + x[:, 1] ** 3) - x[:, 0]
        #     y = target_func(x.T)
        #     return np.array(y.flat)

        # step 3 : run mksr method with underlying method run_spl
        mksr_model = MKSR(
            func_name=f"{func_name}/mksr_y{y_id}",
            random_seed=2,
            neuro_eval=neuro_eval,
            svsr_method=svsr_method,
            svsr_cfg=cfg['svsr_config'],
            **cfg['common'],
            **cfg['mvsr_config'])
        mksr_model.run()

        eqs[f'y{y_id}'] = str(mksr_model)
print("="*50)
for y, eq in eqs.items():
    print(f"discovered for {y}: {eq}")