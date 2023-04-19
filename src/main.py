# import sys
import numpy as np
import torch
from array import array
import sys

from srnn.utils import Trainer
from svsr.spl.train import run_spl
from svsr.pesr.solver import run_pesr
from mksr.solver import MKSR
from config.testconfig import TestSettings

if __name__ == "__main__":
    try:
        func_name = sys.argv[1]
        cfg = TestSettings[func_name]
    except:
        print("please specify test name.\n"
              "e.g. python3 main.py test2\n"
              "e.g. make run test=test2\n")
        exit(0)
    trainer = Trainer(
        cfg=cfg,
        model_mode='MLP',
        func_name=func_name,
        batch_size=128,
        lr=0.0001,
        seed=0,
        cuda=0,
    )
    try:
        trainer.load()
    except:
        trainer.fit(cfg['mlp_fit_times'])
        trainer.load()
        trainer.model.cpu()
    if __debug__:
        diff = array('d')
    def neuro_eval(x: np.ndarray):              
        """assume the NN can eval for any x
        """
        if func_name=="test-div":
            x = x.T
            ground_truth_result = cfg['func'](x).reshape((len(x), 1))
            x = x.T
            return ground_truth_result
        torch_x = torch.tensor(x.T, dtype=torch.float32)
        nn_result = trainer.model(torch_x).detach().numpy()
        if __debug__:
            global diff
            x = x.T
            ground_truth_result = cfg['func'](x).reshape((len(x), 1))
            x = x.T
            diff.extend((nn_result - ground_truth_result).flat)
        return nn_result.reshape(len(nn_result))
    if cfg['svsr_config']['method'] == 'spl':
        svsr_method = run_spl
    elif cfg['svsr_config']['method'] == 'pesr':
        svsr_method = run_pesr
    else:
        raise Exception("No '{cfg['svsr_config']['method']}' svsr method ")
    mksr_model = MKSR(
        func_name=func_name,
        neuro_eval=neuro_eval,
        svsr_method=svsr_method,
        random_seed=10,
        **cfg)
    mksr_model.run()
    print(f"discovered euqation: {mksr_model}")
    if __debug__:
        np.save(f"results/{func_name}/eval_error.npy", np.array(diff))