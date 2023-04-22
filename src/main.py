# import sys
import numpy as np
import torch
from array import array
import sys

from srnn.mlnn.trainer import Trainer
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
        func_name=func_name,
        random_seed=2,
        **cfg)
    trainer.run()
    neuro_eval = trainer.get_eval()
    
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
        random_seed=2,
        **cfg)
    mksr_model.run()
    print(f"discovered euqation: {str(mksr_model)}")