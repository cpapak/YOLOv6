import argparse
from logging import Logger
import os
import yaml
import os.path as osp
from pathlib import Path
import torch
import torch.distributed as dist
import sys
import datetime
import numpy as np

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import Model
from yolov6.utils.config import *

conf_file = "./configs/yolov6l.py"
cfg = Config.fromfile(conf_file)
if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')

model = Model(cfg, num_classes=80)
device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,640,640, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("{}".format(cfg.filename))
print(1000.0/mean_syn)