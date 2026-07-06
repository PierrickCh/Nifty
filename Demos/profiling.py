import torch
from multiprocessing import cpu_count
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from time import perf_counter
# Add parent directory to path for imports and file paths to work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

import Nifty.method as _nifty_method
import Nifty.method_reworked as _nifty_method_reworked
from Nifty.method import *

cpu_nb = cpu_count()
_nifty_method.device = manually_select_device(try_gpu=False)
_nifty_method_reworked.device = manually_select_device(try_gpu=False)
# img = Tensor_load('results/red_peppers.jpg')
# t0 = perf_counter()
# torch.set_num_threads(16)
# synth= Nifty(img,rs=1,T=50,k=5,patchsize=16,stride=4,octaves=4,size=(512,512),renoise=0.3,warmup=0,memory=False,show=False)
# print(perf_counter()-t0)

y1 = []
y2 = []
x = []

torch.manual_seed(0)
img = Tensor_load('results/red_peppers.jpg')
for i in range(6):
    x.append(i)
    torch.set_num_threads(cpu_count())
    
    
    t0 = perf_counter()
    synth= _nifty_method_reworked.Nifty(img,rs=1,T=50,k=5,patchsize=16,stride=4,octaves=4,size=(512,512),renoise_time=0.9,warmup=0,memory=False,show=False)
    y1.append(perf_counter()-t0)
    print(y1[-1])
    
    t0 = perf_counter()
    synth= _nifty_method_reworked.Nifty_improved_2(img,rs=1,T=50,k=5,patchsize=16,stride=4,octaves=4,size=(512,512),renoise_time=0.9,warmup=0,memory=False,show=False)
    y2.append(perf_counter()-t0)
    print(y2[-1])

plt.plot(x,y1, label="original")
plt.plot(x,y2, label="improved")
plt.legend()
plt.xlabel("CPU Cores")
plt.ylabel("Computation Time (s)")
plt.title("CPU Count / Computation Time")
plt.show()


