import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import math
import gc
import collections
from PIL import Image
from torchvision import transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''6G environment settings'''
BW = 5*1e9      #5GHz
Ptx = 100       #dBW
fc = 600*1e6    #600MHz
d = 500*1e3     #500km
Gtx = 10        #dBi
Grx = 10        #dBi
Labs = 74       #dbm
No = -160       #dBW
c_bit = 1      #num of bits coded
num = 100       #num of files

def to_dB(value):
    return 10* math.log10(value)

def Air_Latency(bit):
    L= Labs + 92.45 + math.log10(fc*d)
    Prx= Ptx-L+Gtx+Grx
    air_latency= (num*bit*c_bit) / (BW*(1+Prx-(No+to_dB(BW))))
    return air_latency #sec

def print_sec(sec):
  millisec = sec * 1000
  print("{:.3f}ms".format(millisec))

