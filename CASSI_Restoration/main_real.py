
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import time
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
from optimization_real import ADMM_Iter
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)
#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', default = 50000,            help="Maximum number of iterations")
parser.add_argument('--DP_iter',  default = 1,           help="Training epochs of deep Low-rank networks")
parser.add_argument('--R_iter',   default = 1,            help="Reduced Training epochs of deep Low-rank networks")
parser.add_argument('--scene',    default = 'scene1',      help="Scene01-10")

args = parser.parse_args()

#----------------------- Data Configuration -----------------------#
h, w, nC, step = 660, 660, 28, 2
dataset_dir = '../Data/TSA_real_data/TSA_reconstruction/'
data_name = args.scene

results_dir = './Results/' + 'real'+ data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
matfile = dataset_dir  + 'Recon_'+ data_name + '.mat'
a = sio.loadmat(matfile)
data_truth = torch.from_numpy(sio.loadmat(matfile)['recon']).to(device)
data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC)).to(device)
for i in range(nC):
    data_truth_shift[:, i*step:i*step+w, i] = data_truth[:, :, i]

mask = torch.zeros((h, w + step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy(sio.loadmat('../Data/TSA_real_data/mask.mat')['mask'])
for i in range(nC):
    mask_3d[:, i*step:i*step+w, i] = mask_256
Phi = mask_3d.to(device)
meas00 = torch.sum(Phi*data_truth_shift,dim=2)


meas = sio.loadmat('../Data/TSA_real_data/Measurements/'+data_name+'.mat')['meas_real']
meas = torch.FloatTensor(meas.copy()).cuda()
meas = meas/meas.max()*8
data_truth= data_truth/data_truth.max()
#-------------------------- Optimization --------------------------#
recon = ADMM_Iter(meas, Phi, data_truth, args, results_dir)

