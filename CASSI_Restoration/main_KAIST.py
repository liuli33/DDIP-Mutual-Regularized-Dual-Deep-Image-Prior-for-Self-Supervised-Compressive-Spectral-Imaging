import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
from optimziation_HQS_acce import ADMM_Iter
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)
print(1111111111111)
# -----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
# with out x_loss 11000
parser.add_argument('--iter_num', default=30000, help="Maximum number of iterations")
parser.add_argument('--DP_iter', default=1, help="Training epochs of deep Low-rank networks")
parser.add_argument('--R_iter', default=1, help="Reduced Training epochs of deep Low-rank networks")
parser.add_argument('--scene', default='scene10', help="Scene01-10")
args = parser.parse_args()

# ----------------------- Data Configuration -----------------------#
h, w, nC, step = 256, 256, 28, 2
psnr_list = []
ssim_list = []
for scene in range(10):

    dataset_dir = '../Data/KAIST_Dataset/KAIST/'
    if scene+1==10:
        data_name = 'scene' + str(scene + 1)
    else:
        data_name = 'scene0' +str(scene+1)
    print(data_name)
    results_dir = './Results_with_outatt/' + data_name + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    matfile = dataset_dir + '/' + data_name + '.mat'
    data_truth = torch.from_numpy(sio.loadmat(matfile)['img']).to(device)
    data_truth_shift = torch.zeros((h, w + step * (nC - 1), nC)).to(device)
    for i in range(nC):
        data_truth_shift[:, i * step:i * step + w, i] = data_truth[:, :, i]


    matfile = dataset_dir + '/' + data_name + '.mat'
    data_truth = torch.from_numpy(sio.loadmat(matfile)['img']).to(device)
    data_truth_shift = torch.zeros((h, w + step * (nC - 1), nC)).to(device)
    for i in range(nC):
        data_truth_shift[:, i * step:i * step + w, i] = data_truth[:, :, i]

    mask = torch.zeros((h, w + step * (nC - 1)))
    mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
    mask_256 = torch.from_numpy(sio.loadmat('../Data/KAIST_Dataset/mask.mat')['mask'])
    for i in range(nC):
        mask_3d[:, i * step:i * step + w, i] = mask_256
    Phi = mask_3d.to(device)

    meas = torch.sum(Phi * data_truth_shift, 2).to(device)
    # -------------------------- Optimization --------------------------#
    PSNR, SSIM = ADMM_Iter(meas, Phi, data_truth, args, results_dir)
    psnr_list.append(PSNR)
    ssim_list.append(SSIM)
    print(psnr_list)
    print(ssim_list)
psnr_mean = np.mean(psnr_list)
ssim_mean = np.mean(ssim_list)
print(psnr_mean)
print(ssim_mean)
print('mean_psnr is', psnr_mean, 'mean_psnr is', ssim_mean)
