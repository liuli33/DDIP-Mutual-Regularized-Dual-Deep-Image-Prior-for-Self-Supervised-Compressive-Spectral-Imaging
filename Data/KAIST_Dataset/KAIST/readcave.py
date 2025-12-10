
#coding:utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import time
import torch
import argparse
import numpy as np

from numpy import *

import scipy.io as sio
import matplotlib.pyplot as plt

#from torchmetrics import SpectralAngleMapper#
import torch.nn.functional as F





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)


#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ADMM', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV', help="Select which denoiser: Total Variation (TV) or Deep denoiser (HSICNN)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=320, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=6, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=10, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[130, 100, 80, 70, 60, 90], help="The noise levels")
args = parser.parse_args()
#------------------------------------------------------------------#


#----------------------- Data Configuration -----------------------#
dataset_dir = './'
results_dir = 'results'
for i in range (10) :
    data_name = i+1
    matfile = 'scene0'+str(data_name) + '.mat'
    h, w, nC, step = 256, 256, 28, 2
    a = sio.loadmat(matfile)
    data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
    data_truth = data_truth/torch.max(data_truth)
    data_truth_shift = torch.zeros((h, w, nC))
    ref_img_1 = torch.unsqueeze(data_truth[:, :, 25], 2)
    ref_img_2 = torch.unsqueeze(data_truth[:, :, 15], 2)
    ref_img_3 = torch.unsqueeze(data_truth[:, :, 5], 2)
    ref_img = torch.cat((ref_img_1, ref_img_2, ref_img_3), dim=2)

    print(matfile)
    #-----------------------------------------------------------------------------------------
    rgb_wei = ref_img
    plt.figure()
    plt.imshow(rgb_wei)
    plt.savefig('RGB_gray/'+str(data_name)+'.png')
    #------------------------------------------------------------------#
