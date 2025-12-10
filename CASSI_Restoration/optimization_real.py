import torch
from func import *
from os.path import exists
from model.skip3 import *
from model.model_loader import *
import os
import scipy.io as sio
from model.utils import MSIQA
import warnings
import random
import time
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def seed_torch(seed=5):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def show_rgb(data_truth, item, num):
    ref_img_1 = torch.unsqueeze(data_truth[:, :, 25], 2)
    ref_img_2 = torch.unsqueeze(data_truth[:, :, 15], 2)
    ref_img_3 = torch.unsqueeze(data_truth[:, :, 5], 2)
    ref_img = torch.cat((ref_img_1, ref_img_2, ref_img_3), dim=2)

    rgb_wei = ref_img
    plt.figure()
    plt.imshow(rgb_wei)
    plt.savefig('RGB/'+str(item)+str(num)+'.png')

def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros((bs, nC, row, col + (nC - 1) * step)).cuda()
    for i in range(nC):
        output[:, i, :, i * step:i * step + col] = inputs[:, i, :, :]
    return output

def getresponse(data_batch, mask3d_batch, step=2):
    x = shift_3d(data_batch, step).squeeze(0)
    temp = x*mask3d_batch
    meas = torch.sum(temp.permute(1, 2, 0), 2)

    return meas

def ADMM_Iter(meas, Phi, data_truth, args, result_dir):
    seed_torch()
    Phi = Phi.to(device)
    # -------------- Initialization --------------#
    x0 = At(meas, Phi)
    Phi = Phi / Phi.max()

    Phi_sum = torch.sum(Phi ** 2, 2)
    Phi_sum[Phi_sum == 0] = 1
    Phi_sum = Phi_sum.to(device)
    theta = x0.to(device)
    b = torch.zeros_like(x0).to(device)
    iter_num, DP_iter = args.iter_num, args.DP_iter
    im_input = get_input([1, 28, 672, 672]).to(device)
    im_input1 = get_input([1, 28, 672, 672]).to(device)
    im_net = Modelreal(in_channels=28).cuda()
    im_net1 = Modelreal(in_channels=28).cuda()
    # im_net1.initialize_ex()
    # im_net1.initialize()
    im_net.train()
    im_net1.train()
    net_parameters = list(im_net.parameters())
    net_parameters1 = list(im_net1.parameters())
    params =  net_parameters
    params1 = net_parameters1
    optimizer = torch.optim.Adam(params=params, lr=0.001)
    optimizer1 = torch.optim.Adam(params=params1, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.5, last_epoch=-1)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer1, 500, gamma=0.5, last_epoch=-1)
    reg_noise_std = 1.0 / 50.0
    loss_fn = torch.nn.L1Loss().to(device)
    loss_fn2 = nn.MSELoss().to(device)
    mask = Phi.permute(2, 0, 1).to(device)
    meas = meas.to(device)
    theta = x0.to(device)
    theta1 = x0.to(device)
    rou = 0.01
    plus = 1.01
    best_PSNR = 0
    begin_time = time.time()
    # ---------------- Iteration ----------------#
    for iter in range(iter_num):
        #rou = plus * rou

        if iter < 30:
            c = theta.to(device) + b.to(device)
            meas_b = A(c, Phi)
            x = c + At((meas - meas_b) / (Phi_sum ), Phi)
            x1 = shift_back(x - b, 2)

            theta = TV_minimization(x1, 90, 10)
            theta = shift(theta, 2)
        else:
            im_input_perturbed = im_input + im_input.clone().normal_() * reg_noise_std
            model_out = im_net(im_input_perturbed)[:, :, :660, :660]
            pred_response = getresponse(model_out, mask, 2)
            im_input_perturbed1 = im_input1 + im_input1.clone().normal_() * reg_noise_std
            model_out1 = im_net1(im_input_perturbed1)[:, :, :660, :660]
            pred_response1 = getresponse(model_out1, mask, 2)
            theta = model_out.detach().squeeze(0).permute(1, 2, 0)
            theta1 = model_out1.detach().squeeze(0).permute(1, 2, 0)
            c = shift((theta + theta1)/2, 2).cuda()
            meas_b = A(c, Phi)
            x = c + At((meas - meas_b) / (Phi_sum + 2*rou), Phi)
            for_lossx = shift_back(x, 2).permute(2, 0, 1).unsqueeze(0)
            loss = loss_fn(pred_response, meas)*0.05
            loss += loss_fn(for_lossx, model_out)
            loss1 = loss_fn(pred_response1, meas)*0.05
            loss1 += loss_fn(for_lossx, model_out1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            scheduler.step()
            scheduler_1.step()
        if iter%100 == 0:
            # x1 = shift(x, 2)
            ox = shift_back(x, 2)
            # ox = ox/ox.max()
            psnr_x, ssim_x, _ = MSIQA(data_truth.permute(2, 0, 1).detach().cpu().numpy(), ox.permute(2, 0, 1).detach().cpu().numpy())
            end_time = time.time()
            print('Iter {} | PSNR = {:.2f}dB | SSIM = {:.3f} | lr = {:.6f}| time = {:.3f}'.format(iter, psnr_x, ssim_x,
                                                                                   optimizer.state_dict()[
                                                                                       'param_groups'][0]['lr'],(end_time - begin_time)))
            if iter>= 30:
                psnr_x2, ssim_x, _ = MSIQA(data_truth.permute(2, 0, 1).detach().cpu().numpy(),
                                          model_out.detach().squeeze(0).cpu().numpy())
                print('Iter {} | PSNR = {:.2f}dB | SSIM = {:.3f} | lr = {:.6f}'.format(iter, psnr_x2, ssim_x, optimizer.state_dict()['param_groups'][0]['lr']))

            if  iter > 100 and iter%100 == 0:
                print('-------------------------save_mat------------------------------------')
                sio.savemat(result_dir + '/Iter-{}-PSNR-{:2.2f}-SSIM-{:2.2f}.mat'.format(iter, psnr_x, ssim_x), {'img': ox.detach().cpu().numpy()})
                best_PSNR = psnr_x
                show_rgb(ox.detach().cpu(), iter, args.scene)

    return theta


