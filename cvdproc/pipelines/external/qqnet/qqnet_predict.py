# -*- coding: utf-8 -*-

# modified from: https://github.com/jc2852/QQ-NET,
# https://wcm.box.com/s/v1thrs0ezuuhb13cdho1xaoiu3bd9h5j,
# the 'QQ_NET_test_simul.ipynb' file

# Youjie Wang, 2025-08-24

import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pylab as plt
import matplotlib as matp
import numpy as np
import scipy.io as sio
from cvdproc.pipelines.external.qqnet.basics.utils_u import *
#from basics.utils_u import *
from cvdproc.pipelines.external.qqnet.utils import *
#from utils import *
import array as arr
from cvdproc.pipelines.external.qqnet.basics.unet3d_b_limit_m import unet3d
#from basics.unet3d_b_limit_m import unet3d
from torch.utils import data
from numpy import zeros
from cvdproc.pipelines.external.qqnet.JH_funcs.funcJH import * 
import time as time
import nibabel as nib
from typing import List, Tuple
from scipy.ndimage import zoom

# functions need
def flip_y_axis(arr3d: np.ndarray) -> np.ndarray:
    return arr3d[:, ::-1, :]

def flip_y_axis_4d(arr4d: np.ndarray) -> np.ndarray:
    return arr4d[:, ::-1, :, :]

def resample_to_shape(arr3d: np.ndarray, target_shape: Tuple[int,int,int], order: int) -> np.ndarray:
    src_shape = arr3d.shape
    factors = tuple(t / s for t, s in zip(target_shape, src_shape))
    return zoom(arr3d, zoom=factors, order=order)

def dxp_3D(a):
    return torch.cat((a[1:,:,:], a[-1:,:,:]), dim=0) - a

def dyp_3D(a):
    return torch.cat((a[:,1:,:], a[:,-1:,:]), dim=1) - a

def dzp_3D(a):
    return torch.cat((a[:,:,1:], a[:,:,-1:]), dim=2) - a

# package to a function
def qqnet_predict(
    mag_4d_path: str,
    qsm_path: str,
    mask_path: str,
    output_dir: str,
    prefix: str = "QQNET",
    r2star_path: str = None,
    s0_path: str = None,
    header_path: str = None
) -> dict:
    """
    Predicts physiological parameters using the QQ-NET model.

    Parameters
    ----------
    mag_4d_path : str
        Path to the 4D magnitude image.
    qsm_path : str
        Path to the QSM image.
    mask_path : str
        Path to the brain mask image.
    output_dir : str
        Output directory for the results.
    prefix : str, optional
        Prefix for the output files, by default "QQNET".
    r2star_path : str, optional
        Path to the R2* image, by default None.
    s0_path : str, optional
        Path to the S0 image, by default None.
    header_path : str, optional
        Path to the sepia header file, by default None.

    Returns
    -------
    dict
        A dictionary containing paths to the output images.
    """
    #########################################################################
    #                             Prepare Input                             #
    #########################################################################
    mag_4d_img = nib.load(mag_4d_path)
    mag_4d = mag_4d_img.get_fdata()
    qsm_img = nib.load(qsm_path)
    qsm = qsm_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    mask_orig = mask.copy()

    # handle echos
    # only keep the first 8 echoes if more than 8
    if mag_4d.shape[3] > 8:
        print('More than 8 echoes, only keep the first 8 echoes.')
        mag_4d = mag_4d[:,:,:,0:8]
    if mag_4d.shape[3] < 8:
        print('Less than 8 echoes, synthesize more echoes to make it 8 echoes.')
        mag_4d_8echo = np.zeros((mag_4d.shape[0], mag_4d.shape[1], mag_4d.shape[2], 8), dtype=np.float32)
        mag_4d_8echo[:,:,:,:mag_4d.shape[3]] = mag_4d
        echo_n = mag_4d.shape[3]
        # must have r2star_path, s0_path, header_path to synthesize more echoes
        if r2star_path is None or s0_path is None or header_path is None:
            raise ValueError("To synthesize more echoes, r2star_path, s0_path, and header_path must be provided.")
        te = sio.loadmat(header_path)['TE'].squeeze()  # in s
        delta_te = sio.loadmat(header_path)['delta_TE'].squeeze()  # in s

        r2s_img = nib.load(r2star_path)
        r2s = r2s_img.get_fdata().astype(np.float32)   # 1/s
        s0_img = nib.load(s0_path)
        s0 = s0_img.get_fdata().astype(np.float32)

        echo_n_to_add = 8 - echo_n
        for i in range(echo_n_to_add):
            te_extra = te[-1] + delta_te * (i + 1)
            synth = s0 * np.exp(-r2s * float(te_extra))
            mag_4d_8echo[:,:,:,echo_n + i] = synth
        mag_4d = mag_4d_8echo

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # resample to target shape and direction
    TARGET_SHAPE: Tuple[int, int, int] = (512, 512, 70)

    mag_4d = flip_y_axis_4d(mag_4d)
    qsm = flip_y_axis(qsm)
    mask = flip_y_axis(mask)

    # create a new mag_4d (512 512 70 8)
    new_mag_4d = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2], mag_4d.shape[3]), dtype=np.float32)

    for i in range(mag_4d.shape[3]):
        new_mag_4d[:,:,:,i] = resample_to_shape(mag_4d[:,:,:,i], TARGET_SHAPE, order=1)
    qsm = resample_to_shape(qsm, TARGET_SHAPE, order=1)
    mask = resample_to_shape(mask, TARGET_SHAPE, order=0)

    # again mask the qsm and mag
    mask = (mask > 0).astype(np.float32)
    qsm = qsm * mask
    for i in range(new_mag_4d.shape[3]):
        new_mag_4d[:,:,:,i] = new_mag_4d[:,:,:,i] * mask

    # normalize the mag by the 1st echo (voxel by voxel)
    divider = new_mag_4d[:,:,:,0] + 1e-6

    for i in range(new_mag_4d.shape[3]):
        new_mag_4d[:,:,:,i] = new_mag_4d[:,:,:,i] / divider

    # stack to a 9 channel input (8-echo mag + QSM)
    input_9ch = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2], 9), dtype=np.float32)
    for i in range(new_mag_4d.shape[3]):
        input_9ch[:,:,:,i] = new_mag_4d[:,:,:,i]
    input_9ch[:,:,:,8] = qsm

    print('Input prepared!')

    #########################################################################
    #                            Main Prediction                            #
    #########################################################################
    print('Start Prediction ...')
    p_s_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data0', 'p_s.mat'))
    p_s= np.real(load_mat(p_s_path, varname='p_s')) # p_s should be (59, 1)

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda:0") # GPU

    avgs = [1.0983, 20.1093, 0.6833, 0.0183, -0.0100]
    stds = [0.0396, 7.2432, 0.1003, 0.0109/2, 0.0371]

    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # parameter boundary
    # [S0,R2,Y,v,chinb]
    ymins = [0.8,2,0,0,-1]
    ymaxs = [2.5,190,0.98,0.08,1.6]
    xmins = zeros([5,1])
    xmaxs = zeros([5,1])

    xmins[0] = (ymins[0]-avgs[0])/stds[0]
    xmins[1] = (ymins[1]-avgs[1])/stds[1]
    xmins[2] = (ymins[2]-avgs[2])/stds[2]
    xmins[3] = (ymins[3]-avgs[3])/stds[3]
    xmins[4] = (ymins[4]-avgs[4])/stds[4]

    xmaxs[0] = (ymaxs[0]-avgs[0])/stds[0]
    xmaxs[1] = (ymaxs[1]-avgs[1])/stds[1]
    xmaxs[2] = (ymaxs[2]-avgs[2])/stds[2]
    xmaxs[3] = (ymaxs[3]-avgs[3])/stds[3]
    xmaxs[4] = (ymaxs[4]-avgs[4])/stds[4]

    learning_rate = 0.0001
    wd = 0.0; # weight decay (L2 regularization on weights)

    #  model loading
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'models', 'qqnet', 'QQ_NET_trained_model.pt'))
    model = unet3d()
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)

    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    Y_ss = 0.6344

    Output3 = input_9ch.copy()
    Mask3 = mask.copy()

    batch_size =[256,256,32]
    overlap = 0.5
    sx = np.around(batch_size[0]*(1-overlap)).astype(int)
    sy = np.around(batch_size[1]*(1-overlap)).astype(int)
    sz = np.around(batch_size[2]*(1-overlap)).astype(int)
    EXTRACTION_STEP = (sx,sy,sz)
    batch_size1 = (batch_size[0],batch_size[1],batch_size[2])

    Output3.shape[3]
    xx = extract_patches(Output3[..., 0], batch_size1, EXTRACTION_STEP)
    xx_M = extract_patches(Mask3[...], batch_size1, EXTRACTION_STEP)
    xx_M1 = np.zeros([xx_M.shape[0],batch_size[0],batch_size[1],batch_size[2],1])
    xx_M1[:,:,:,:,0] = xx_M

    batch_total = np.zeros([xx.shape[0],batch_size[0],batch_size[1],batch_size[2],Output3.shape[3]])

    for p in range(Output3.shape[3]):  
        xx = extract_patches(Output3[..., p], batch_size1, EXTRACTION_STEP)
        batch_total[:,:,:,:,p] = xx[:,:,:,:]

    batch_test = np.zeros([batch_total.shape[0],5,batch_size[0],batch_size[1],batch_size[2]])
    checkc = round(batch_total.shape[0]/20)

    with torch.no_grad():
        for q in range(batch_total.shape[0]): 

            model = model.cuda()

            Output_temp = batch_total[q,:,:,:,:]
            Output_temp1 = np.transpose(Output_temp,(3,2,1,0))

            Mask_temp = xx_M1[q,:,:,:,:]
            Mask_temp1 = np.transpose(Mask_temp,(3,2,1,0))
            Mask_batch = zeros([1, Mask_temp1.shape[0], Mask_temp1.shape[1], Mask_temp1.shape[2], Mask_temp1.shape[3]])
            Mask_batch[0,:,:,:,:] = Mask_temp1[:,:,:,:]
            Mask_t1 = torch.DoubleTensor(Mask_batch).cuda()

            # Change to 5D  [1,channel,z,y,x]  
            ppx = Output_temp1.shape
            Output1_batch = zeros([1,ppx[0],ppx[1],ppx[2],ppx[3]])
            Output1_batch[0,:,:,:,:] = Output_temp1
            # y_batch = model(torch.FloatTensor(Output1_batch).cuda())
            y_batch = model(torch.DoubleTensor(Output1_batch).cuda(),xmins,xmaxs,Mask_t1)

            y_batch1 = y_batch.cpu().numpy()
            y_batch2 = np.transpose(y_batch1,(0,1,4,3,2))
            batch_test[q,:,:,:,:] = y_batch2[0,:,:,:,:]

    Target_test  = np.zeros([batch_test.shape[1],Output3.shape[0],Output3.shape[1],Output3.shape[2]])
    for h in range(batch_test.shape[1]):
        temp = batch_test[:,h,:,:,:]
        recon = reconstruct_patches(temp,Output3.shape[:3], EXTRACTION_STEP)
        Target_test[h,:,:,:] = recon[:,:,:]

    Target_test[0,:,:,:] = Target_test[0,:,:,:]*stds[0] + avgs[0]
    Target_test[1,:,:,:] = Target_test[1,:,:,:]*stds[1] + avgs[1]
    Target_test[2,:,:,:] = Target_test[2,:,:,:]*stds[2] + Y_ss
    Target_test[3,:,:,:] = Target_test[3,:,:,:]*stds[3] + avgs[3]
    Target_test[4,:,:,:] = Target_test[4,:,:,:]*stds[4] + avgs[4]
    Target_test = np.float32(Target_test)
    Mask3 = np.float32(Mask3)

    print('Prediction Done!')

    #########################################################################
    #                              Save Results                             #
    #########################################################################
    # we need to return results to the original space (no need for mask)
    # 1. resample to original shape
    # 2. flip y axis
    # 3. save as nii.gz

    Target_test_resamp = np.zeros((mag_4d.shape[0], mag_4d.shape[1], mag_4d.shape[2], Target_test.shape[0]), dtype=np.float32)
    for i in range(Target_test.shape[0]):
        Target_test_resamp[:,:,:,i] = resample_to_shape(Target_test[i,:,:,:], mag_4d.shape[:3], order=1)
    Target_test_resamp = flip_y_axis_4d(Target_test_resamp)

    # apply original mask
    for i in range(Target_test_resamp.shape[3]):
        Target_test_resamp[:,:,:,i] = Target_test_resamp[:,:,:,i] * mask_orig

    # save
    S0_img = nib.Nifti1Image(Target_test_resamp[:,:,:,0], mag_4d_img.affine, mag_4d_img.header)
    S0_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_S0.nii.gz')
    nib.save(S0_img, S0_path)
    R2_img = nib.Nifti1Image(Target_test_resamp[:,:,:,1], mag_4d_img.affine, mag_4d_img.header)
    R2_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_R2.nii.gz')
    nib.save(R2_img, R2_path)
    Y_img = nib.Nifti1Image(Target_test_resamp[:,:,:,2], mag_4d_img.affine, mag_4d_img.header)
    Y_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_Y.nii.gz')
    nib.save(Y_img, Y_path)
    v_img = nib.Nifti1Image(Target_test_resamp[:,:,:,3], mag_4d_img.affine, mag_4d_img.header)
    v_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_v.nii.gz')
    nib.save(v_img, v_path)
    chinb_img = nib.Nifti1Image(Target_test_resamp[:,:,:,4], mag_4d_img.affine, mag_4d_img.header)
    chinb_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_Chinb.nii.gz')
    nib.save(chinb_img, chinb_path)

    # calculate OEF (1 - Y/Ya), Ya = SaO2 = 0.98
    Ya = 0.98
    OEF = 1 - Target_test_resamp[:,:,:,2] / Ya
    # mask it
    OEF = OEF * mask_orig
    OEF_img = nib.Nifti1Image(OEF, mag_4d_img.affine, mag_4d_img.header)
    OEF_path = os.path.join(output_dir, f'{prefix}_desc-QQNET_OEF.nii.gz')
    nib.save(OEF_img, OEF_path)

    print('All Done!')

    output_dict = {
        "chinb": chinb_path,
        "oef": OEF_path,
        "r2": R2_path,
        "s0": S0_path,
        "v": v_path,
        "y": Y_path
    }

    return output_dict