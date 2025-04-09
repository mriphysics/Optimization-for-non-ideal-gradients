#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:50:53 2023

@author: dw16, fmglang
"""

import matplotlib.pyplot as plt
import torch
import os
from EPI_EC_builder import EPI2D_EC
from GRE3D_builder import GRE3D
import util
import MRzeroCore as mr0
from reconstruction import sos, reconstruct, get_EPI_kspace
from grappa import create_grappa_weight_set, grappa_imspace
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
from scipy.io import loadmat
import numpy as np

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

### SCAN PARAMETERS ###
FOV_export      = 250
FOV_sim         = FOV_export*1e-3
res             = 129
partial_fourier = 129/129
smax            = 180
gmax            = 72e-3
R_accel         = 3
nseg            = 1
navg            = 1
TRfill          = 0

NACS        = res
kernel_size = None
lambd       = 0.01

experiment_id = 'GRAPPA3_GIRFinv'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None
util.use_gpu = False

# %% Loading of simulation data.

# sequence and reconstruction resolution
size = (res, res, 1)
size_sim = (res, res, 1)

# load phantom
phantom = util.VoxelGridPhantomMod.brainweb("data/subject20.npz")
phantom = phantom.slices([70]).interpolate(*size_sim) # FG: slice 60 seems to match roughly what we had before (70 according to old brainweb data handling)

# load and add Rx coil sensitivities
# IF MULTISHOT:
# NCoils = 14
# coil_sens = load_external_coil_sensitivities3D('../../data/B1minus/tueb/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, size_sim)
NCoils = 32
coil_sens = load_external_coil_sensitivities3D('data/B1minus_32ch_7T.mat', NCoils, size_sim)

phantom.coil_sens = coil_sens

if util.use_gpu:
    data = phantom.build().cuda()
else:
    data = phantom.build()
    
data.B0 = torch.tensor(0,dtype=torch.float32).to(util.get_device()) 
data.coil_sens /= torch.abs(data.PD).sum()

# create target data for mapping
target_data = data

max_state_count = 1000
min_state_mag = 1e-9

# %% Simulate target fully sampled.

size_tmp = [res,res]
params_target = EPI2D_EC(res, nseg, navg, TRfill=TRfill, R_accel=R_accel, partial_fourier=partial_fourier)

seq_full = params_target.generate_sequence()
seq_full = mr0.sequence.chain(*seq_full)
if util.use_gpu:
    seq_full = seq_full.cuda()

GIRF = loadmat('7T_GIRF.mat')

# %% Simulate ACS

params_acs = GRE3D(size_tmp[0], NACS, 1)
seq_acs = params_acs.generate_sequence()
seq_acs = mr0.sequence.chain(*seq_acs)
if util.use_gpu:
    seq_acs = seq_acs.cuda()
graph_acs = mr0.compute_graph(seq_acs, data, max_state_count, min_state_mag)
signal_acs = mr0.execute_graph(graph_acs, seq_acs, target_data)
reco_acs = reconstruct(signal_acs, seq_acs.get_kspace(), size, return_multicoil=True, FOV=[1.0,1.0,1.0])
reco_acs_sos = sos(reco_acs)

ksp_acs = signal_acs.reshape([res, NACS, NCoils]).permute([2,1,0]).unsqueeze(-1)
wsKernel, ws_imspace = create_grappa_weight_set(
    ksp_acs, R_accel, 1, 0, size, lambd=lambd, kernel_size=kernel_size)

# %% GIRF perturbation

gmoms1 = seq_full[0].gradm

seq_full_perturbed =  ec_tools.GIRF_perturbation(seq_full, FOV_sim, GIRF)

gmoms2 = seq_full_perturbed[0].gradm
    
plt.figure(2)
plt.subplot(231)
plt.plot(gmoms1[:,0].cpu().detach().numpy())
plt.plot(gmoms2[:,0].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()
plt.subplot(232)
plt.plot(gmoms1[:,1].cpu().detach().numpy())
plt.plot(gmoms2[:,1].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

plt.subplot(233)
plt.plot(kloc_unperturbed[:,0].cpu())
plt.plot(kloc_perturb[:,0].cpu())
plt.plot(kloc_unperturbed[:,1].cpu())
plt.plot(kloc_perturb[:,1].cpu())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)

target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)

# IF MULTISHOT:
# target = sos(reconstruct_EPI_fft_naive(seq_full,target_signal_full_unperturbed,size, nseg))
# reco0 = sos(reconstruct_EPI_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size, nseg))
target = grappa_imspace(
    get_EPI_kspace(seq_full,target_signal_full_unperturbed,size,nseg,R_accel,pf=partial_fourier)[0],
    ws_imspace)
target = sos(target) / (size[0]*size[1])

reco0 = grappa_imspace(
    get_EPI_kspace(seq_full_perturbed,target_signal_full_perturbed,size,nseg,R_accel,pf=partial_fourier)[0],
    ws_imspace)
reco0 = sos(reco0) / (size[0]*size[1])

plt.subplot(234)
plt.imshow(torch.rot90(target.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.subplot(235)
plt.imshow(torch.rot90(reco0.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.subplot(236)
plt.imshow(torch.abs(torch.rot90(target.cpu()*1e4)-torch.rot90(reco0.cpu()*1e4)))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14)

# %% GIRF inversion

gmoms1 = seq_full[0].gradm # target waveforms
[seq_inverted, twx1, twy1, twx2, twy2, srx1, sry1, srx2, sry2] = ec_tools.GIRF_invert(seq_full, FOV_sim, GIRF, epsilon=1e-6, freq_limit = 45000, return_slew=True)
gmoms2 = seq_inverted[0].gradm # after inversion
[seq_final, srx3, sry3, twx3, twy3, twx4, twy4] = ec_tools.GIRF_perturbation(seq_inverted, FOV_sim, GIRF, return_slew=True)
gmoms3 = seq_final[0].gradm # after perturbation
    
graph_inverted = mr0.compute_graph(seq_inverted, data, max_state_count, min_state_mag)
graph_final = mr0.compute_graph(seq_final, data, max_state_count, min_state_mag)

target_signal_full_inverted = mr0.execute_graph(graph_inverted, seq_inverted, target_data)
target_signal_full_final = mr0.execute_graph(graph_final, seq_final, target_data)

# IF MULTISHOT:
# recoi = sos(reconstruct_EPI_fft_naive(seq_inverted,target_signal_full_inverted,size, nseg))
# recof = sos(reconstruct_EPI_fft_naive(seq_final,target_signal_full_final,size, nseg))
recoi = grappa_imspace(
    get_EPI_kspace(seq_inverted,target_signal_full_inverted,size,nseg,R_accel,pf=partial_fourier)[0],
    ws_imspace)
recoi = sos(recoi) / (size[0]*size[1])

recof = grappa_imspace(
    get_EPI_kspace(seq_final,target_signal_full_final,size,nseg,R_accel,pf=partial_fourier)[0],
    ws_imspace)
recof = sos(recof) / (size[0]*size[1])

# look at hardware limits
wxpc = (torch.max(twx3)/gmax)*100
wypc = (torch.max(twy3)/gmax)*100
sxpc = (torch.max(srx3)/smax)*100
sypc = (torch.max(sry3)/smax)*100

# load corresponding NISO results for plotting
gamma_ = 42.5764
dt   = 10e-6
FOV  = 250e-3
smax = 180

def moms2phys(moms, FOV):
    return moms / FOV / (gamma_*1e6)

G27_data1 = torch.load('opt_results/GRAPPA3_nramp71.pth')
G27_data2 = torch.load('opt_results/GRAPPA3_nramp72.pth')
G27t = G27_data1.get('reco_target')
G27o = G27_data2.get('reco_opt')
tw3x_NISO = moms2phys(G27_data2.get('gmoms3'),FOV) / dt
sr3x_NISO = (tw3x_NISO[1:,0,0] - tw3x_NISO[:-1,0,0]) / dt