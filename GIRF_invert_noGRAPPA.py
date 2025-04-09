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
import util
from reconstruction import sos, reconstruct_EPI_fft_naive
import MRzeroCore as mr0
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
from scipy.io import loadmat
import numpy as np
import matplotlib.ticker as ticker

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

### SCAN PARAMETERS ###
FOV_export      = 250
FOV_sim         = FOV_export*1e-3
res             = 96
partial_fourier = 73/96
smax            = 180
gmax            = 72e-3

experiment_id = 'noGRAPPA_GIRFinv'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None
util.use_gpu = True
           
# %% Loading of simulation data.

# sequence and reconstruction resolution
size = (res, res, 1)
size_sim = (res, res, 1)

# load phantom
phantom = util.VoxelGridPhantomMod.brainweb("data/subject20.npz")
phantom = phantom.slices([70]).interpolate(*size_sim) # FG: slice 60 seems to match roughly what we had before (70 according to old brainweb data handling)

# load and add Rx coil sensitivities
NCoils = 14
coil_sens = load_external_coil_sensitivities3D('data/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, size_sim)
phantom.coil_sens = coil_sens

if util.use_gpu:
    data = phantom.build().cuda()
else:
    data = phantom.build()
    
data.B0 = torch.tensor(0,dtype=torch.float32).to(util.get_device()) 
data.coil_sens = data.coil_sens / torch.abs(data.PD).sum()

# create target data for mapping
target_data = data

max_state_count = 1000
min_state_mag = 1e-9

# %% Simulate target fully sampled.

R_accel = (1,1) # [phase, partition]
size_tmp = [res,res]
params_target = EPI2D_EC(res, 1, partial_fourier=partial_fourier) # 1 = single shot

seq_full = params_target.generate_sequence()
seq_full = mr0.sequence.chain(*seq_full)
if util.use_gpu:
    seq_full = seq_full.cuda()

GIRF = loadmat('7T_GIRF.mat')
    
# %% GIRF perturbation

gmoms1 = seq_full[0].gradm

seq_full_perturbed =  ec_tools.GIRF_perturbation(seq_full, FOV_sim, GIRF)

gmoms2 = seq_full_perturbed[0].gradm

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)

target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)

target = sos(reconstruct_EPI_fft_naive(seq_full,target_signal_full_unperturbed,size, pf=partial_fourier))
reco0 = sos(reconstruct_EPI_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size, pf=partial_fourier))

# %% GIRF inversion

gmoms1 = seq_full[0].gradm # target waveforms
[seq_inverted, twx1, twy1, twx2, twy2, srx1, sry1, srx2, sry2] = ec_tools.GIRF_invert(seq_full, FOV_sim, GIRF, epsilon=1e-6, freq_limit = 45000, return_slew=True)
gmoms2 = seq_inverted[0].gradm # after inversion
[seq_final, srx3, sry3, twx3, twy3, twx4, twy4] = ec_tools.GIRF_perturbation(seq_inverted, FOV_sim, GIRF, return_slew=True)
gmoms3 = seq_final[0].gradm # after perturbation

kloc_final = seq_final.get_kspace()
    
graph_inverted = mr0.compute_graph(seq_inverted, data, max_state_count, min_state_mag)
graph_final = mr0.compute_graph(seq_final, data, max_state_count, min_state_mag)

target_signal_full_inverted = mr0.execute_graph(graph_inverted, seq_inverted, target_data)
target_signal_full_final = mr0.execute_graph(graph_final, seq_final, target_data)

recoi = sos(reconstruct_EPI_fft_naive(seq_inverted,target_signal_full_inverted,size, pf=partial_fourier))
recof = sos(reconstruct_EPI_fft_naive(seq_final,target_signal_full_final,size, pf=partial_fourier))

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

G27_data1 = torch.load('opt_results/noGRAPPA_nramp7_PF1.pth')
G27_data2 = torch.load('opt_results/noGRAPPA_nramp7_PF2.pth')
G27t = G27_data1.get('reco_target')
G27p = G27_data1.get('reco_perturb')
G27o = G27_data2.get('reco_opt')
tw3x_NISO = moms2phys(G27_data2.get('gmoms3'),FOV) / dt
sr3x_NISO = (tw3x_NISO[1:,0,0] - tw3x_NISO[:-1,0,0]) / dt
kt_NISO = G27_data1.get('klocs_target')
kp_NISO = G27_data1.get('klocs_perturbed')
ko_NISO = G27_data2.get('klocs_opt')

max_sig = torch.max(target)
max_sig2 = torch.max(G27t)

dt = 10e-6
t_axis   = torch.linspace(0,dt*(torch.Tensor.size(srx2,0)-1),torch.Tensor.size(srx2,0))
idx1     = 0
idx2     = len(srx2)

# %% Plotting for Figure 2

fig1 = plt.figure(1)
plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "dimgray"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.5

gs   = fig1.add_gridspec(2,1)
ax1  = fig1.add_subplot(gs[1,:])
ax1.plot(1000*t_axis[idx1:idx2].cpu().detach(),srx3.cpu().detach().numpy(),linewidth=2)
ax1.plot(1000*t_axis[idx1:idx2-1].cpu().detach(),sr3x_NISO[idx1:idx2].cpu().detach().numpy(),linewidth=2)
ax1.set_yticks([-360,-270,-180,-90,0,90,180,270,360])
ax1.set_yticklabels(['-360','-270','-180','-90','0','90','180','270','360'])
ax1.set_ylim(-360,360)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(90))
ax1.grid(True)
ax1.axhspan(180, 360, facecolor='lightgray', alpha=0.5)
ax1.axhspan(-360, -180, facecolor='lightgray', alpha=0.5)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),plt.xlim(6,17.5)
plt.xlabel('Time [ms]',fontsize=16)
plt.ylabel('s [mT/m/ms]',fontsize=16)

ax2  = fig1.add_subplot(gs[0,:])
ax2.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*twx3.cpu().detach(),linewidth=2)
ax2.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*tw3x_NISO[idx1:idx2,0,0].cpu().detach(),linewidth=2)
ax2.grid(True)
ax2.set_ylim(-15,15)
plt.xlabel('Time [ms]',fontsize=16)
plt.ylabel('g [mT/m]',fontsize=16)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),plt.xlim(6,17.5)
ax2.legend(
    ['Pre-emphasis', 'NISO'],
    fontsize=17,
    loc='upper left',
    bbox_to_anchor=(0.745, 1.185),  
    ncol=2
)

default_blue = '#348ABD'
default_orange = '#E24A33'

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1,hspace=0.3,wspace=0.3)

fig2 = plt.figure(2)

gs  = fig2.add_gridspec(1,4)

ax4 = fig2.add_subplot(gs[0,0])
plt.imshow(np.rot90(to_numpy(torch.abs((target/max_sig)))))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('cividis')
cbar.ax.tick_params(labelsize=14)
plt.clim(0,1)
plt.title('NOMINAL',fontsize=22,fontweight='bold')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig2.add_subplot(gs[0,1])
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig2))))-np.rot90(to_numpy(torch.abs((G27p/max_sig2)))))))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
plt.clim(0,1)
plt.title('UNCORRECTED',fontsize=22,fontweight='bold')
ax5.set_xticklabels([])
ax5.set_yticklabels([])
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = fig2.add_subplot(gs[0,2])
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig))))-np.rot90(to_numpy(torch.abs((recof/max_sig)))))))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
plt.title('PRE-EMPHASIS',fontsize=22,fontweight='bold')
plt.clim(0,0.01)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.set_xticks([])
ax6.set_yticks([])

ax7 = fig2.add_subplot(gs[0,3])
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((target/max_sig2))))-np.rot90(to_numpy(torch.abs((G27o/max_sig2)))))))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
plt.title('NISO',fontsize=22,fontweight='bold')
plt.clim(0,0.01)
ax7.set_xticklabels([])
ax7.set_yticklabels([])
ax7.set_xticks([])
ax7.set_yticks([])

plt.subplots_adjust(left=0.03, right=0.95, top=0.95, bottom=0.05,hspace=0.3,wspace=0.3)

NRMSE_GMTF = util.NRMSE(target, recof)
NRMSE_NISO = util.NRMSE(G27t, G27o)
