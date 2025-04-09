#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:50:53 2023

@author: dw16, fmglang
"""

import time
import matplotlib.pyplot as plt
from termcolor import colored
import MRzeroCore as mr0
import torch
import os
from EPI_EC_builder import EPI2D_EC
from GRE3D_builder import GRE3D
import util
from reconstruction import sos, reconstruct, get_EPI_kspace
from grappa import create_grappa_weight_set, grappa_imspace
import ec_tools
from sensitivity_tools import load_external_coil_sensitivities3D
from scipy.io import loadmat
from pulseq_exporter import pulseq_write_EPG_3D_EC_EPI, pulseq_write_EPG_3D

### SCAN PARAMETERS ###
pulseq         = 0

FOV_export     = 250
FOV_sim        = FOV_export*1e-3

DO_IMAGE_SIM   = True
DO_IMAGE_OP    = True

res     = 129
partial_fourier = 1 # make sure that res / R_accel * partial_fourier is an integer (=number of measured lines)
nseg    = 1  # epi segmentation
navg    = 1  # number of repeated epi images (mainly to see transient state effects)
TRfill  = 0  # seconds at the end of each shot

### GRAPPA START ###
R_accel     = 3   # acceleration factor
NACS        = res  # number of autocalibration lines
kernel_size = None # None=automatic (according to some heuristic)
lambd       = 0.01 # Tikhonov regularization for GRAPPA weight fitting
### GRAPPA END ###

smax = 180
gmax = 72e-3

experiment_id = 'GRAPPA3_7R'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None # file name which should be loaded
util.use_gpu = False 

### OPTIMIZATION PARAMETERS ###
MAXITER = 50000
TOTITER = 50000

# %% Loading of simulation data.

# sequence and reconstruction resolution
size = (res, res, 1)
size_sim = (res, res, 1)

# load phantom
phantom = util.VoxelGridPhantomMod.brainweb("data/subject20.npz")
phantom = phantom.slices([70]).interpolate(*size_sim) # FG: slice 60 seems to match roughly what we had before (70 according to old brainweb data handling)

# load and add Rx coil sensitivities
NCoils = 32
coil_sens = load_external_coil_sensitivities3D('data/B1minus_32ch_7T.mat', NCoils, size_sim)

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

# %% Simulate target.

size_tmp = [res,res]
params_target = EPI2D_EC(res, nseg, navg, TRfill=TRfill, R_accel=R_accel, partial_fourier=partial_fourier)

seq_full = params_target.generate_sequence()
seq_full = mr0.sequence.chain(*seq_full)
if util.use_gpu:
    seq_full = seq_full.cuda()

GIRF = loadmat('7T_GIRF.mat')
    
# export initial sequence
if pulseq == 1:
    pulseq_write_EPG_3D_EC_EPI(seq_full, 'GRAPPA3_7R1.seq', FOV_export, GIRF, plot_seq=True, smax=smax, R_accel=R_accel)

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

plt.figure()
plt.imshow(torch.rot90(reco_acs_sos.cpu()))
plt.title(f'ACS scan ({NACS} lines)')

# get grappa kernel for later use
ksp_acs = signal_acs.reshape([res, NACS, NCoils]).permute([2,1,0]).unsqueeze(-1) # coils x phase x read x 1
wsKernel, ws_imspace = create_grappa_weight_set(
    ksp_acs, R_accel, 1, 0, size, lambd=lambd, kernel_size=kernel_size)

if pulseq:
    pulseq_write_EPG_3D(seq_acs, 'GRAPPA3_7R_ACS.seq', FOV_export, plot_seq=True)

# %% GIRF perturbation

gmoms1 = torch.cat([rep.gradm for rep in seq_full], dim=0)
timeaxis = torch.cumsum(torch.cat([rep.event_time for rep in seq_full], dim=0), dim=0)

seq_full_perturbed =  ec_tools.GIRF_perturbation(seq_full, FOV_sim, GIRF)

gmoms2 = torch.cat([rep.gradm for rep in seq_full_perturbed], dim=0)
    
plt.figure(3)
plt.subplot(231)
plt.plot(timeaxis.cpu(), gmoms1[:,0].cpu().detach().numpy())
plt.plot(timeaxis.cpu(), gmoms2[:,0].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

plt.subplot(232)
plt.plot(timeaxis.cpu(), gmoms1[:,1].cpu().detach().numpy())
plt.plot(timeaxis.cpu(), gmoms2[:,1].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

plt.subplot(233)
plt.plot(kloc_unperturbed[:,0].cpu())
plt.plot(kloc_perturb[:,0].cpu())
plt.plot(kloc_unperturbed[:,1].cpu())
plt.plot(kloc_perturb[:,1].cpu())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

# %% Compare perturbed and unperturbed.

if DO_IMAGE_SIM:
    graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
    graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)
    
    # simulate unperturbed
    target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)
    
    # simulate perturbed
    target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)

# %% Test GRAPPA reco.

# apply grappa kernel to undersampled dataset
ksp_target_unperturbed, reco_zf = get_EPI_kspace(seq_full,target_signal_full_unperturbed,size,nseg,R_accel, pf=partial_fourier)
reco_grappa = grappa_imspace(ksp_target_unperturbed, ws_imspace)
reco_grappa_sos = sos(reco_grappa) / (size[0]*size[1]) # FG: somehow empirical scaling again... probably can be figured out analytically, but due to different echo train lengths etc. the signal levels of non-accelerated and accelerated EPI will always be different

# %% Define loss and helper functions.

torch.cuda.empty_cache()
gif_array = []
loss_history_gauss = []

if DO_IMAGE_SIM:
    
    target = grappa_imspace(
        get_EPI_kspace(seq_full,target_signal_full_unperturbed,size,nseg,R_accel,pf=partial_fourier)[0],
        ws_imspace)
    target = sos(target) / (size[0]*size[1])
    
    reco_0 = grappa_imspace(
        get_EPI_kspace(seq_full_perturbed,target_signal_full_perturbed,size,nseg,R_accel,pf=partial_fourier)[0],
        ws_imspace)
    reco_0 = sos(reco_0) / (size[0]*size[1])
    
    plt.figure(3)
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
    plt.imshow(torch.rot90(reco_0.cpu()*1e4))
    cbar = plt.colorbar(fraction=0.0453)
    plt.set_cmap('viridis')
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('x1e-4',fontsize=16)
    tx = cbar.ax.yaxis.get_offset_text()
    tx.set_fontsize(14)
    plt.xticks(fontsize=14), plt.yticks(fontsize=14),
    plt.subplot(236)
    plt.imshow(torch.abs(torch.rot90(target.cpu()*1e4)-torch.rot90(reco_0.cpu()*1e4)))
    cbar = plt.colorbar(fraction=0.0453)
    plt.set_cmap('gray')
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('x1e-4',fontsize=16)
    tx = cbar.ax.yaxis.get_offset_text()
    tx.set_fontsize(14)
    plt.xticks(fontsize=14), plt.yticks(fontsize=14)

class opt_history:
    def __init__(self):
        self.loss_history = []
        self.FA = []
opt_history = opt_history()

f = open(experiment_id+'.txt','w')

def calc_loss(gradm_all: torch.Tensor,
              params: EPI2D_EC,
              iteration: int):
    
    ### MAIN LOSS FUNCTION ###
    global gmoms3, gmoms4
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    if util.use_gpu:
        seq = seq.cuda()
    
    # plug back all grad_moms
    for jj in range(gradm_all.shape[2]):
        seq[jj].gradm = gradm_all[:,:,jj]
        
    gmoms3 = gradm_all
    
    # GIRF perturbation
    seq, slew_x, slew_y, waveform_x,waveform_y, waveformp_x, waveformp_y = ec_tools.GIRF_perturbation(seq, FOV_sim, GIRF, return_slew=True)
    
    global graph
    
    if DO_IMAGE_OP:
        # forward simulation
        signal = mr0.execute_graph(graph, seq, data)
        
        # reco: naive FFT + sum-of-squares coil combine
        reco = grappa_imspace(
            get_EPI_kspace(seq,signal,size,nseg,R_accel, pf=partial_fourier)[0],
            ws_imspace)
        reco = sos(reco) / (size[0]*size[1])
    
    # perturbed kspace locations
    kloc_perturb = seq.get_kspace()
    
    if (iteration == 1) or (iteration % 5000 == 0):

        plt.figure(figsize=(15, 20))       
        plt.subplot(421)        
        plt.plot(kloc_unperturbed[:,0].cpu().detach().numpy(), kloc_unperturbed[:,1].cpu().detach().numpy(), 'x', label='target')
        plt.plot(kloc_perturb[:,0].cpu().detach().numpy(), kloc_perturb[:,1].cpu().detach().numpy(), '.', label='current')
        plt.title("sampling locations"), plt.legend()        
        
        if DO_IMAGE_OP:
            plt.subplot(422)
            plt.imshow(util.to_numpy(torch.abs((reco)).transpose(2,1).reshape(size[0],size[1]*size[2])))
            plt.colorbar()
            plt.title("Reco")       
            plt.subplot(424)
            plt.imshow(util.to_numpy((torch.abs(target)).transpose(2,1).reshape(size[0],size[1]*size[2])))
            plt.colorbar()
            plt.title("Target")       
        
        ax=plt.subplot(423)
        quotients = [number / opt_history.loss_history[0] for number in opt_history.loss_history] # Normalized
        plt.plot(quotients)
        ax.set_yscale('log')
        plt.grid()
        plt.title("Loss Curve")
        
        gmoms4 = torch.cat([rep.gradm.unsqueeze(-1).clone()
                            for rep in seq], dim=2)    

        shot1_end = int(len(waveform_x))        
        plt.subplot(413)
        plt.plot(waveform_x[:shot1_end].cpu().detach()*1e3, label='waveform x')
        plt.plot(waveform_y[:shot1_end].cpu().detach()*1e3, label='waveform y')
        plt.ylabel('waveform (mT/m)'), plt.legend()
        plt.title(f'max = {torch.max(torch.abs(waveform_x).cpu().detach())*1e3:.2f} (x), {torch.max(torch.abs(waveform_y).cpu().detach())*1e3:.2f} (y)')
        plt.xlim(0,20000)
        
        plt.subplot(414)
        plt.plot(slew_x[:shot1_end].cpu().detach(), label='slew x')
        plt.plot(slew_y[:shot1_end].cpu().detach(), label='slew y')
        plt.ylabel('slew rate (T/m/s)'), plt.legend()
        plt.title(f'max = {torch.max(torch.abs(slew_x).cpu().detach()):.2f} (x), {torch.max(torch.abs(slew_y).cpu().detach()):.2f} (y)')
        plt.xlim(0,20000)

        plt.suptitle(f'iter {iteration}')

        gif_array.append(util.current_fig_as_img())
        plt.show()

    if iteration == 1:
        if DO_IMAGE_OP:
            checkout = {
                'reco_target': target,
                'klocs_target': kloc_unperturbed,
                'klocs_perturbed': kloc_perturb,
                'reco_perturb':reco,
                'slewc':smax,
                'gradc':gmax,          
                }
        else:
            checkout = {
                'klocs_target': kloc_unperturbed,
                'klocs_perturbed': kloc_perturb,
                'slewc':smax,
                'gradc':gmax,          
                }
        torch.save(checkout, os.path.join(path, experiment_id+'1.pth'))
    
    if iteration == TOTITER:
        if DO_IMAGE_OP:
            checkout = {
                'reco_opt': reco,
                'klocs_opt': kloc_perturb,
                'loss_history': opt_history.loss_history,      
                'gmoms1':gmoms1,
                'gmoms2':gmoms2,
                'gmoms3':gmoms3,
                'gmoms4':gmoms4,
                'FAs':params.pulse_angles
                }
        else:
            checkout = {
                'klocs_opt': kloc_perturb,
                'loss_history': opt_history.loss_history,      
                'gmoms1':gmoms1,
                'gmoms2':gmoms2,
                'gmoms3':gmoms3,
                'gmoms4':gmoms4,
                'FAs':params.pulse_angles
                }
        torch.save(checkout, os.path.join(path, experiment_id+'2.pth'))

    loss_image = torch.tensor(0.0, device=util.get_device())
    if DO_IMAGE_OP:
        loss_image = util.MSR(reco, target, root=True)
    
    # klocation loss: euclidian distance of kspace sampling locations to 'optimal' ones
    loss_kloc = torch.sum((torch.abs(kloc_perturb[:,0:3] - kloc_unperturbed[:,0:3])**2).flatten())
    
    # slew rate penalty
    slew = torch.cat([slew_x, slew_y])
    loss_slew = torch.tensor(0.0, device=util.get_device())
    loss_slew = torch.abs(slew.flatten()) - smax
    loss_slew[loss_slew < 0] = 0 # only keep exceeding values
    loss_slew = torch.sum(loss_slew) # sum of all slew exceedances
    
    # gradient amplitude penalty
    gamp = torch.cat([waveform_x, waveform_y])
    loss_gamp = torch.tensor(0.0, device=util.get_device())
    loss_gamp = torch.abs(gamp.flatten()) - gmax
    loss_gamp[loss_gamp < 0] = 0 # only keep exceeding values
    loss_gamp = torch.sum(loss_gamp) # sum of all slew exceedances
    
    indices = torch.tensor(list(range(200)), device=util.get_device())
    xp_result = torch.index_select(waveformp_x,0,indices)
    yp_result = torch.index_select(waveformp_y,0,indices)
    gampRF = torch.cat([xp_result,yp_result])
    loss_RF = torch.tensor(0.0, device=util.get_device())
    loss_RF = torch.sum(gampRF**2)

    if torch.isnan(loss_image) or torch.isnan(loss_kloc) or torch.isnan(loss_gamp) or torch.isnan(loss_RF):
        print("NaNs PRESENT")
    
    # Lambdas
    lbd_image = 0
    lbd_kloc  = 1
    lbd_slew  = 10000
    lbd_gamps = 10000
    lbd_RF    = 1
    
    loss = (lbd_image*loss_image +
            lbd_kloc*loss_kloc +
            lbd_slew*loss_slew +
            lbd_gamps*loss_gamp + 
            lbd_RF*loss_RF) 
        
    opt_history.loss_history.append(loss.detach().cpu())
    
    print(f"{lbd_image*loss_image:.12f},"+f"{lbd_kloc*loss_kloc:.12f},"+f"{lbd_slew*loss_slew:.12f},"+f"{lbd_gamps*loss_gamp:.12f},"+f"{lbd_RF*loss_RF:.12f}\n",file=f)
    print(
        "% 4d |  image %s | gamp %s | kloc %s | slew %s | RF %s loss %s | "
        % (
            iteration,
            colored(f"{lbd_image*loss_image.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_gamps*loss_gamp.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_kloc*loss_kloc.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_slew*loss_slew.detach().cpu():.3e}", 'green'),
            colored(f"{lbd_RF*loss_RF.detach().cpu():.3e}", 'green'),
            colored(f"{loss.detach().cpu():.3e}", 'yellow'),
        )
    )
    return loss

# %% Optimization

# define the starting parameters for the optimisation process
size_tmp = [res,res]
params = EPI2D_EC(res, nseg, navg, TRfill=TRfill, R_accel=R_accel, partial_fourier=partial_fourier)

seq_opt = params.generate_sequence()
seq_opt = mr0.sequence.chain(*seq_opt)
if util.use_gpu:
    seq_opt = seq_opt.cuda()

gradm_all = torch.cat([rep.gradm.unsqueeze(-1).clone()
                        for rep in seq_opt], dim=2).to(util.get_device())

gradm_all.requires_grad = True

optimizable_params = [
    {'params': gradm_all, 'lr': 5e-6},
]

NRestarts = 1
NIter = MAXITER

t0 = time.time()

iteration = 0
for restart in range(NRestarts):
    optimizer = torch.optim.Adam(optimizable_params, lr=5e-6, betas = [0.9, 0.999])
        
    for i in range((restart + 1) * NIter):
        iteration += 1

        if i % 10 == 0:
            graph = mr0.compute_graph(seq_opt, data, max_state_count, min_state_mag)
        
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(False)
        optimizer.zero_grad()
        
        loss = calc_loss(gradm_all, params, iteration)
        loss.backward()
        optimizer.step()
                
f.close()

# %% Export optimal sequence.

if pulseq == 1:
    seq = params.generate_sequence()
    seq = mr0.sequence.chain(*seq)
    
    # plug back all grad_moms
    for jj in range(gradm_all.shape[2]):
        seq[jj].gradm = gradm_all[:,:,jj].detach()
        
    pulseq_write_EPG_3D_EC_EPI(seq, 'GRAPPA3_7R2.seq', FOV_export, GIRF, plot_seq=True, smax=smax+1, R_accel=R_accel)