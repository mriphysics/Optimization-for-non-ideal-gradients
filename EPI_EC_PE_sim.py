# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:11:34 2025

@author: dw16, fmglang
"""
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch
import os
from EPI_EC_builder import EPI2D_EC
import util
from reconstruction import sos, reconstruct_EPI_fft_naive
import ec_tools
import ec_tools_PE
from ec_tools import truncate
from sensitivity_tools import load_external_coil_sensitivities3D

FOV_export     = 250
FOV_sim        = FOV_export*1e-3

res             = 96
partial_fourier = 73/96

smax           = 500
gmax           = 50e-3

experiment_id = 'noGRAPPA_PE'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None # file name which should be loaded
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

# %% GIRF perturbation

gmoms1 = seq_full[0].gradm

seq_full_perturbed, srx, sry, twx, twy, _, _ = ec_tools.EC_perturbation_EPI(seq_full, FOV_sim, return_slew=True)

gmoms2 = seq_full_perturbed[0].gradm
    
# %% Optimize preemphasis parameters.
 
def preemph_or_perturb(alphas, taus, twx, twy, FOV, sign=+1, gradt=1e-5):
    
    '''
    Apply either EC pre-emphasis or EC perturbation to input waveforms twx, twy
    [alphas, taus]: EC params (variable for preemphasis case (user choice), fixed for perturbation (hardware property)!)
    sign: +1 -> preemph, -1 -> perturb
    
    returns resulting waveforms (x, y)
    '''

    srx = (twx[1:] - twx[:-1]) / gradt 
    srx = torch.cat((srx, torch.zeros([1]).to(util.get_device())))

    sry = (twy[1:] - twy[:-1]) / gradt 
    sry = torch.cat((sry, torch.zeros([1]).to(util.get_device())))

    maxtime = truncate(torch.tensor(twx.shape[0]*gradt),10)
    timings = torch.arange(0, maxtime, gradt).to(util.get_device())
    
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # sum up all exponentials
        ec_perturb += alpha*torch.exp(-timings/tau)
        
    # use neural network convolution
    responsex = torch.nn.functional.conv1d(srx.reshape([1,1,-1]), # [batch,channels=1,time]
                                         ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # flip as conv in machine learning terms is actually crosscorrelation, add singleton for batch & channel
                                         padding=len(ec_perturb) # figured out by playing around, not fully sure if always correct
                                         ).flatten() # back to original shape
    
    # in case of numerical precision error
    if ec_perturb.size(dim=0) > twx.size(dim=0):
        diff = ec_perturb.size(dim=0) - twx.size(dim=0)
        responsex = responsex[:len(ec_perturb)-diff]
    elif ec_perturb.size(dim=0) < twx.size(dim=0):
        diff = twx.size(dim=0) - ec_perturb.size(dim=0)
        diff_tensor = torch.zeros(diff)
        responsex = responsex[:len(ec_perturb)]
        responsex = torch.concat([responsex.cpu(),diff_tensor],-1)
    else:
        responsex = responsex[:len(ec_perturb)]
    
    responsey = torch.nn.functional.conv1d(sry.reshape([1,1,-1]), # [batch,channels=1,time]
                                         ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # flip as conv in machine learning terms is actually crosscorrelation, add singleton for batch & channel
                                         padding=len(ec_perturb) # figured out by playing around, not fully sure if always correct
                                         ).flatten() # back to original shape
        
    # in case of numerical precision error
    if ec_perturb.size(dim=0) > twy.size(dim=0):
        diff = ec_perturb.size(dim=0) - twy.size(dim=0)
        responsey = responsey[:len(ec_perturb)-diff]
    elif ec_perturb.size(dim=0) < twy.size(dim=0):
        diff = twy.size(dim=0) - ec_perturb.size(dim=0)
        diff_tensor = torch.zeros(diff)
        responsey = responsex[:len(ec_perturb)]
        responsey = torch.concat([responsey.cpu(),diff_tensor],-1)
    else:
        responsey = responsey[:len(ec_perturb)]
    
    ampl = 5e-6
    gradxp = twx + sign * ampl * responsex # plus for preemphasis, minus for perturbation
    gradyp = twy + sign * ampl * responsey 
    
    return gradxp, gradyp

def alpha_tau_loss(alphas, taus):
    '''
    loss function to measure suitability of [alphas, taus] for EC pre-emphasis
    - apply preemph based on current [alphas, taus] (preemph_or_perturb with sign +1)
    - apply EC perturbation with fixed [alphas_orig, taus_orig] (preemph_or_perturb with sign -1)
    - return MSE between original (ideal) and currently resulting waveform
    '''
    twx_pre, twy_pre = preemph_or_perturb(alphas, taus, twx, twy, FOV_sim, sign=+1)
    twx_pre_perturb, twy_pre_perturb = preemph_or_perturb(alphas_orig, taus_orig, twx_pre, twy_pre, FOV_sim, sign=-1)
    
    mse_x = torch.sum((twx - twx_pre_perturb)**2)
    mse_y = torch.sum((twy - twy_pre_perturb)**2)
    
    loss = mse_x + mse_y
    
    return loss

### Put original EC parameters here manually! ###
alphas_orig = torch.tensor([   1.,  ])
taus_orig   = torch.tensor([  50e-6, ])

alphas = alphas_orig.clone()
alphas.requires_grad = True
taus = taus_orig.clone()
taus.requires_grad = True

### Optimization loop. ###
NIter = 1000
lr = 1e-6
optimizable_params = [
    {'params': alphas, 'lr': lr*1e4},
    {'params': taus, 'lr': lr},
]

optimizer = torch.optim.Adam(optimizable_params, lr=lr, betas = [0.9, 0.999])

losses = torch.zeros((NIter,))
paramhist = torch.zeros((NIter,2))
for i in range(NIter):
    torch.autograd.set_detect_anomaly(False)
    optimizer.zero_grad()
    
    loss = alpha_tau_loss(alphas, taus)
    loss.backward()
    optimizer.step()
    print(f'iter {i}, loss: {loss.detach().item():.4g}, lr={lr}, alphas: {alphas.detach()}, taus: {taus.detach()}')
    losses[i] = loss.clone().detach().cpu()
    paramhist[i,0] = alphas.clone().detach().cpu()
    paramhist[i,1] = taus.clone().detach().cpu()
    
    if (i > 0) and (i % 1000) == 0:
        plt.figure(figsize=(9,16))
        plt.subplot(3,1,1)
        plt.semilogy(losses[:i],'.-')
        plt.title('loss')
        plt.subplot(3,1,2)
        plt.plot(paramhist[:i,0],'.-')
        plt.title('alpha')
        plt.subplot(3,1,3)
        plt.plot(paramhist[:i,1],'.-')
        plt.title('tau')
        plt.xlabel('iteration')
        plt.pause(1e-3)

# apply pre-emphasis based on the found parameters
alphas_pe = alphas.detach().clone()
taus_pe = taus.detach().clone()

twx_pre, twy_pre = preemph_or_perturb(alphas_pe, taus_pe, twx, twy, FOV_sim, sign=+1) # preemph with optimized params
twx_pre_perturb, twy_pre_perturb = preemph_or_perturb(alphas_orig, taus_orig, twx_pre, twy_pre, FOV_sim, sign=-1) # perturb with fixed params

seq_preemph = ec_tools_PE.EC_EPI_preemph(seq_full.clone(), FOV_sim, alphas=alphas_pe, taus=taus_pe) # calculate pre-emphasised gradient waveform
seq_preemph_perturb, srx_pe, sry_pe, twx_pe, twy_pe, _, _ = ec_tools.EC_perturbation_EPI(seq_preemph.clone(), FOV_sim, return_slew=True) # apply perturbation to pre-emphasised waveform and return preemph waveform (pre-perturbation!)

# %% Plot results.

plt.figure()
plt.subplot(3,2,1)
plt.plot(twx.cpu())
plt.plot(twx_pre.detach().cpu())
plt.plot(twx_pre_perturb.detach().cpu())
plt.legend(['original', 'preemph input', 'preemph realized'])

plt.subplot(3,2,3)
plt.plot((twx - twx_pre_perturb).detach().cpu())
plt.title('difference desired - realized')

plt.subplot(3,2,5)
plt.plot(srx.detach().cpu())
plt.plot(srx_pe.detach().cpu())
plt.legend(['slew original', 'slew preemph'])

plt.subplot(3,2,2)
plt.plot(twy.cpu())
plt.plot(twy_pre.detach().cpu())
plt.plot(twy_pre_perturb.detach().cpu())
plt.legend(['original', 'preemph input', 'preemph realized'])

plt.subplot(3,2,4)
plt.plot((twy - twy_pre_perturb).detach().cpu())
plt.title('difference desired - realized')

plt.subplot(3,2,6)
plt.plot(sry.detach().cpu())
plt.plot(sry_pe.detach().cpu())
plt.legend(['slew original', 'slew preemph'])

# %% Compare perturbed and unperturbed. 

graph_unperturbed = mr0.compute_graph(seq_full, data, max_state_count, min_state_mag)
graph_perturbed = mr0.compute_graph(seq_full_perturbed, data, max_state_count, min_state_mag)
graph_preemph_perturbed = mr0.compute_graph(seq_preemph_perturb, data, max_state_count, min_state_mag)

# simulate unperturbed
target_signal_full_unperturbed = mr0.execute_graph(graph_unperturbed, seq_full, target_data)

# simulate perturbed
target_signal_full_perturbed = mr0.execute_graph(graph_perturbed, seq_full_perturbed, target_data)

# simulate preemph -> perturbed
target_signal_full_preemph_perturbed = mr0.execute_graph(graph_preemph_perturbed, seq_preemph_perturb, target_data)

# reconstructions
target_reco_full_unperturbed = reconstruct_EPI_fft_naive(seq_full,target_signal_full_unperturbed,size, pf=partial_fourier)
target_reco_full_perturbed = reconstruct_EPI_fft_naive(seq_full_perturbed,target_signal_full_perturbed,size, pf=partial_fourier)
target_reco_full_preemph_perturbed = reconstruct_EPI_fft_naive(seq_preemph_perturb,target_signal_full_preemph_perturbed,size, pf=partial_fourier)

# %% Plot more results.

target = sos(target_reco_full_unperturbed)
reco_0 = sos(target_reco_full_perturbed)
reco_1 = sos(target_reco_full_preemph_perturbed)

plt.figure(3)
plt.subplot(331)
plt.plot(gmoms1[:,0].cpu().detach().numpy())
plt.plot(gmoms2[:,0].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

plt.subplot(332)
plt.plot(gmoms1[:,1].cpu().detach().numpy())
plt.plot(gmoms2[:,1].cpu().detach().numpy())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

kloc_perturb = seq_full_perturbed.get_kspace()
kloc_unperturbed = seq_full.get_kspace()

plt.subplot(333)
plt.plot(kloc_unperturbed[:,0].cpu())
plt.plot(kloc_perturb[:,0].cpu())
plt.plot(kloc_unperturbed[:,1].cpu())
plt.plot(kloc_perturb[:,1].cpu())
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.grid()

plt.subplot(334)
plt.imshow(torch.rot90(target.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),

plt.subplot(335)
plt.imshow(torch.rot90(reco_0.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.title('nominal -> perturbed')

plt.subplot(336)
plt.imshow(torch.rot90(reco_1.cpu()*1e4))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('viridis')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14),
plt.title('preemph -> perturbed')

plt.subplot(338)
plt.imshow(torch.abs(torch.rot90(target.cpu()*1e4)-torch.rot90(reco_0.cpu()*1e4)))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14)

plt.subplot(339)
plt.imshow(torch.abs(torch.rot90(target.cpu()*1e4)-torch.rot90(reco_1.cpu()*1e4)))
cbar = plt.colorbar(fraction=0.0453)
plt.set_cmap('gray')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('x1e-4',fontsize=16)
tx = cbar.ax.yaxis.get_offset_text()
tx.set_fontsize(14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14)