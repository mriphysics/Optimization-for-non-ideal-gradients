# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:17:23 2022

@author: fmglang, dwest
"""

import torch
import math
import util

gamma_ = 42.5764 # MHz/T

def moms2phys(moms, FOV=32e-3):
    return moms / FOV / (gamma_*1e6)

def phys2moms(moms, FOV=32e-3):
    return moms * FOV * (gamma_*1e6)

def truncate(number, digits) -> float:
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number.item()) / stepper

def EC_perturbation_EPI(seq, FOV, return_slew=False):
    
    seq_new = seq.clone()
    
    time_pick = seq_new[0].event_time
    gradt = time_pick[1]
    
    if seq[0].event_time[-1] > 1e-5:
        gtemp = []
        for rep in seq_new:
            gradm = rep.gradm[:-1,:]
            Tgap = rep.event_time[-1]
            Ngap = torch.floor(Tgap / gradt).int()
            gtemp.append(torch.cat((gradm, torch.zeros([Ngap,3]).to(gradm.device)), dim=0))
        gall = torch.cat(gtemp)
        gradx = gall[:,0]
        grady = gall[:,1]
    else:
        gradx = torch.cat([rep.gradm[:,0] for rep in seq_new]).to(torch.float32)
        grady = torch.cat([rep.gradm[:,1] for rep in seq_new]).to(torch.float32) 
        
    # Define gradient waveforms in physcial units.
    twx = moms2phys(gradx[:],FOV) / gradt
    srx = (twx[1:] - twx[:-1]) / gradt 
    srx = torch.cat((srx, torch.zeros([1]).to(util.get_device())))
    twy = moms2phys(grady[:],FOV) / gradt
    sry = (twy[1:] - twy[:-1]) / gradt 
    sry = torch.cat((sry, torch.zeros([1]).to(util.get_device())))

    maxtime = truncate(twx.shape[0]*gradt,10)
    timings = torch.arange(0, maxtime, gradt).to(util.get_device())

    alphas = [   1,    0, 0] # Paper parameters.
    taus   = [50e-6, 1e-1, 1] # Paper parameters.
    
    ec_perturb = torch.zeros(timings.shape).to(util.get_device())
    for alpha, tau in zip(alphas, taus): # Sum up all exponentials .
        ec_perturb += alpha*torch.exp(-timings/tau)
        
    # Use neural network convolution, this should be hopefully differentiable.
    responsex = torch.nn.functional.conv1d(srx.reshape([1,1,-1]), # [batch,channels=1,time]
                                         ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # Flip as conv in machine learning terms is actually crosscorrelation, add singleton for batch & channel.
                                         padding=len(ec_perturb) # Figured out by playing around, not fully sure if always correct.
                                         ).flatten() # Back to original shape.
    
    # Needed for numerical precision error.
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
                                         ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # Flip as conv in machine learning terms is actually crosscorrelation, add singleton for batch & channel.
                                         padding=len(ec_perturb) # Figured out by playing around, not fully sure if always correct.
                                         ).flatten() # Back to original shape.
        
    # Needed for numerical precision error.
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
    gradxp = twx - ampl * responsex # Minus due to Lenz's law.
    gradyp = twy - ampl * responsey # Minus due to Lenz's law.
    
    # Convert back to moments.
    nseg = len(seq_new)
    newmomsx = (phys2moms(gradxp,FOV)*gradt).reshape([nseg, -1]) 
    newmomsy = (phys2moms(gradyp,FOV)*gradt).reshape([nseg, -1])
    
    # Cut away everything in TR gap, since these waveform samples are not considered in the original sequence structure.
    newmomsx = newmomsx[:, :seq_new[0].gradm.shape[0]]
    newmomsy = newmomsy[:, :seq_new[0].gradm.shape[0]]
    
    # Assign these to sequence structure.
    for jj in range(nseg):
        gmomsp = seq_new[jj].gradm.clone()
        gmomsp[:,0] = newmomsx[jj,:] 
        gmomsp[:,1] = newmomsy[jj,:]
        seq_new[jj].gradm = gmomsp
    
    if return_slew:
        return seq_new, srx, sry, twx, twy, gradxp, gradyp
    else:
        return seq_new
   
def GIRF_perturbation(seq, FOV, GIRF, return_slew=False):
    
    seq_new = seq.clone()
    
    time_pick = seq_new[0].event_time
    gradt = time_pick[1]
    
    # Extract gradients from different directions.
    if seq[0].event_time[-1] > 1e-5: # If there is TR fill: put explicit zeros there (again defined on raster time for consistent GIRF application)
        gtemp = []
        for rep in seq_new:
            gradm = rep.gradm[:-1,:]
            Tgap = rep.event_time[-1]
            Ngap = torch.floor(Tgap / gradt).int()
            gtemp.append(torch.cat((gradm, torch.zeros([Ngap,3]).to(gradm.device)), dim=0))
        gall = torch.cat(gtemp)
        gradx = gall[:,0]
        grady = gall[:,1]
    else: # No TR fill.
        gradx = torch.cat([rep.gradm[:,0] for rep in seq_new]).to(torch.float32)
        grady = torch.cat([rep.gradm[:,1] for rep in seq_new]).to(torch.float32) 
        
    # Define gradient waveforms in physical units.
    twx = moms2phys(gradx[:],FOV) / gradt
    srx = (twx[1:] - twx[:-1]) / gradt 
    srx = torch.cat((srx, torch.zeros([1]).to(util.get_device())))
    twy = moms2phys(grady[:],FOV) / gradt
    sry = (twy[1:] - twy[:-1]) / gradt 
    sry = torch.cat((sry, torch.zeros([1]).to(util.get_device())))
    
    # Define GIRF terms - these are already on the right time axis (10us)
    hxx = torch.from_numpy(GIRF['hxx'])[:,0].to(torch.float32).to(util.get_device())
    hxy = torch.from_numpy(GIRF['hxy'])[:,0].to(torch.float32).to(util.get_device())
    hyx = torch.from_numpy(GIRF['hyx'])[:,0].to(torch.float32).to(util.get_device())
    hyy = torch.from_numpy(GIRF['hyy'])[:,0].to(torch.float32).to(util.get_device())

    # Perform convolutions.
    GIRFroll = -1 # roll=circshift applied to time-domain GIRF before convolving with waveform
    gxhxx = torch.nn.functional.conv1d(
            twx.unsqueeze(0).unsqueeze(0), hxx.roll(GIRFroll).flip(0).unsqueeze(0).unsqueeze(0), padding=len(hxx) // 2
            ).squeeze()
    gyhyy = torch.nn.functional.conv1d(
            twy.unsqueeze(0).unsqueeze(0), hyy.roll(GIRFroll).flip(0).unsqueeze(0).unsqueeze(0), padding=len(hyy) // 2
            ).squeeze()
    gxhxy = torch.nn.functional.conv1d(
            twx.unsqueeze(0).unsqueeze(0), hxy.roll(GIRFroll).flip(0).unsqueeze(0).unsqueeze(0), padding=len(hxy) // 2
            ).squeeze()
    gyhyx = torch.nn.functional.conv1d(
            twy.unsqueeze(0).unsqueeze(0), hyx.roll(GIRFroll).flip(0).unsqueeze(0).unsqueeze(0), padding=len(hyx) // 2
            ).squeeze()
    
    # Deal with inconsistent padding for even and odd sequence lengths.
    if hxx.shape[-1] % 2 == 0:
        gxhxx = gxhxx[:-1]
        gyhyy = gyhyy[:-1]
        gxhxy = gxhxy[:-1]
        gyhyx = gyhyx[:-1]       
    
    # Add terms to get final perturbed gradients.
    gradxp = gxhxx + gyhyx
    gradyp = gyhyy + gxhxy

    # Convert back to moments.
    nseg = len(seq_new)
    newmomsx = (phys2moms(gradxp,FOV)*gradt).reshape([nseg, -1]) 
    newmomsy = (phys2moms(gradyp,FOV)*gradt).reshape([nseg, -1])
    
    # Cut away everything in TR gap, since these waveform samples are not considered in the original sequence structure.
    newmomsx = newmomsx[:, :seq_new[0].gradm.shape[0]]
    newmomsy = newmomsy[:, :seq_new[0].gradm.shape[0]]
    
    # Assign these to sequence structure.
    for jj in range(nseg):
        gmomsp = seq_new[jj].gradm.clone()
        gmomsp[:,0] = newmomsx[jj,:] 
        gmomsp[:,1] = newmomsy[jj,:]
        seq_new[jj].gradm = gmomsp
    
    if return_slew:
        return seq_new, srx, sry, twx, twy, gradxp, gradyp
    else:
        return seq_new


def GIRF_invert(seq, FOV, GIRF, epsilon=1e-6, freq_limit = 50000, return_slew=True):
    """
    Inverts the convolution G(t) = h(t) * g(t) to solve for g(t).
    
    Args:
        G_t (torch.Tensor): Realized gradients (time-domain signal).
        h_t (torch.Tensor): GIRF (time-domain signal).
        epsilon (float): Regularization parameter to avoid division by near-zero values.
    
    Returns:
        torch.Tensor: Demanded gradients g(t).
    """

    seq_new = seq.clone()
    
    time_pick = seq_new[0].event_time
    gradt = time_pick[1]
    
    # Extract gradients from different directions.
    if seq[0].event_time[-1] > 1e-5: # If there is TR fill: put explicit zeros there (again defined on raster time for consistent GIRF application).
        gtemp = []
        for rep in seq_new:
            gradm = rep.gradm[:-1,:]
            Tgap = rep.event_time[-1]
            Ngap = torch.floor(Tgap / gradt).int()
            gtemp.append(torch.cat((gradm, torch.zeros([Ngap,3]).to(gradm.device)), dim=0))
        gall = torch.cat(gtemp)
        gradx = gall[:,0]
        grady = gall[:,1]
    else: # No TR fill.
        gradx = torch.cat([rep.gradm[:,0] for rep in seq_new]).to(torch.float32)
        grady = torch.cat([rep.gradm[:,1] for rep in seq_new]).to(torch.float32) 
        
    # Define gradient waveforms in physical units.
    twx = moms2phys(gradx[:],FOV) / gradt
    srx1 = (twx[1:] - twx[:-1]) / gradt 
    srx1 = torch.cat((srx1, torch.zeros([1]).to(util.get_device())))
    twy = moms2phys(grady[:],FOV) / gradt
    sry1 = (twy[1:] - twy[:-1]) / gradt 
    sry1 = torch.cat((sry1, torch.zeros([1]).to(util.get_device())))

    hxx = torch.from_numpy(GIRF['hxx'])[:,0].to(torch.float32).to(util.get_device())
    hyy = torch.from_numpy(GIRF['hyy'])[:,0].to(torch.float32).to(util.get_device())

    padding = (len(twx)-len(hxx))//2
    hxx = torch.nn.functional.pad(hxx,(padding,padding))
    hyy = torch.nn.functional.pad(hyy,(padding,padding))
    
    fwx_f = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(twx.unsqueeze(0))))
    fwy_f = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(twy.unsqueeze(0))))
    
    GIRFroll = 0
    Hxx = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(hxx.roll(GIRFroll).unsqueeze(0))))
    Hyy = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(hyy.roll(GIRFroll).unsqueeze(0))))

    Hxx = torch.where(Hxx.abs() < epsilon, epsilon + 0j, Hxx)  # Add small epsilon to small values?
    Hyy = torch.where(Hyy.abs() < epsilon, epsilon + 0j, Hyy)  # Add small epsilon to small values?
    
    # Self-term inversion only for now...
    fwx_f = fwx_f / Hxx 
    fwy_f = fwy_f / Hyy
    
    # Filtering...
    N = len(hxx)
    freq = torch.fft.fftshift(torch.fft.fftfreq(N, gradt))
    fwx_f = apply_tukey_bandpass_filter(fwx_f, freq, -freq_limit, freq_limit)
    fwy_f = apply_tukey_bandpass_filter(fwy_f, freq, -freq_limit, freq_limit)    
    
    gx = torch.real(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fwx_f)))).flatten()
    srx2 = (gx[1:] - gx[:-1]) / gradt 
    srx2 = torch.cat((srx2, torch.zeros([1]).to(util.get_device())))
    gy = torch.real(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fwy_f)))).flatten()
    sry2 = (gy[1:] - gy[:-1]) / gradt 
    sry2 = torch.cat((sry2, torch.zeros([1]).to(util.get_device())))

    # Convert back to moments.
    nseg = len(seq_new)
    newmomsx = (phys2moms(gx,FOV)*gradt).reshape([nseg, -1]) 
    newmomsy = (phys2moms(gy,FOV)*gradt).reshape([nseg, -1])
    
    # Cut away everything in TR gap, since these waveform samples are not considered in the original sequence structure.
    newmomsx = newmomsx[:, :seq_new[0].gradm.shape[0]]
    newmomsy = newmomsy[:, :seq_new[0].gradm.shape[0]]
    
    # Assign these to sequence structure.
    for jj in range(nseg):
        gmomsp = seq_new[jj].gradm.clone()
        gmomsp[:,0] = newmomsx[jj,:] 
        gmomsp[:,1] = newmomsy[jj,:]
        seq_new[jj].gradm = gmomsp

    if return_slew:
        return seq_new, twx, twy, gx, gy, srx1, sry1, srx2, sry2
    else:
        return seq_new

def apply_tukey_bandpass_filter(signal_f, freq, low_cut, high_cut, transition_width=0.3):
    """
    Apply a Tukey bandpass filter to a signal in the frequency domain.
 
    Args:
        signal_f (torch.Tensor): Signal in the frequency domain (complex tensor).
        freq (torch.Tensor): Frequency vector corresponding to the signal.
        low_cut (float): Lower cutoff frequency (Hz).
        high_cut (float): Upper cutoff frequency (Hz).
        transition_width (float): Width of the transition band as a fraction of the cutoff frequencies.
 
    Returns:
        torch.Tensor: Filtered signal in the frequency domain.
    """
    # Compute transition band in Hz.
    trans_band = transition_width * (high_cut - low_cut)
    # Create a mask initialized with zeros.
    filter_mask = torch.zeros_like(freq, dtype=torch.float32)
 
    # Passband
    passband = (freq >= low_cut + trans_band) & (freq <= high_cut - trans_band)
    filter_mask[passband] = 1.0
 
    # Transition bands (apply smooth Tukey window).
    low_trans_start = low_cut
    low_trans_end = low_cut + trans_band
    high_trans_start = high_cut - trans_band
    high_trans_end = high_cut
 
    # Low transition band (0 to 1).
    low_transition = (freq >= low_trans_start) & (freq < low_trans_end)
    if low_transition.any():
        x = (freq[low_transition] - low_trans_start) / trans_band  # Normalize
        filter_mask[low_transition] = 0.5 * (1 - torch.cos(torch.pi * x))
 
    # High transition band (1 to 0).
    high_transition = (freq > high_trans_start) & (freq <= high_trans_end)
    if high_transition.any():
        x = (freq[high_transition] - high_trans_start) / trans_band  # Normalize
        filter_mask[high_transition] = 0.5 * (1 + torch.cos(torch.pi * x))
 
    # Apply the filter in the frequency domain.
    filtered_signal = signal_f.to(util.get_device()) * filter_mask.to(util.get_device())
    
    return filtered_signal
