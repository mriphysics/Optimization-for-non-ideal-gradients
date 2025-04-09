# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:17:23 2022

@author: fmglang
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
 
def EC_EPI_preemph(seq, FOV, alphas=[0,1,0], taus=[1e-3,1e-1,1], return_slew=False):
    
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
    gradxp = twx + ampl * responsex # Plus due to preemphasis.
    gradyp = twy + ampl * responsey # Plus due to preemphasis.
    
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