from __future__ import annotations
import torch
import MRzeroCore as mr0
from typing import Optional
from numpy import pi
import numpy as np
import time
import util
import torchkbnufft as tkbn



def reconstruct(signal: torch.Tensor,
                kspace: torch.Tensor,
                resolution: tuple[int, int, int] | float | None = None,
                FOV: tuple[float, float, float] | float | None = None,
                return_multicoil: bool = False,
                ) -> torch.Tensor:
    """Adjoint reconstruction of the signal, based on a provided kspace.

    Parameters
    ----------
    signal : torch.Tensor
        A complex tensor containing the signal,
        shape (sample_count, coil_count)
    kspace : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory
    resolution : (int, int, int) | float | None
        The resolution of the reconstruction. Can be either provided directly
        as tuple or set to None, in which case the resolution will be derived
        from the k-space (currently only for cartesian trajectories). A single
        float value will be used as factor for a derived resolution.
    FOV : (float, float, float) | float | None
        Because the adjoint reconstruction adapts to the k-space used
        for measurement, scaling gradients will not directly change the FOV of
        the reconstruction. All SimData phantoms have a normalized size of
        (1, 1, 1). Similar to the resolution, a value of None will
        automatically derive the FOV of the sequence based on the kspace. A
        float value can be used to scale this derived FOV.
    return_multicoil : bool
        Specifies if coils should be combined or returned separately.

    Returns
    -------
    torch.Tensor
        A complex tensor with the reconstructed image, the shape is given by
        the resolution.
    """
    res_scale = 1.0
    fov_scale = 1.0
    if isinstance(resolution, float):
        res_scale = resolution
        resolution = None
    if isinstance(FOV, float):
        fov_scale = FOV
        FOV = None

    # Atomatic detection of FOV - NOTE: only works for cartesian k-spaces
    # we assume that there is a sample at 0, 0 nad calculate the FOV
    # based on the distance on the nearest samples in x, y and z direction
    if FOV is None:
        def fov(t: torch.Tensor) -> float:
            t = t[t > 1e-3]
            return 1.0 if t.numel() == 0 else float(t.min())
        tmp = kspace[:, :3].abs()
        fov_x = fov_scale / fov(tmp[:, 0])
        fov_y = fov_scale / fov(tmp[:, 1])
        fov_z = fov_scale / fov(tmp[:, 2])
        FOV = (fov_x, fov_y, fov_z)
        print(f"Detected FOV: {FOV}")

    # Atomatic detection of resolution
    if resolution is None:
        def res(scale: float, fov: float, t: torch.Tensor) -> int:
            tmp = (scale * (fov * (t.max() - t.min()) + 1)).round()
            return max(int(tmp), 1)
        res_x = res(res_scale, FOV[0], kspace[:, 0])
        res_y = res(res_scale, FOV[1], kspace[:, 1])
        res_z = res(res_scale, FOV[2], kspace[:, 2])
        resolution = (res_x, res_y, res_z)
        print(f"Detected resolution: {resolution}")

    # Same grid as defined in SimData
    pos_x = torch.linspace(-0.5, 0.5, resolution[0] + 1)[:-1] * FOV[0]
    pos_y = torch.linspace(-0.5, 0.5, resolution[1] + 1)[:-1] * FOV[1]
    pos_z = torch.linspace(-0.5, 0.5, resolution[2] + 1)[:-1] * FOV[2]
    pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

    voxel_pos = util.set_device(torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1)).t()

    NCoils = signal.shape[1]
    # assert NCoils == 1, "reconstruct currently does not support multicoil"

    # (Samples, 4)
    kspace = util.set_device(kspace)
    # (Samples, 3) x (3, Voxels)
    phase = kspace[:, :3] @ voxel_pos
    # (Samples, Voxels): Rotation of all voxels at every event
    rot = torch.exp(2j*pi * phase)  # Matches definition of (i)DFT [FG: I think we agreed on FFT (NOT iFFT) for ksp->img]

    NCoils = signal.shape[1]

    if return_multicoil:
        return (signal.t() @ rot).view((NCoils, *resolution))
    elif NCoils == 1:
        return (signal.t() @ rot).view(resolution)
    else:
        return torch.sqrt(((torch.abs(signal.t() @ rot))**2).sum(0)).view(resolution)






def get_kmatrix(seq: mr0.Sequence | torch.tensor, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0,
                kspace_scaling: torch.Tensor | torch.Tensor | None = None,
                adc_usage: torch.Tensor | None = None,
                DREAM: bool = False
                ) -> torch.Tensor:
    '''
    reorder scanner signal according to kspace trajectory, works only for
    cartesian (under)sampling (where kspace grid points are hit exactly)
    
    FG: TODO: fix for odd matrix size (e.g. 129 breaks things currently)
    '''
    # import pdb; pdb.set_trace()
    
    # If seq input is Sequence, generating kspace with function get_kspace()
    # If seq input is the kspace, no further caluclations are necessary
    # Contrast has to be set to 0, no information about adc_usage is available
    
    if not torch.is_tensor(seq):
        kspace = seq.get_kspace()
    else:
        kspace = seq
    NCoils = signal.shape[1]
    
    if kspace_scaling is None:
        kmax = torch.round(torch.max(torch.abs(kspace[:,:3]),0).values)
        kspace_scaling = kmax*2/util.set_device(torch.tensor(resolution))
    
        kspace_scaling[kspace_scaling==0] = 1
    traj = kspace[:,:3]/kspace_scaling
    kindices = (traj + torch.floor(util.set_device(torch.tensor(resolution)) / 2)).round().to(int)
    if contrast and not torch.is_tensor(seq):
        mask = seq.get_contrast_mask(contrast)
        signal = signal[mask]
        kindices = kindices[mask]
        if DREAM:
            kindices[:,0] = kindices[:,0] - torch.min(kindices[:,0])
                
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(kindices[:,0], kindices[:,1], '.', ms=1)
    # plt.axis('equal')
    
    kmatrix = util.set_device(torch.zeros(*resolution, NCoils, dtype=torch.complex64))
    
    for jj in range(kindices.shape[0]): # I'm sure there must be a way of doing this without any loop...
        ix, iy, iz = kindices[jj,:]
        # if ix < 0 or ix >=32 or iy < 0 or iy >=32 or iz < 0 or iz >=1:
        #     print(f"Invalid indices at iteration {jj}: ix={ix}, iy={iy}, iz={iz}")
            #continue

        kmatrix[ix,iy,iz,:] = signal[jj,:]
        
        
    return kmatrix.permute([3,0,1,2]) # Dim (NCoils x resolution)

def reconstruct_cartesian_fft(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0
                ) -> torch.Tensor:
    '''
    do fft reco for Cartesian kspace grid
    '''
    
    ksp = get_kmatrix(seq, signal, resolution, contrast)
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first

def reconstruct_cartesian_fft_naive(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first

def reconstruct_EPI_fft_naive(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], nseg = 1, contrast = 0, pf = 1
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    handle multishot EPI by nseg parameter (used to reorder interleaved epi train acquisitions)
    
    TBD: handle partitions / 3D!%
    
    pf: partial fourier factor
    '''
    
    pf_start_ix = round((1 - pf) * resolution[0]) # index of first measured line in zero-filled k-matrix
    
    NRep = round(resolution[0] * pf) # #measured lines
    NCol = resolution[1] 
    NCoils = signal.shape[1]
    
    ksp = torch.zeros([NCoils, resolution[1], resolution[0], 1], dtype=signal.dtype, device=signal.device) # zero-filled k-space for partial fourier
    ksp[:,:,pf_start_ix:,:] = signal.reshape([-1,NRep,NCol,NCoils]).permute(3,2,1,0) # coils first,  compensate xy flip, [coil, read, blip, avg]
    ksp[:,:,(pf_start_ix+1)::2,:] = torch.flip(ksp[:,:,(pf_start_ix+1)::2,:], dims=(1,)) # EPI flip: flip all lines that were acquired with reverse readout gradient polarity. pf_start_ix: even, pf_start_ix+1: odd (reverse), pf_start_ix+2: even, and so on
    
    # take care of segmentation by reordering the interleaved shots into consistent k-space matrix
    idx = torch.arange(resolution[0]).reshape([nseg,-1]).permute([1,0]).flatten()
    ksp = ksp[:,:,idx,:]
    
    dim = (1,2)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft

def get_EPI_kspace(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], nseg = 1, R_accel = 1, contrast = 0, pf = 1,
                ) -> torch.Tensor:
    '''
    pf: partial fourier factor
    '''
    
    NRep = round(resolution[0] // R_accel * pf)
    NCol = resolution[1] 
    NCoils = signal.shape[1]
    
    pf_start_ix = round((1 - pf) * resolution[0]) # index of first measured line in zero-filled k-matrix
    
    ksp = torch.zeros([NCoils, resolution[1], resolution[0], 1], dtype=signal.dtype, device=signal.device) # zero-filled k-space for partial fourier
    ksp[:,:,pf_start_ix::R_accel,:] = signal.clone().reshape([-1,NRep,NCol,NCoils]).permute(3,2,1,0) # coils first,  compensate xy flip, [coil, read, blip, avg]
    ksp[:,:,(pf_start_ix+R_accel)::2*R_accel,:] = torch.flip(ksp[:,:,(pf_start_ix+R_accel)::2*R_accel,:], dims=(1,)) # EPI flip: flip all lines that were acquired with reverse readout gradient polarity. pf_start_ix: even, pf_start_ix+R_accel: odd (reverse), pf_start_ix+2*R_accel: even, and so on
    
    # take care of segmentation by reordering the interleaved shots into consistent k-space matrix
    # idx = torch.arange(NRep).reshape([nseg,-1]).permute([1,0]).flatten()
    # ksp = ksp[:,:,idx,:]
    
    # for parallel imaging: undersampling is represented by zeros in the k-space matrix
    # ksp_zf = torch.zeros([NCoils, resolution[1], resolution[0], 1], device=signal.device, dtype=signal.dtype)
    # ksp_zf[:,:,pf_start_ix::R_accel, :] = ksp
    
    dim = (1,2)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return ksp, reco_fft


def reconstruct_cartesian_fft_naive_ZF(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, nPF, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    tmp = torch.complex(torch.zeros([14,32,32,1]),torch.zeros([14,32,32,1])).to(util.get_device())
    tmp[:,nPF:,:,:] = ksp    
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(tmp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first

def reconstruct_cartesian_fft_naive_ZF_lowres(seq: mr0.Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], Ndummies, prew_moment, contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)-Ndummies # NO. DUMMY TRs
    NCol = torch.sum(seq[Ndummies].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    tmp = torch.zeros([14,10*resolution[0],resolution[1],resolution[2]], dtype=ksp.dtype).to(util.get_device()) # FG: factor 10 and number of coils hard-coded here
    tmp[:,10*resolution[0]-NCol:,(resolution[1]-NRep)//2:(resolution[1]-NRep)//2+NRep,:] = ksp  # put in center (phase-encode direction, less lines), and to edge (read direction, partial-Fourier) 
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(tmp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first

def remove_oversampling(signal: torch.Tensor, ax=0, oversampling=2):
    # central cropping of signal along axis ax by a given oversampling factor
    sz = signal.shape
    ix = np.array(range(len(sz)))
    signal = signal.permute((ax, *np.setdiff1d(ix, ax))) # put axis that is cropped to front
    lnew = sz[ax]//oversampling # new signal size along cropped axes
    cropix = np.arange(sz[ax]//2 - lnew//2, sz[ax]//2 + lnew//2)
    signal = signal[cropix,:]
    
    rix = np.argsort([ax, *np.setdiff1d(ix, ax)]) # back-permutation to original shape
    return signal.permute((*rix,))

def adaptive_combine(im, bs=None, modeSVD=True, modeSliBySli=False, donorm=0):
    '''
    Adaptive recon based on Walsh et al.
    Walsh DO, Gmitro AF, Marcellin MW.
    Adaptive reconstruction of phased array MR imagery. 
    Magn Reson Med. 2000 May;43(5):682-90.
    
    and
    
    Mark Griswold, David Walsh, Robin Heidemann, Axel Haase, Peter Jakob. 
    The Use of an Adaptive Reconstruction for Array Coil Sensitivity Mapping and Intensity Normalization, 
    Proceedings of the Tenth  Scientific Meeting of the International Society for Magnetic Resonance in Medicine pg 2410 (2002)
    
    implementation as in adaptiveCombine.m, which is part of recoVBVD
    
    -----------------------------------------------------------------
    
    im: Ncoils x Nx x Ny x Nz
    bs: optional (block size for sliding window svd)
    modeSVD: do some form of coil compression before getting weights (seems to be good according to recoVBVD)
    modeSliBySli: only 2D kernels
    donorm: empirical intensity normalization
    
    -----------------------------------------------------------------
    outputs:
        reco: Nx x Ny x Nz coil combined image
        weights: Ncoils x Nx x Ny x Nz Walsh weights
        norm: only if donorm=True, intensity normalization image
    
    
    ignores noise correlation for now!
    '''
    
    sz = im.shape
    nc = sz[0] # coils first!
    n = torch.tensor(sz[1:])
    
    weights = torch.zeros([nc, *n],dtype=torch.complex64,device=util.get_device())
    
    if bs is None: # automatic determination of block size
        bs = n.clone()
        bs[bs>7] = 7
        if n[2] > 1:
            bs[2] = 3 if n[2] > 3 else n[2]
            
    if modeSliBySli:
        bs[2] = 1
        
    if modeSVD:
        # intuitively: if more then 12 coils, use low rank approximation of coil images to determine coil weights
        nc_svd = int(min(min(12,max(9,np.floor(nc/2))),nc))
    else:
        nc_svd = nc
        
    cnt = 0
    maxcoil = 0
    if not modeSliBySli:
        if modeSVD:
            imcorr = im.reshape((nc,-1)) @ im.reshape((nc,-1)).conj().t()
            _, _, Vh = torch.linalg.svd(imcorr)
            V = Vh.conj().t()
            V = V[:,:nc_svd]
        else:
            V = torch.eye(nc, dtype=torch.complex64)
            _, maxcoil = torch.max(torch.sum(torch.abs(im), dim=(1,2,3)),0)
            
    # sliding window SVD for coil combination weights
    
    for z in range(n[2]): 
        if modeSliBySli:
            if modeSVD:
                tmp = im[:,:,:,z].reshape((nc,-1))
                _, _, Vh = torch.linalg.svd(tmp @ tmp.conj().t())
                V = Vh.conj().t()
                V = V[:,:nc_svd]
            else:
                V = torch.eye(nc, dtype=torch.complex64)
                _, maxcoil = torch.max(torch.sum(torch.abs(im[:,:,:,z]), dim=(1,2)),0)

        for y in range(n[1]): 
            for x in range(n[0]): 
                # current position of sliding window
                ix = torch.tensor([x,y,z])
                imin = torch.max(ix - torch.floor(bs.float()/2), torch.tensor([0.])).int()
                imax = torch.min(ix + torch.ceil(bs.float()/2) -1, (n-1.)).int() + 1
                
                # import pdb; pdb.set_trace()
                m1 = im[:, imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]].reshape((nc,-1))
                m1 = V.conj().t() @ m1
                m = m1 @ m1.conj().t() # signal covariance
                
                # d, v = torch.linalg.eigh(m) 
                # tmp = v[:,-1] # last eigenvalue is always largest
                d, v = torch.linalg.eig(m) 
                _, ind = torch.max(torch.abs(d),0)
                tmp = v[:,ind]
                tmp = V @ tmp # transform back to original coil space
                
                # Correct phase based on coil with max intensity
                tmp = tmp * torch.exp(-1j*torch.angle(tmp[maxcoil]));
                
                weights[:, x,y,z] = tmp.conj() / (tmp.conj().t() @ tmp)
                
                cnt += 1

    # now combine coils
    #weights = weights.reshape(sz).permute([0,2,1,3]) # permute is neccessary due to inverted row/column major flattening order between Matlab and python
    recon = torch.sum(weights * im, dim=0).reshape((*n, ))
    
    if donorm:
        norm = torch.sum(torch.abs(weights)**2, dim=0).reshape(n)
        recon = recon * norm
        return recon, weights, norm
    else:
        return recon, weights
    
def sos(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(x)**2,0))