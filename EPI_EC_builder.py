from __future__ import annotations
import torch
import numpy as np
import MRzeroCore as mr0
import util

gamma_ = 42.5764 # MHz/T

def trap_grad(M, T, S, dt=1e-5):
    '''
    generate trapezoidal gradient waveform of TOTAL moment M (including ramps),
    duration T, slew rate S (will only be hit approximately due to rounding)
    used for prewinder, phase encoding and spoiler
    '''
    
    if 4*torch.abs(M)/S > T**2: # too large moment required in too short time with too small slew
        raise Exception("Error in trap_grad: gradient moment cannot be reached") 
    
    tramp = (T - torch.sqrt(T**2 - 4*torch.abs(M)/S)) / 2 # from geometry of symmetric trapezoid: which ramping duration is needed for given total duration, moment and slew?
    
    nramp = torch.ceil(tramp/dt) # ceil makes sure we always have slew rate <= S
    ntot = T//dt
    nflat = ntot - 2*nramp
    
    if nramp > 0:
        dg = M / (nramp*(nramp-1)+nflat*nramp) # again geometry: which ramping increment is needed to achieve desired total moment?
        G = nramp * dg # flat top amplitude
        rampup = torch.arange(0, nramp, 1) * dg
        flat = torch.ones((nflat.int().item(),)) * G
        rampdown = torch.flip(rampup,dims=(0,))
        wave = torch.cat([rampup, flat, rampdown])
    else: # no ramping needed
        G = M / ntot
        wave = torch.ones((ntot.int().item(),)) * G
        
    return wave

def trap_grad_flat(M, T, S, dt=1e-5):
    '''
    generate trapezoidal gradient waveform of FLATTOP moment M (excluding ramps),
    duration T, slew rate S (will only be hit approximately due to rounding)
    used for readout gradient
    return:
        - waveform
        - ramp moment (single-sided)
        - number of ramping samples
    '''
    n = T//dt
    G = M / n
    nramp = torch.ceil(G / S / dt**2)
    dg = G / nramp
    rampup = torch.arange(0, nramp, 1) * dg
    Mramp = dg*nramp*(nramp-1)/2 # =rampup.sum() , ramping moment
    flat = torch.ones((n.int().item(),)) * G
    wave = torch.cat([rampup, flat, torch.flip(rampup,(0,))])
    
    return wave, Mramp, nramp

class EPI2D_EC:
    """Stores all parameters needed to create a 2D EPI sequence."""

    def __init__(self, adc_count: int, rep_count: int = 1, shots: int = 1, TRfill: float = 0.0, R_accel: int = 1, partial_fourier: int = 1):
        
        self.adc_count = adc_count
        self.rep_count = rep_count # = segmentation factor (e.g. 2 means two RF excitations and interleaved EPI trains to fill kspace)
        self.event_count = 8520 # HARD-CODED: 200 (RF) + 500 (prew) + len(p2r_x) + len(RF2p_x) + len(newx) + 2
        self.part_count = 1
        self.shots = shots # FG: bit confusing terminology, I think Siemens might call this "averages" (repeating the same total k-space acquisition multiple times)
        self.R_accel = R_accel # parallel imaging undersampling factor
        self.partial_fourier = partial_fourier # partial Fourier factor, e.g. 5/8 means that for 128 matrix only 80 lines are acquired
        ros     = 1
        self.TE = 0
        self.TR = 0

        self.pulse_angles = torch.tensor(90*np.pi/180)
        self.pulse_phases = 0*torch.tensor(
            [util.phase_cycler(r, 117) for r in range(self.rep_count*self.part_count*self.shots)])

        self.gradm_xpre = -adc_count/2
        self.gradm_ypre = -adc_count*(partial_fourier-0.5) + torch.arange(rep_count) * R_accel # different y-prewinders for segments

        nlines = round(adc_count//rep_count//R_accel*partial_fourier) # number of k-space lines
        self.gradm_x = torch.ones((adc_count*ros+1,nlines)) * 1/ros # FG: inserted +1 here to have all ADC samples during flat top
        self.gradm_x[:,1::2] = -self.gradm_x[:,1::2]
        self.gradm_x[-1,:] = 0
        
        self.gradm_y = torch.zeros((adc_count*ros+1,nlines))
        self.gradm_y[-1,:-1] = rep_count * R_accel # segmentation -> larger blips
        
        self.gradm_spoiler_x = 0*1.5 * adc_count
        self.gradm_spoiler_y = 0*1.5 * adc_count
        
        self.TRfill = TRfill # = event time of last event in each repetition (seconds)


    def clone(self) -> EPI2D_EC:
        """Create a copy with cloned tensors."""
        clone = EPI2D_EC(self.adc_count, self.rep_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_x      = self.gradm_x.clone()
        clone.gradm_xpre   = self.gradm_xpre.clone()
        clone.gradm_y      = self.gradm_y.clone()
        clone.gradm_ypre   = self.gradm_ypre.clone()

        return clone

    def generate_sequence(self, oversampling = 1) -> mr0.Sequence:
        """Generate a EPI sequence based on the given parameters."""
        
        def insert_ramp_periods(original_tensor,nsteps):
            new_tensor = []
            
            prev_value = original_tensor[0].item()
            new_tensor.append(prev_value)
            
            for current_value in original_tensor[1:]:
                current_value = current_value.item()
                
                if current_value != prev_value:
                    if current_value > 0 and prev_value == 0:
                        ramp_values = torch.linspace(0, current_value, nsteps)[1:-1]
                    elif current_value == 0 and prev_value > 0:
                        ramp_values = torch.linspace(prev_value, 0, nsteps)[1:-1]
                    elif current_value < 0 and prev_value == 0:
                        ramp_values = torch.linspace(0, current_value, nsteps)[1:-1]
                    elif current_value == 0 and prev_value < 0:
                        ramp_values = torch.linspace(prev_value, 0, nsteps)[1:-1]
                    else:
                        ramp_values = []
                    
                    new_tensor.extend(ramp_values)
                    new_tensor.append(current_value)
                else:
                    new_tensor.append(current_value)
                
                prev_value = current_value
            
            return torch.tensor(new_tensor)       
        
        seq_all = []

        nsteps = 7 # CHANGE THIS TO MODIFY INITIAL SLEW RATE

        jj = 0
        for shot in range(self.shots): 
            seq = mr0.Sequence()
            
            for ii in torch.arange(self.rep_count):
               
                rep = seq.new_rep(self.event_count)
                
                rep.event_time[:] = 10e-6
                rep.event_time[-1] = self.TRfill # FG: deviate from 10us raster time grid here for longer TR gap without exploiding memory
                
                rep.pulse.angle = self.pulse_angles
                rep.pulse.phase = self.pulse_phases[jj]
                jj += 1
                rep.pulse.usage = mr0.PulseUsage.EXCIT
                                
                # Prepare RO period waveforms.
                tmpx = self.gradm_x.permute((1,0)).reshape((-1,))
                tmpy = self.gradm_y.permute((1,0)).reshape((-1,)) * (1/(nsteps-1)) # to account for larger triangle area with ramps
                
                newx = insert_ramp_periods( tmpx, nsteps)
                newy = insert_ramp_periods(-tmpy, nsteps)
                
                # pad to same length: (due to last blip missing, insert_ramp_periods returns a shorter waveform for y than for x)
                ndiff = len(newx) - len(newy)
                newy = torch.cat([newy, torch.zeros([ndiff,])])
                
                # define RF period
                RF_len = 200 # RF AT START
                rep.gradm[:RF_len,:] = 0
                
                # define prewinder - analytical attempts of taking ramping moments into account
                pw_len = 500
                px_tmp = (self.gradm_xpre     - nsteps/2*(rep.gradm[RF_len,0]+newx[0])) / (pw_len + nsteps - 2)
                py_tmp = (self.gradm_ypre[ii] - nsteps/2*(0+0)) / (pw_len + nsteps - 2) # 0+0 corresponds to initial and final grad value
                
                # deal with ramp before prewinder
                RF2p_x = torch.linspace(rep.gradm[RF_len,0], px_tmp, nsteps)[1:-1]
                RF2p_y = torch.linspace(rep.gradm[RF_len,1], py_tmp, nsteps)[1:-1]
                              
                rep.gradm[RF_len:RF_len+len(RF2p_x),0] = RF2p_x
                rep.gradm[RF_len:RF_len+len(RF2p_y),1] = RF2p_y
                 
                rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,0] = px_tmp
                rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,1] = py_tmp
                
                # deal with ramp after prewinder
                p2r_x = torch.linspace(rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,0][-1], newx[0], nsteps)[1:-1]
                p2r_y = torch.linspace(rep.gradm[RF_len+len(RF2p_x):RF_len+len(RF2p_x)+pw_len,1][-1], newy[0], nsteps)[1:-1]                   

                rep.gradm[RF_len+len(RF2p_x)+pw_len:RF_len+len(RF2p_x)+pw_len+len(p2r_x),0] = p2r_x
                rep.gradm[RF_len+len(RF2p_x)+pw_len:RF_len+len(RF2p_x)+pw_len+len(p2r_y),1] = p2r_y

                # readout
                RO_len = len(newx) # x and y have same no. elements
                rep.gradm[RF_len+len(RF2p_x)+pw_len+len(p2r_x):RF_len+len(RF2p_x)+pw_len+len(p2r_x)+RO_len,0] =  newx
                rep.gradm[RF_len+len(RF2p_x)+pw_len+len(p2r_x):RF_len+len(RF2p_x)+pw_len+len(p2r_x)+RO_len,1] = -newy                
                
                # spoilers
                if self.rep_count > 1 and False:
                    spoiler_len = 500
                    sx_tmp = (self.gradm_spoiler_x - nsteps/2*(0+0)) / (spoiler_len + nsteps - 2)
                    sy_tmp = (self.gradm_spoiler_y - nsteps/2*(0-newy[-1])) / (spoiler_len + nsteps - 2) # 0+0 corresponds to initial and final grad value
                    
                    # spoiler ramp-up
                    r2s_x = torch.linspace(0, sx_tmp, nsteps)[1:-1]
                    r2s_y = torch.linspace(-newy[-1], sy_tmp, nsteps)[1:-1] # try to directly ramp from last blip value to spoiler
                    
                    ix_r_end = RF_len+len(RF2p_x)+pw_len+len(p2r_x)+RO_len # end of readout train
                    
                    rep.gradm[ix_r_end:ix_r_end+len(r2s_x),0] = r2s_x
                    rep.gradm[ix_r_end:ix_r_end+len(r2s_y),1] = r2s_y
                    
                    # spoiler flat
                    rep.gradm[ix_r_end+len(r2s_x):ix_r_end+len(r2s_x)+spoiler_len,0] = sx_tmp
                    rep.gradm[ix_r_end+len(r2s_y):ix_r_end+len(r2s_y)+spoiler_len,1] = sy_tmp
                    
                    # spoiler ramp-down
                    s20_x = torch.linspace(sx_tmp, 0, nsteps)[1:-1]
                    s20_y = torch.linspace(sy_tmp, 0, nsteps)[1:-1]                   
                    
                    rep.gradm[ix_r_end+len(r2s_x)+spoiler_len:ix_r_end+len(r2s_x)+spoiler_len+len(s20_x),0] = s20_x
                    rep.gradm[ix_r_end+len(r2s_y)+spoiler_len:ix_r_end+len(r2s_y)+spoiler_len+len(s20_y),1] = s20_y
                
                # ADC
                indices_p = torch.nonzero(newx ==  1)
                indices_n = torch.nonzero(newx == -1)  
                                
                rep.adc_usage[RF_len+len(RF2p_x)+pw_len+len(p2r_x)+indices_p-0] = 1 # FG: odd/even line discrepancy?
                rep.adc_usage[RF_len+len(RF2p_x)+pw_len+len(p2r_x)+indices_n-1] = 1 # Use 0
                rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                
            seq_all.append(seq)
        
        return seq_all