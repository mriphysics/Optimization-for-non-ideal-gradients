from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import util
import MRzeroCore as mr0

class GRE3D:
    """Stores all parameters needed to create a 3D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int, part_count: int, R_accel: (int,int) = (1,1)):
        """Initialize parameters with default values."""
        self.adc_count = adc_count
        self.event_count = adc_count + 4
        self.rep_count = rep_count // R_accel[0]
        self.part_count = part_count // R_accel[1]
        self.shots = 1 # Number of shots
        self.R_accel = R_accel
        
        self.bwd = None
        self.dummy_angle = torch.tensor([5*np.pi/180])
        
        # Pulse
        self.pulse_angles = torch.full((self.rep_count*self.part_count, ), 7 * np.pi / 180)
        self.pulse_phases = torch.tensor(
            [util.phase_cycler(r, 84) for r in range(self.rep_count*self.part_count)])
        
        # Gradients
        self.gradm_rewinder = torch.full((rep_count*part_count, ), -adc_count/2-1)
        self.gradm_phase = torch.arange(-rep_count//2+np.mod(rep_count//2,R_accel[0]), rep_count//2, R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,R_accel[1])), (part_count+1)//2, R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_adc = torch.full((self.rep_count*self.part_count, ), 1.0)
        self.gradm_spoiler = torch.full((self.rep_count*self.part_count, ), 2.0 * self.adc_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        
        # Timings
        self.time_pulse = torch.tensor( 1.5e-3)
        self.TE = torch.tensor(0)
        self.TR = torch.tensor(0)
        self.time_spoiler = torch.tensor(2.0e-3)
        self.time_adc = torch.tensor(5*1e-5)
        
        self.relaxation_time = torch.tensor(1e-5)
        
    def linearEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D:
        self.gradm_phase = torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        
    def centricEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D:
        
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2  
        
        tmp = torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0])
        self.gradm_phase = tmp[permvec(self.rep_count)].repeat(self.part_count)
        tmp = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1])
        self.gradm_part = tmp[permvec(self.part_count)].repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
    
    
    def spiralEncoding(self, spiral_elongation = 0, alternating = False) -> GRE3D:
        """Create spiral encoding in y and z direction.""" 
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2   
        
        a, b = torch.meshgrid(self.gradm_phase[:self.rep_count],self.gradm_part[::self.rep_count])
        reordering_y = []
        reordering_z = []
        size_y = a.shape[0]
        size_z = a.shape[1]
        
        corr = 0
        if spiral_elongation == 0:
            Iy = 1 # Number of first encoding line in y direction
            Iz = 1 # Number of first encoding line in z directio
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation > 0:
            Iy = int(np.ceil(np.abs(spiral_elongation)*size_y)) + 1
            Iz = 1
            pos_lin = a.shape[0]//2+int(np.ceil(Iy/2))-1 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation < 0:
            Iy = 1
            Iz = int(np.ceil(np.abs(spiral_elongation)*size_z))
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2-int(np.ceil(Iz/2)) # Position in z direction
            for jj in range(0,Iz):
                #print(jj)
                reordering_y.append(a[pos_lin,pos_par+jj])
                reordering_z.append(b[pos_lin,pos_par+jj])
            pos_par += Iz
            corr = 1
        
        sign = 1
        Iy = Iy
        Iz = Iz+corr
                
        while (Iy < size_y) or (Iz < size_z) or len(reordering_y) < size_y*size_z:
            pos_lin = min(pos_lin,size_y-1)
            pos_par = min(pos_par,size_z-1)
            if Iz <= a.shape[1]:
                for ii in range(0,min(Iy,size_y)):
                    #print(ii)
                    reordering_y.append(a[pos_lin-sign*ii,pos_par])
                    reordering_z.append(b[pos_lin-sign*ii,pos_par])
            else:
                Iz = min(Iz,size_z)
            pos_lin -= sign*min(Iy,size_y-1)
            
            if Iy <= size_y:
                for jj in range(0,Iz):
                    #print(jj)
                    reordering_y.append(a[pos_lin,pos_par-sign*jj])
                    reordering_z.append(b[pos_lin,pos_par-sign*jj])
            else:
               Iy = min(Iy,size_y) 
            Iy += 1
            pos_par -= sign*min(Iz,size_z-1)
            Iz += 1
            # print(j)
            # print(i)
            sign *= -1

        num_perm = max(int(np.ceil(spiral_elongation*size_y))-1,int(np.ceil(-spiral_elongation*size_z)))+1
        perm = permvec(num_perm) 
        
        self.gradm_phase = torch.tensor(reordering_y)
        self.gradm_part = torch.tensor(reordering_z)
        
        if alternating:
            self.gradm_phase[:num_perm] = self.gradm_phase[perm]
            self.gradm_part[:num_perm] = self.gradm_part[perm]
        
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        cmap = plt.cm.get_cmap('rainbow')
        
        plt.plot(self.gradm_part,self.gradm_phase);
        plt.scatter(self.gradm_part,self.gradm_phase,c=torch.abs(self.pulse_angles.detach()*180/np.pi),s=200,cmap=cmap);plt.xlabel('z'); plt.ylabel('y');plt.title('Spiral elongation = ' + str(spiral_elongation))
        cbar = plt.colorbar()
        cbar.set_label('Flip Angle')
        for i in range(num_perm):
            plt.plot(self.gradm_part[i], self.gradm_phase[i],'.', c=cmap(i / num_perm))
        plt.show()
        
    def clone(self) -> GRE3D:
        """Create a copy with cloned tensors."""
        clone = GRE3D(self.adc_count, self.rep_count, self.part_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_rewinder = self.gradm_rewinder.clone()
        clone.gradm_phase = self.gradm_phase.clone()
        clone.gradm_part = self.gradm_part.clone()
        clone.gradm_adc = self.gradm_adc.clone()
        clone.gradm_spoiler = self.gradm_spoiler.clone()
        clone.gradm_spoiler_phase = self.gradm_spoiler_phase.clone()
        clone.gradm_spoiler_part = self.gradm_spoiler_part.clone()
        clone.relaxation_time = self.relaxation_time.clone()
        clone.time_spoiler = self.time_spoiler.clone()
        clone.time_adc = self.time_adc.clone()
        clone.bwd = self.bwd
        clone.time_pulse = self.time_pulse.clone()
        clone.TR = self.TR.clone()
        clone.TE = self.TE.clone()
        clone.dummy_angle = self.dummy_angle.clone()
 
        return clone

    def generate_sequence(self, oversampling = 1, mask_kspace = torch.ones(1), fatsat = 0, bwd: None | int = None, dummy = None) -> mr0.Sequence:
        """Generate a GRE sequence based on the given parameters."""
        if self.bwd is not None and bwd is None:
            bwd = self.bwd
            self.time_adc = torch.tensor(np.round(1/(bwd*self.rep_count),decimals=5))
            print('Set bandwidth to {setbwd}'.format(setbwd=bwd))
        
        mask_kspace = torch.ones(self.part_count*self.rep_count)
        seq_all = []
        
        phase_cycler = 0
        if dummy is not None:
            if self.__class__.__name__ == 'SNAPSHOT':
                self.pulse_phases = torch.tensor(
                    [util.phase_cycler(r, 84) for r in range(self.rep_count*self.part_count+dummy*self.shots)])
            elif self.__class__.__name__ == 'DREAM':
                self.pulse_phases = torch.tensor(
                    [util.phase_cycler(r, 84) for r in range(self.rep_count*self.part_count+dummy*self.shots)])    
            elif self.__class__.__name__ == 'BSSFP':
                self.pulse_phases =  torch.full((self.rep_count*self.part_count+dummy*self.shots, ), np.pi)
                self.pulse_phases[::2] = 0
            elif self.__class__.__name__ == 'GRE3D':
                self.pulse_phases =  torch.full((self.rep_count*self.part_count+dummy*self.shots, ), np.pi)
                self.pulse_phases[::2] = 0
                print("Set BSSFP phase cycler")
            else:
                raise Exception("Define, what happens to phase for dummy in GRE3D_builder")
            
        for shot in range(self.shots): 
            seq = mr0.Sequence()
            
            if fatsat:
                    rep = seq.new_rep(3)
                    
                    rep.pulse.angle = torch.tensor(0.0)
                    rep.pulse.phase = torch.tensor(0.0)
                    rep.pulse.usage = mr0.PulseUsage.FATSAT
                    
                    # Spoiler
                    rep.gradm[1, 0] = self.adc_count
                    rep.gradm[1, 1] = self.adc_count
                    rep.gradm[1, 2] = self.part_count
                    
                    # Event_time
                    rep.event_time[0] = 6.120 * 1e-3 # Fatsat pulse
                    rep.event_time[1] = 1.5*1e-3   # Spoiler
                    rep.event_time[2] = 1e-4   # Spoiler
                    
            for ii in torch.arange(shot,self.part_count*self.rep_count,self.shots):

                if torch.abs(mask_kspace[ii]) > 0.5:
                    # extra events: pulse + winder + rewinder
                    rep = seq.new_rep(self.event_count+(oversampling-1)*self.adc_count)
                    
                    rep.pulse.angle = self.pulse_angles[ii]
                    rep.pulse.phase = self.pulse_phases[phase_cycler+ii]
                    rep.pulse.usage = mr0.PulseUsage.EXCIT
        
                    rep.event_time[0] = self.time_pulse # Pulse 
                    rep.event_time[1] = 0.70*1e-3 + self.TE # Winder 
                    
                    rep.event_time[2:-2] =  self.time_adc/oversampling # Readout

                    rep.gradm[1, 0] = self.gradm_rewinder[ii]
                    rep.gradm[1, 1] = self.gradm_phase[ii]
                    rep.gradm[1, 2] = self.gradm_part[ii]
                    rep.gradm[2:-2, 0] = self.gradm_adc[ii]/oversampling
        
                    # Rewinder / Spoiler, centers readout in rep
                    rep.event_time[-1] = 0.060e-3 + self.TR #64x64
                    rep.event_time[-2] = self.time_spoiler  # Spoiler
        
                    rep.gradm[-2, 0] = self.gradm_spoiler[ii]
                    rep.gradm[-2, 1] = self.gradm_spoiler_phase[ii]
                    rep.gradm[-2, 2] = self.gradm_spoiler_part[ii]
        
                    rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                    
                    
                    if (dummy is not None and ii < self.shots):
                        rep_dummy = rep.clone()
                        rep_dummy.adc_usage[2:-2] = -1
                        for jj in range(dummy):
                            if len(self.dummy_angle) != dummy:
                                self.dummy_angle = torch.full((dummy,),self.pulse_angles[ii])
                                print("New Dummy Angles set!")
                            rep_dummy.pulse.angle = self.dummy_angle[jj]
                            rep_dummy.pulse.phase = self.pulse_phases[phase_cycler]
                            rep_dummy.adc_phase[:] = np.pi/2 - rep_dummy.pulse.phase
                            seq.append(rep_dummy.clone()) 
                            phase_cycler = phase_cycler + 1
                        rep.pulse.phase = self.pulse_phases[phase_cycler+ii]
                        rep.pulse.angle = self.pulse_angles[ii]
                        rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                            
                    rep.adc_usage[2:-2] = 1
                     
                    
            seq[-1].event_time[-1] = self.relaxation_time
            
            seq_all.append(seq)

        return seq_all        
        
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            
    @classmethod
    def load(cls, file_name) -> GRE3D:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
