#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:38:43 2025

@author: dw16
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import util
from PIL import Image
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()

gamma_ = 42.5764
dt   = 10e-6
FOV  = 250e-3
smax = 180

def moms2phys(moms, FOV):
    return moms / FOV / (gamma_*1e6)

#%% FIGURE 1: EPI Eddy Current Optimizations #%%

UC_data1 = torch.load('opt_results/EC_EPI_4paper_5em5_1m1.pth')
UC_data2 = torch.load('opt_results/EC_EPI_4paper_5em5_1m2_nolossh.pth')

UCtw3x = moms2phys(UC_data2.get('gmoms3'),FOV) / dt
UCsr3x = (UCtw3x[1:] - UCtw3x[:-1]) / dt
UCtw4x = moms2phys(UC_data2.get('gmoms4'),FOV) / dt
UCsr4x = (UCtw4x[1:] - UCtw4x[:-1]) / dt

gradc = UC_data1.get('gradc')
slewc = UC_data1.get('slewc')
klocs_target    = UC_data1.get('klocs_target')
klocs_perturbed = UC_data1.get('klocs_perturbed')
reco_target     = UC_data1.get('reco_target')
reco_perturb    = UC_data1.get('reco_perturb')

klocs_UC = UC_data2.get('klocs_opt')
reco_UC  = UC_data2.get('reco_opt')

device_1 = torch.device("cuda:1")
PEsol = torch.load('opt_results/pre_emph_solution.pt')
PEtw     = PEsol[0].to(device_1)
PEsr     = PEsol[1].to(device_1)
PEtarget = PEsol[2].to(device_1)
PEreco0  = PEsol[3].to(device_1)
PEreco1  = PEsol[4].to(device_1)
PEk1     = PEsol[5].to(device_1)
PEk2     = PEsol[6].to(device_1)

t_axis   = torch.linspace(0,dt*(torch.Tensor.size(UCtw3x,0)-1),torch.Tensor.size(UCtw3x,0))
idx1     = 0
idx2     = len(UCtw3x)

max_sig = torch.max(reco_target)

fig1 = plt.figure(1)
plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "dimgray"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.5

gs   = fig1.add_gridspec(2,4)
ax1  = fig1.add_subplot(gs[1,:3])
ax1.plot(1000*t_axis[idx1:idx2].cpu().detach(),PEsr.cpu().detach().numpy(),linewidth=2)
ax1.plot(1000*t_axis[idx1:idx2-1].cpu().detach(),UCsr3x[idx1:idx2,0].cpu().detach().numpy(),linewidth=2)
ax1.set_yticks([-360,-270,-180,-90,0,90,180,270,360])
ax1.set_yticklabels(['-360','-270','-180','-90','0','90','180','270','360'])
ax1.set_ylim(-360,360)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(90))
ax1.grid(True)
ax1.axhspan(180, 360, facecolor='lightgray', alpha=0.5)
ax1.axhspan(-360, -180, facecolor='lightgray', alpha=0.5)
plt.xticks(fontsize=14), plt.yticks(fontsize=14), plt.xlim(6,17.5)
plt.xlabel('Time [ms]',fontsize=16)
plt.ylabel('s [mT/m/ms]',fontsize=16)
plt.title('Gradient slew rate',fontsize=20,fontweight='bold',loc='left')

ax2  = fig1.add_subplot(gs[0,:3])
ax2.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*PEtw.cpu().detach(),linewidth=2)
ax2.plot(1000*t_axis[idx1:idx2].cpu().detach(),1000*UCtw3x[idx1:idx2,0].cpu().detach(),linewidth=2)
ax2.grid(True)
ax2.set_ylim(-15,15)
plt.xlabel('Time [ms]',fontsize=16)
plt.ylabel('g [mT/m]',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(6,17.5)
ax2.legend(
    ['Pre-emphasis', 'NISO'],
    fontsize=17,
    loc='upper left',
    bbox_to_anchor=(0.606, 1.185),  
    ncol=2
)
plt.title('Gradient amplitude',fontsize=20,fontweight='bold',loc='left')

default_blue = '#348ABD'
default_orange = '#E24A33'

ax3 = fig1.add_subplot(gs[1,3])
ax3.plot(klocs_perturbed[:,0].cpu().detach().numpy(), klocs_perturbed[:,1].cpu().detach().numpy(), '.k',markersize=20,markerfacecolor='none', markeredgewidth=1.5)
ax3.plot(klocs_UC[:,0].cpu().detach().numpy(), klocs_UC[:,1].cpu().detach().numpy(), 'x',markersize=10, color=default_blue,markeredgewidth=2)
ax3.tick_params(axis='both', labelsize=14)
ax3.set_xlim([-49,-39])
ax3.set_ylim([38,48])
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax3.grid(True, which='major', linestyle='-', linewidth=3)
ax3.set_ylabel('$k_y$ index',fontsize=16)
ax3.set_xlabel('$k_x$ index',fontsize=16)
ax3.set_aspect('equal')
ax3.set_xticks([-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39])
ax3.set_yticks([38,39,40,41,42,43,44,45,46,47,48])
ax3.set_xticklabels(['-49','','-47','','-45','','-43','','-41','','-39'])
ax3.set_yticklabels(['38','','40','','42','','44','','46','','48'])

ax8 = fig1.add_subplot(gs[0,3])
ax8.plot(PEk1[:,0].cpu().detach().numpy(), PEk1[:,1].cpu().detach().numpy(), '.k',markersize=20,markerfacecolor='none', markeredgewidth=1.5)
ax8.plot(PEk2[:,0].cpu().detach().numpy(), PEk2[:,1].cpu().detach().numpy(), 'x',markersize=10, color=default_orange,markeredgewidth=2)
ax8.tick_params(axis='both', labelsize=14)
ax8.set_xlim([-49,-39])
ax8.set_ylim([38,48])
ax8.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax8.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax8.grid(True, which='major', linestyle='-', linewidth=3)
ax8.set_ylabel('$k_y$ index',fontsize=16)
ax8.set_xlabel('$k_x$ index',fontsize=16)
ax8.set_aspect('equal')
ax8.set_xticks([-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39])
ax8.set_yticks([38,39,40,41,42,43,44,45,46,47,48])
ax8.set_xticklabels(['-49','','-47','','-45','','-43','','-41','','-39'])
ax8.set_yticklabels(['38','','40','','42','','44','','46','','48'])
legend_handles = [
    Line2D([0], [0], marker='o', color='k', linestyle='None', label='Uncorrected', markersize=20, markerfacecolor='none', markeredgewidth=1.5),
    Line2D([0], [0], marker='x', color=default_orange, linestyle='None', label='Pre-emphasis', markersize=10, markeredgewidth=2),
    Line2D([0], [0], marker='x', color=default_blue, linestyle='None', label='NISO', markersize=10, markeredgewidth=2)
]
ax8.legend(handles=legend_handles, fontsize=17, loc='upper left', bbox_to_anchor=(1.05, -0.1))

plt.subplots_adjust(left=0.08, right=0.85, top=0.93, bottom=0.1,hspace=0.3,wspace=0.4)

fig2 = plt.figure(2)

gs  = fig2.add_gridspec(1,4)

ax4 = fig2.add_subplot(gs[0,0])
plt.imshow(np.rot90(to_numpy(torch.abs((reco_target/max_sig)))))
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
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_target/max_sig))))-np.rot90(to_numpy(torch.abs((reco_perturb/max_sig)))))))
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
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_target/max_sig))))-np.rot90(to_numpy(torch.abs((PEreco1.to("cuda:0")/max_sig)))))))
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
plt.imshow(np.abs((np.rot90(to_numpy(torch.abs((reco_target/max_sig))))-np.rot90(to_numpy(torch.abs((reco_UC/max_sig)))))))
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

NRMSE_0  = util.NRMSE(reco_perturb, reco_target)
NRMSE_NISO = util.NRMSE(reco_UC, reco_target)
NRMSE_PE = util.NRMSE(PEreco1, PEtarget)

#%% Combining figures.

image1 = Image.open('FIG1A.png')
image2 = Image.open('FIG1B.png')
width = max(image1.width,image2.width)
image1 = image1.resize((width,image1.height))
image2 = image2.resize((width,image2.height))
stacked_image = Image.new('RGB',(width,image1.height+image2.height))
stacked_image.paste(image1,(0,0))
stacked_image.paste(image2,(0,image1.height))
stacked_image.save('FIG1.png')

#%% GMTF (Figure 2 inversion results are in GIRF_invert_GRAPPA.py) #%%

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

data = loadmat('GMTF_Terra_241209.mat')
GMTF = data['GMTF']

Hxx = GMTF['GMTF1xx'][0, 0] 
Hyy = GMTF['GMTF1yy'][0, 0] 
Hxy = GMTF['GMTF1xy'][0, 0] 
Hyx = GMTF['GMTF1yx'][0, 0]
ff  = np.transpose(GMTF['freq'][0, 0] )

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(1e-3*ff,np.abs(Hxx),color='#009E73')
plt.plot(1e-3*ff,np.abs(Hyy),color='#E69F00')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('f [kHz]',fontsize=20)
plt.ylabel('|H(f)|',fontsize=20)
plt.xlim([-50,50])
plt.grid(True)
plt.legend(['XX','YY'],fontsize=24,loc='upper right',ncol=2);

ax2 = plt.subplot(2,1,2)
plt.plot(1e-3*ff,np.abs(Hxy),color='#009E73')
plt.plot(1e-3*ff,np.abs(Hyx),color='#E69F00')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('f [kHz]',fontsize=20)
plt.ylabel('|H(f)|',fontsize=20)
plt.xlim([-50,50]), plt.ylim([0,0.008])
plt.grid(True)
plt.legend(['XY','YX'],fontsize=24,loc='upper right',ncol=2);
ax2.set_yticklabels(['0','','0.002','','0.004','','0.006','','0.008'])

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,hspace=0.3)

#%% FIGURE 4: SIMULATED RESULTS #%%

SS6_data1 = torch.load('opt_results/noGRAPPA_nramp6_PF1.pth')
SS6_data2 = torch.load('opt_results/noGRAPPA_nramp6_PF2.pth')
SS6t = SS6_data1.get('reco_target')
SS6p = SS6_data1.get('reco_perturb')
SS6o = SS6_data2.get('reco_opt')

SS7_data1 = torch.load('opt_results/noGRAPPA_nramp7_PF1.pth')
SS7_data2 = torch.load('opt_results/noGRAPPA_nramp7_PF2.pth')
SS7t = SS7_data1.get('reco_target')
SS7p = SS7_data1.get('reco_perturb')
SS7o = SS7_data2.get('reco_opt')

G27_data1 = torch.load('opt_results/GRAPPA2_nramp7_PF1.pth')
G27_data2 = torch.load('opt_results/GRAPPA2_nramp7_PF2.pth')
G27t = G27_data1.get('reco_target')
G27p = G27_data1.get('reco_perturb')
G27o = G27_data2.get('reco_opt')

G37_data1 = torch.load('opt_results/GRAPPA3_nramp71.pth')
G37_data2 = torch.load('opt_results/GRAPPA3_nramp72.pth')
G37t = G37_data1.get('reco_target')
G37p = G37_data1.get('reco_perturb')
G37o = G37_data2.get('reco_opt')

MS11_data1 = torch.load('opt_results/multishot_nramp111.pth')
MS11_data2 = torch.load('opt_results/multishot_nramp112.pth')
MS11t = MS11_data1.get('reco_target')
MS11p = MS11_data1.get('reco_perturb')
MS11o = MS11_data2.get('reco_opt')

fig1 = plt.figure(1)
gs   = fig1.add_gridspec(2,4)

max_sig = torch.max(SS7t)
ax2 = fig1.add_subplot(gs[0,0])
plot2 = plt.imshow(np.rot90(to_numpy(torch.abs((SS7p/max_sig)))))
plt.set_cmap('gray')
plt.clim(0,1)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = fig1.add_subplot(gs[1,0])
plot3 = plt.imshow(np.rot90(to_numpy(torch.abs((SS7o/max_sig)))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])

max_sig = torch.max(G27t)
ax2 = fig1.add_subplot(gs[0,1])
plot2 = plt.imshow(np.rot90(to_numpy(torch.abs((G27p/max_sig)))))
plt.set_cmap('gray')
plt.clim(0,1)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = fig1.add_subplot(gs[1,1])
plot3 = plt.imshow(np.rot90(to_numpy(torch.abs((G27o/max_sig)))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])

max_sig = torch.max(G37t)
ax2 = fig1.add_subplot(gs[0,2])
plot2 = plt.imshow(np.rot90(to_numpy(torch.abs((G37p/max_sig)))))
plt.set_cmap('gray')
plt.clim(0,1)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = fig1.add_subplot(gs[1,2])
plot3 = plt.imshow(np.rot90(to_numpy(torch.abs((G37o/max_sig)))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])

max_sig = torch.max(MS11t)
ax2 = fig1.add_subplot(gs[0,3])
plot2 = plt.imshow(np.rot90(to_numpy(torch.abs((MS11p/max_sig)))))
plt.set_cmap('gray')
plt.clim(0,1)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3 = fig1.add_subplot(gs[1,3])
plot3 = plt.imshow(np.rot90(to_numpy(torch.abs((MS11o/max_sig)))))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.set_cmap('gray')
plt.clim(0,1)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xticks([])
ax3.set_yticks([])

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.1)

#%% FIGURE 4: BAR CHART #%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

categories = ["no GRAPPA\n(shorter TE)", "no GRAPPA", "GRAPPA2", "GRAPPA3", "Multishot"]
values_nominal = [1.1149506, 0.958608, 0.98617, 0.987031, 0.835101]
values_pre_emphasis = [1.957344, 1.632981, 1.685629, 1.685011, 1.457267]
values_niso = [0.999999, 0.96221, 0.998357, 0.991811, 0.991704]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 8))
ax.bar(x - width, values_nominal, width, label='Nominal', color='#999999', alpha=0.7)
ax.bar(x, values_pre_emphasis, width, label='Pre-emphasis', color='#E24A33', alpha=0.7)
ax.bar(x + width, values_niso, width, label='NISO', color='#348ABD', alpha=0.7)
ax.grid(True)
ax.set_ylabel(r"$s$ / $s_{\max}$", fontsize=38)
ax.set_xticks(x)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=24,fontweight="bold")
ax.tick_params(axis='y', labelsize=22)
ax.legend(fontsize=22)
ax.axhline(y=1, color='black', linestyle='--', linewidth=3)
plt.tight_layout()
ax.set_ylim(0,2)
plt.show()
plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.25)
