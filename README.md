# MR sequence design to account for non-ideal gradient performance.

## Installation
- Start with an empty Python environment and run ```pip install -r requirements.txt```
- Download coil sensitivity files for the simulations from https://doi.org/10.5281/zenodo.15498143 and put them into the sub-directory ```/data```
  - data/B1minus_14ch_simu_3D_Gaussians.mat
  - data/B1minus_32ch_7T.mat

## Main Scripts
- ```EPI_EC_sim.py```: NISO optimization with an exponential eddy current model for a single-shot EPI sequence (no GRAPPA)
- ```EPI_EC_PE_sim.py```: Standard pre-emphasis with an exponential eddy current model for a single-shot EPI sequence, in which the eddy current parameters $\alpha$ and $\tau$ assumed for pre-emphasis are optimized
- ```EPI_GIRF_noGRAPPA.py```, ```EPI_GIRF_GRAPPA.py```, and ```EPI_GIRF_multishot.py```: NISO optimization with GMTF model for a single- or multishot EPI sequence with optional GRAPPA acceleration
- ```GIRF_invert_noGRAPPA.py``` and ```GIRF_invert_multishot_GRAPPA.py```: GMTF inversion (non-iterative, division in frequency domain) for a single- or multishot EPI sequence with optional GRAPPA acceleration

The general structure of the NISO scripts is:
1. Define sequence and optimization parameters
2. Load simulation data (numerical brain phantom with realistic tissue properties (PD, T1, T2,...))
3. Generate and simulate the target sequence without any perturbation due to gradient imperfections
4. Define a loss function that measures the difference between k-space locations with and without perturbation, including hardware constraints and optional simulation of intermediate images with perturbation
5. Run a backpropagation-based Adam optimization that adjusts the gradient waveform samples to compensate for the assumed hardware imperfection model
6. Optionally export the non-optimized and optimized sequences to Pulseq for real scanner measurements

## Utility Scripts and Functions
- ```EPI_EC_builder.py```: Define a 2D EPI sequence with options for single-shot, multishot, GRAPPA and partial Fourier acceleration, compatible with the MrZeroCore simulation framework
- ```ec_tools.py``` and ```ec_tools_PE.py```: Functions for applying gradient imperfection forward models (exponential eddy current and GMTF model), as well as corresponding pre-emphasis and inversion functions
- ```reconstruction.py```, ```grappa.py```: Functions for image reconstruction (FFT, partial Fourier, GRAPPA)
- ```util.py```, ```sensitivity_tools.py```: various helper functions
- ```pulseq_exporter.py```: functions for tranlating MRzero sequence structures (as used for optimization and simulation) to Pulseq for real scanner measurements

## Results
- sub-directory ```opt_results```: optimization results for the scenarios shown in the manuscript, to be loaded and plotted in ```TN_plots.py```, ```GIRF_invert_noGRAPPA.py``` and ```GIRF_invert_multishot_GRAPPA.py``` 
- sub-directory ```klocs4analysis```: k-space trajectories that can be loaded and plotted with ```vectorfigure_singleshot.m``` and ```vectorfigure_multishot.m``` 

## References
- MRzeroCore documentation: https://mrzero-core.readthedocs.io
- Pulseq: https://pulseq.github.io/
- PyPulseq: https://github.com/imr-framework/pypulseq
