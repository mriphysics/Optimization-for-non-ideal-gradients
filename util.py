"""This module contains helper functions only."""

from __future__ import annotations
from typing import Tuple
import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from torch.nn.functional import interpolate
from skimage.metrics import structural_similarity as ssim
import MRzeroCore as mr0
from MRzeroCore import VoxelGridPhantom, SimData
from MRzeroCore.phantom.voxel_grid_phantom import sinc, sigmoid, identity, recover
from typing import Literal, Optional, Dict

use_gpu = True
gpu_dev = 0

class VoxelGridPhantomMod(VoxelGridPhantom):

    def __init__(
        self,
        PD: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T2dash: torch.Tensor,
        D: torch.Tensor,
        B0: torch.Tensor,
        B1: torch.Tensor,
        coil_sens: torch.Tensor,
        base_fov: torch.Tensor,
        rel_fov: torch.Tensor,
        normalize_B0_B1: bool = True
    ) -> None:
        """Set the phantom attributes to the provided parameters.

        This function does no cloning nor contain any other funcionality. You
        probably want to use :meth:`brainweb` to load a phantom instead.
        """
        self.PD = PD
        self.T1 = T1
        self.T2 = T2
        self.T2dash = T2dash
        self.D = D
        self.B0 = B0
        self.B1 = B1
        self.coil_sens = coil_sens
        self.base_fov = base_fov
        self.rel_fov = rel_fov   
        
        if normalize_B0_B1:
            mask = PD > 1e-6
            self.B1 /= self.B1[mask.unsqueeze(0)].mean()
            self.B0 -= self.B0[mask].mean()

    def build(self, PD_threshold: float = 1e-6,
              use_SI_FoV: bool = False, voxel_shape="sinc") -> SimData:
        """Build a :class:`SimData` instance for simulation.

        Arguments
        ---------
        PD_threshold: float
            All voxels with a proton density below this value are ignored.
        use_SI_FoV: bool
            If set to ``True``, the built :class:`SimData` will use its
            physical size in meters. If set to ``False``, the ``rel_fov`` is
            used, which means a sequence FOV of 1 is assumed.
        """
        mask = self.PD > PD_threshold

        fov = (self.base_fov * self.rel_fov) if use_SI_FoV else (self.rel_fov)
        shape = torch.tensor(mask.shape)
        # pos_x, pos_y, pos_z = torch.meshgrid(
        #     fov[0] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[0]), device=self.PD.device)),
        #     fov[1] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[1]), device=self.PD.device)),
        #     fov[2] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[2]), device=self.PD.device)),
        # )

        # voxel_pos = torch.stack([
        #     pos_x[mask].flatten(),
        #     pos_y[mask].flatten(),
        #     pos_z[mask].flatten()
        # ], dim=1)
        
        # Normalized voxel positions, together with the gradient definition
        # a linear cartesian readout will match the definition of a DFT
        pos_x = torch.linspace(-0.5, 0.5, int(shape[0]) + 1)[:-1]
        pos_y = torch.linspace(-0.5, 0.5, int(shape[1]) + 1)[:-1]
        pos_z = torch.linspace(-0.5, 0.5, int(shape[2]) + 1)[:-1]
        pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

        voxel_pos = torch.stack([
            pos_x[mask].flatten(),
            pos_y[mask].flatten(),
            pos_z[mask].flatten()
        ], dim=1)

        if voxel_shape == "box":
            dephasing_func = lambda t, n: sinc(t, 0.5 / n)
        elif voxel_shape == "sinc":
            dephasing_func = lambda t, n: sigmoid(t, n)
        elif voxel_shape == "point":
            dephasing_func = lambda t, _: identity(t)
        else:
            raise ValueError(f"Unsupported voxel shape '{voxel_shape}'")

        return SimData(
            self.PD[mask],
            self.T1[mask],
            self.T2[mask],
            self.T2dash[mask],
            self.D[mask],
            self.B0[mask],
            self.B1[:, mask],
            self.coil_sens[:, mask],
            self.base_fov * self.rel_fov,  # Always SI, only used for diffusion
            voxel_pos,
            torch.tensor(shape, device=self.PD.device) / 2,
            dephasing_func,
            recover_func=lambda d: recover(mask, self.base_fov, self.rel_fov, d)
        )

    @classmethod
    def brainweb(cls, file_name: str) -> VoxelGridPhantomMod:
        """Load a phantom from data produced by `generate_maps.py`."""
        with np.load(file_name) as data:
            T1 = torch.tensor(data['T1_map'])
            T2 = torch.tensor(data['T2_map'])
            T2dash = torch.tensor(data['T2dash_map'])
            PD = torch.tensor(data['PD_map'])
            D = torch.tensor(data['D_map'])

        # Generate a somewhat plausible B0 and B1 map.
        # Visually fitted to look similar to the numerical_brain_cropped
        x_pos, y_pos, z_pos = torch.meshgrid(
            torch.linspace(-1, 1, PD.shape[0]),
            torch.linspace(-1, 1, PD.shape[1]),
            torch.linspace(-1, 1, PD.shape[2]),
            indexing="ij"
        )
        B1 = torch.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
        dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
        B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)
        # Normalize such that the weighted average is 0 or 1
        # weight = PD / PD.sum()
        # B0 -= (B0 * weight).sum()
        # B1 /= (B1 * weight).sum()
        
        # FG: old normalization convention
        mask = PD > 1e-6
        B1 /= B1[mask].mean()
        B0 -= B0[mask].mean()

        return cls(
            PD, T1, T2, T2dash, D, B0, B1[None, ...],
            coil_sens=torch.ones(1, *PD.shape),
            base_fov=torch.tensor([0.192, 0.192, 0.192]),
            rel_fov=torch.ones(3)
        )
    
    def slices(self, slices: list[int]) -> VoxelGridPhantomMod:
        """Generate a copy that only contains the selected slice(s).

        Parameters
        ----------
        slice: int or tuple
            The selected slice(s)

        Returns
        -------
        SimData
            A new instance containing the selected slice(s).
        """
        assert 0 <= any([slices]) < self.PD.shape[2]

        # fov = self.rel_fov.clone()
        # fov[2] *= len(slices) / self.PD.shape[2]

        def select(tensor: torch.Tensor):
            return tensor[..., slices].view(
                *list(self.PD.shape[:2]), len(slices)
            )

        return VoxelGridPhantomMod(
            select(self.PD),
            select(self.T1),
            select(self.T2),
            select(self.T2dash),
            select(self.D),
            select(self.B0),
            select(self.B1).unsqueeze(0),
            select(self.coil_sens).unsqueeze(0),
            self.base_fov.clone(),
            self.rel_fov.clone(),
            False
        )
    
    def interpolate(self, x: int, y: int, z: int) -> VoxelGridPhantomMod:
        """Return a resized copy of this :class:`SimData` instance.

        This uses torch.nn.functional.interpolate in 'area' mode, which is not
        very good: Assumes pixels are squares -> has strong aliasing.

        Use :meth:`resample_fft` instead.

        Parameters
        ----------
        x : int
            The new resolution along the 1st dimension
        y : int
            The new resolution along the 2nd dimension
        z : int
            The new resolution along the 3rd dimension
        mode : str
            Algorithm used for upsampling (via torch.nn.functional.interpolate)

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing resized tensors.
        """
        def resample(tensor: torch.Tensor) -> torch.Tensor:
            # Introduce additional dimensions: mini-batch and channels
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='area'
            )[0, 0, ...]

        def resample_multicoil(tensor: torch.Tensor) -> torch.Tensor:
            coils = tensor.shape[0]
            output = torch.zeros(coils, x, y, z)
            for i in range(coils):
                output[i, ...] = resample(tensor[i, ...])
            return output

        mask = self.PD > 1e-6

        return VoxelGridPhantomMod(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.D),
            resample(self.B0*mask),
            resample_multicoil(self.B1*mask),
            resample_multicoil(self.coil_sens),
            self.base_fov.clone(),
            self.rel_fov.clone(),
            False
        )


def get_device() -> torch.device:
    """Return the device as given by ``util.use_gpu`` and ``util.gpu_dev``."""
    if use_gpu:
        return torch.device(f"cuda:{gpu_dev}")
    else:
        return torch.device("cpu")


def set_device(x: torch.Tensor) -> torch.Tensor:
    """Set the device of the passed tensor as given by :func:`get_deivce`."""
    if use_gpu:
        return x.cuda(gpu_dev)
    else:
        return x.cpu()


def phase_cycler(pulse: int, dphi: float = 137.50776405) -> float:
    """Generate a phase for cycling through phases in a sequence.

    The default value of 360° / Golden Ratio seems to work well, better than
    angles like 117° which produces very similar phases for every 3rd value.

    Parameters
    ----------
        pulse : int
            pulse number for which the phase is calculated
        dphi : float
            phase step size in degrees

    Returns
    -------
        Phase of the given pulse
    """
    return float(np.fmod(0.5 * dphi * (pulse**2+pulse+2), 360) * np.pi / 180)


def current_fig_as_img(dpi: float = 180) -> np.ndarray:
    """Return the current matplotlib figure as image.

    Parameters
    ----------
    dpi : float
        The resolution of the returned image

    Returns
    -------
    np.ndarray
        The current matplotlib figure converted to a 8 bit rgb image.
    """
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def to_full(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert a sparse to a full tensor by filling indices given by mask.

    Parameters
    ----------
    sparse : torch.Tensor)
        Sparse tensor containing the data.
    mask : torch.Tensor)
        Mask indicating the indices of the elements in ``sparse``

    Raises
    ------
    ValueError
        If ``mask`` requires more or less elements than ``sparse`` contains.

    Returns
    -------
    torch.Tensor
        The full tensor that has the same shape as ``mask`` and contains the
        data of ``sparse``.
    """
    if mask.count_nonzero() != sparse.shape[-1]:
        raise ValueError(
            f"mask requires {mask.count_nonzero()} elements, "
            f"but sparse contains {sparse.shape[-1]}."
        )
    # coil_sens and B1 have an additional dimension for coils
    if sparse.squeeze().dim() > 1:
        full = torch.zeros(sparse.shape[:-1] + mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[..., mask] = sparse
    else:
        full = torch.zeros(mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[mask] = sparse
    return full


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()


def to_torch(x: np.ndarray) -> torch.Tensor:
    """Convert a numpy ndarray to a torch tensor."""
    return torch.tensor(x, dtype=torch.float)

def plot3D(x: torch.Tensor,figsize=(16,8)) -> None:
    """Plot absolute image of a 3D tensor (x,y,z)
    or 4D tensor (coil,x,y,z)."""
    if x.ndim == 4:
        x = torch.sum(torch.abs(x),0)
    plt.figure(figsize=figsize)
    if type(x).__module__ == np.__name__:
        plt.imshow(np.flip(x,1).transpose(1,2,0).reshape(x.shape[1],x.shape[0]*x.shape[2]))
    else:
        plt.imshow(np.flip(to_numpy(x),1).transpose(1,2,0).reshape(x.shape[1],x.shape[0]*x.shape[2]))
    plt.colorbar()

def complex_loss(input, target):
    eps = 1e-10
    real_input = input[...,0]
    imag_input = input[...,1]
    real_target = target[...,0]
    imag_target = target[...,1]        
    mag_input = torch.sqrt(real_input**2+imag_input**2+eps)
    mag_target = torch.sqrt(real_target**2+imag_target**2+eps)
    angle_loss = torch.mean(torch.abs(real_input*imag_target-imag_input*real_target)/(mag_target+eps))
    angle_loss[torch.isnan(angle_loss)] = 0
    mag_loss = torch.nn.L1Loss()(mag_input,mag_target)
    return mag_loss + angle_loss


def SSIM(a: torch.Tensor, b: torch.Tensor,
         window_size: float = 4.0) -> torch.Tensor:
    """Calculate the structural similarity of two 2D tensors.

    Structural similarity is a metric that tries to estimate how similar two
    images look for humans. The calculated value is per-pixel and describes how
    different or similar that particular pixel looks. While doing so it takes
    the neighbourhood into account, as given by the ``window_size``.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    window_size : float
        The window size used when comparing ``a`` and ``b``

    Returns
    -------
    torch.Tensor
        A tensor with the same shape as ``a`` and ``b``, containing for every
        pixel a value between 0 (no similarity) to 1 (identical).
    """
    assert a.shape == b.shape and a.dim() == 2

    x, y = torch.meshgrid([torch.arange(a.shape[0]), torch.arange(a.shape[1])])
    norm = 1 / (2*np.pi*np.sqrt(window_size))

    def gauss(x0: float, y0: float):
        return norm * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2*window_size))

    ssim = torch.zeros_like(a)
    c1 = 1e-4
    c2 = 9e-4

    for x0 in range(a.shape[0]):
        for y0 in range(a.shape[1]):
            window = gauss(x0, y0)
            a_w = a * window
            b_w = b * window

            a_mean = a_w.mean()
            b_mean = b_w.mean()
            a_diff = a_w - a_mean
            b_diff = b_w - b_mean

            ssim[x0, y0] = (
                (
                    (2*a_mean*b_mean + c1)
                    * (2*(a_diff*b_diff).mean() + c2)
                ) / (
                    (a_mean**2 + b_mean**2 + c1)
                    * ((a_diff**2).mean() + (b_diff**2).mean() + c2)
                )
            )

    return ssim


def load_optimizer(optimizer: torch.optim.Optimizer,
                   path: torch.Tensor,
                   NN: torch.nn.Module | None = None
                   ) -> tuple[torch.optim.Optimizer, torch.Tensor,
                              torch.Tensor, torch.Tensor,
                              torch.nn.Module | None]:
    """Load state of optimizer for retraining/restarts

    Parameters
    ----------
    optimizer : torch.optim
        A optimizer
    path : torch.Tensor
        A tensor with the path to the file which sould be loaded

    Returns
    -------
    optimizer : torch.optim
        Optimizer with loaded parameters.
    loss_history : torch.Tensor
        Old loss_history.
    params_target : torch.Tensor
        Sequence parameters for target.
    target_reco : torch.Tensor
        Target reconstruction
    """
    checkin = torch.load(path)
    optimizer.load_state_dict(checkin['optimizer'])
    optimizer.param_groups = checkin['optimizer_params']
    if NN:
        NN.load_state_dict(checkin['NN'])

    return (
        optimizer,
        checkin['loss_history'],
        checkin['params_target'],
        checkin['target_reco'],
        NN
    )


def L1(a: torch.Tensor, b: torch.Tensor,
       absolut: bool = False) -> torch.Tensor:
    """Calculate the L1 norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    absolut : bool
        The flag ``absolut`` indicates if the abs() of ``a`` and ``b`` size is
        taken before calculating the L1 norm.

    Returns
    -------
    torch.Tensor
        A tensor with the L1 norm.
    """
    assert a.shape == b.shape

    if absolut:
        norm = torch.sum(torch.abs(torch.abs(a)-torch.abs(b)))
    else:
        norm = torch.sum(torch.abs(a-b))
    return norm


def MSR(a: torch.Tensor, b: torch.Tensor,
        root: bool = False, weighting: torch.Tensor | float = 1,
        norm: bool = False) -> torch.Tensor:
    """Calculate the (R)MSR norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    root : torch.bool
        The flag ``root indicates if the square root of the RMS is used.
    weighting : torch.Tensor
        Give a weighting on a and b
    norm : torch.bool
        Gives the normalized MSR on b

    Returns
    -------
    torch.Tensor
        A tensor with the (R)MSE norm.
    """
    assert a.shape == b.shape

    tmp = torch.abs(a*weighting - b*weighting)
    tmp = tmp**2

    tmp = torch.sum(tmp)
    if root:
        tmp = torch.sqrt(tmp)
        #tmp = tmp
        
    if norm:
        tmp /= torch.sum(torch.abs(b*weighting))
        #tmp = tmp    
        
    return tmp

def NRMSE(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    diff = a - b
    squared_diff = diff ** 2

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean(squared_diff))

    # Calculate the range of values in tensor a
    denom = torch.sqrt(torch.mean(b**2))

    # Calculate NRMSE
    nrmse = (rmse / denom) * 100
    
    return nrmse

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = 1.0  # assuming pixel values are normalized between 0 and 1
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_val.item()

def calculate_ssim(img1, img2):
    # Convert PyTorch tensors to numpy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Calculate SSIM
    ssim_val, _ = ssim(img1_np[:,:,0], img2_np[:,:,0], multichannel=True, full=True, data_range = np.max([img1_np,img2_np])-np.min([img1_np,img2_np]))
    return ssim_val

def plot_kspace_trajectory(seq: mr0.Sequence,
                           figsize: tuple[float, float] = (5, 5),
                           plotting_dims: str = 'xy',
                           plot_timeline: bool = True,
                           new_figure: bool = True) -> None:
    """Plot the kspace trajectory produced by self.

    Parameters
    ----------
    kspace : list[Tensor]
        The kspace as produced by ``Sequence.get_full_kspace()``
    figsize : (float, float), optional
        The size of the plotted matplotlib figure.
    plotting_dims : string, optional
        String defining what is plotted on the x and y axis ('xy' 'zy' ...)
    plot_timeline : bool, optional
        Plot a second subfigure with the gradient components per-event.
    """
    assert len(plotting_dims) == 2
    assert plotting_dims[0] in ['x', 'y', 'z']
    assert plotting_dims[1] in ['x', 'y', 'z']
    dim_map = {'x': 0, 'y': 1, 'z': 2}

    # TODO: We could (optionally) plot which contrast a sample belongs to,
    # currently we only plot if it is measured or not

    kspace = seq.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq]

    cmap = plt.get_cmap('rainbow')
    if new_figure:
        plt.figure(figsize=figsize)
    if plot_timeline:
        plt.subplot(211)
    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        kx = to_numpy(rep_traj[:, dim_map[plotting_dims[0]]])
        ky = to_numpy(rep_traj[:, dim_map[plotting_dims[1]]])
        measured = to_numpy(mask)

        plt.plot(kx, ky, c=cmap(i / len(kspace)))
        plt.plot(kx[measured], ky[measured], 'r.')
        plt.plot(kx[~measured], ky[~measured], 'k.')
    plt.xlabel(f"$k_{plotting_dims[0]}$")
    plt.ylabel(f"$k_{plotting_dims[1]}$")
    plt.grid()

    if plot_timeline:
        plt.subplot(212)
        event = 0
        for i, rep_traj in enumerate(kspace):
            x = np.arange(event, event + rep_traj.shape[0], 1)
            event += rep_traj.shape[0]
            rep_traj = to_numpy(rep_traj)

            if i == 0:
                plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
            else:
                plt.plot(x, rep_traj[:, 0], c='r', label="_")
                plt.plot(x, rep_traj[:, 1], c='g', label="_")
                plt.plot(x, rep_traj[:, 2], c='b', label="_")
        plt.xlabel("Event")
        plt.ylabel("Gradient Moment")
        plt.legend()
        plt.grid()

    if new_figure:
        plt.show()




def write_data_to_seq_file(seq: mr0.Sequence, file_name: str):
    """Write all sequence data needed for reconstruction into a .seq file.

    The data is compressed, base64 encoded and inserted as a comment into the
    pulseq .seq file, which means it is ignored by all interpreters and only
    slightly increases the file size.

    Parameters
    ----------
    seq : Sequence
        Should be the sequence that was used to produce the .seq file
    file_name : str
        The file name to append the data to, it is not checked if this
        actually is a pulseq .seq file.
    """
    kspace = seq.get_kspace().detach()
    adc_usage = torch.cat([rep.adc_usage[rep.adc_usage > 0] for rep in seq])

    # Transpose for more efficient compression (contiguous components)
    kspace_enc = np.ascontiguousarray(kspace.T.cpu().numpy())
    # Delta encoding (works very well for cartesian trajectories)
    kspace_enc[:, 1:] -= kspace_enc[:, :-1]
    # Reduce precision, don't need 32bit for a kspace
    kspace_enc = kspace_enc.astype(np.float16)

    # Compressing adc_usage
    assert -128 <= adc_usage.min() <= 127, "8 bit are not enough"
    adc_usage_enc = adc_usage.cpu().numpy().astype(np.int8)

    # Compress and encode with base64 to write as legal ASCII text
    buffer = io.BytesIO()
    np.savez_compressed(buffer, kspace=kspace_enc, adc_usage=adc_usage_enc)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

    # The pulseq Siemens interpreter has a bug in the comment code leading to
    # errors if comments are longer than MAX_LINE_WIDTH = 256. We split the
    # data into chunks of 250 bytes to be on the safe side.
    with open(file_name, "a") as file:
        for i in range(0, len(encoded), 250):
            file.write(f"\n# {encoded[i:i+250]}")
        file.write("\n")


def extract_data_from_seq_file(
    file_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts kspace and adc_usage written with ``write_data_to_seq_file``.

    Parameters
    ----------
    file_name : str
        The name of the file the kspace was previously written to.

    Returns
    -------
    The original kspace and the adc_usage. There might be a  loss of precision
    because the kspace is written as 16 bit (half precision) floats and the
    usage as 8 bit integer (-128 to 127), this could be changed.
    """
    try:
        with open(file_name, "r") as file:
            # Find the last n lines that start with a '#'
            lines = file.readlines()
            
            if lines[-1][-1:] != '\n':
                lines[-1] = lines[-1] + '\n'
            
            n = len(lines)
            while n > 0 and lines[n-1][0] == '#':
                n -= 1
            if n == len(lines):
                raise ValueError("No data comment found at the end of the file")

            # Join the parts of the comment while removing "# " and "\n"
            encoded = "".join(line[2:-1] for line in lines[n:])
            # print(encoded)
            decoded = base64.b64decode(encoded, validate=True)

            data = np.load(io.BytesIO(decoded))
            kspace = np.cumsum(data["kspace"].astype(np.float32), 1).T
            adc_usage = data["adc_usage"].astype(np.int32)

            return torch.tensor(kspace), torch.tensor(adc_usage)
    except Exception as e:
        raise ValueError("Could not extract data from .seq") from e




def resize(tensor: torch.Tensor, new_size, mode='area'):
    # Functions expects batch x channels x (depth) x height x width
    if tensor.shape[-1] == 1:  # 2D, possible modes: 'area', 'bicubic'
        tensor_resized = tensor.squeeze().unsqueeze(0)
        tensor_resized = interpolate(tensor_resized, size=new_size[:2], mode=mode)
    else:  # 3D, possible modes: 'area', 'trilinear'
        tensor_resized = tensor.unsqueeze(0)
        tensor_resized = interpolate(tensor_resized, size=new_size, mode=mode)
    return tensor_resized.view(new_size)


