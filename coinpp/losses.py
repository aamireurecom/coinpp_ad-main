import torch


mse_fn = torch.nn.MSELoss()
per_element_mse_fn = torch.nn.MSELoss(reduction="none")


def batch_mse_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).

    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).

    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    per_element_mse = per_element_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def mse2psnr(mse):
    """Computes PSNR from MSE, assuming the MSE was calculated between signals
    lying in [0, 1]. Add a small value to MSE to avoid division by zero.

    Args:
        mse (torch.Tensor or float):
    """
    return -10.0 * (torch.log10(mse + 10e-20))


def psnr_fn(x1, x2):
    """Computes PSNR between signals x1 and x2. Note that the values of x1 and
    x2 are assumed to lie in [0, 1].

    Args:
        x1 (torch.Tensor): Shape (*).
        x2 (torch.Tensor): Shape (*).
    """
    return mse2psnr(mse_fn(x1, x2))


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    sqrt = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return sqrt.transpose(2, 1)

def stft_metrics(y, yhat, fft_size, hop_size, win_length, window):
    """

    Args:
        y (Tensor) input: [B, L] batch, length, dim (input signal dimensionality)
        yhat (Tensor) fitted: [B, L] batch, length, dim
        fft_size:
        hop_size:
        win_length:

    Returns:
       l1_metrics, log_metrics
    """
    # dim = y.shape[-1]
    # l1_metric, log_metric = torch.zeros(dim), torch.zeros(dim)
    # Cycle through the  of the input
    # for i in range(dim):

    y_stft = stft(y, fft_size, hop_size, win_length, window=window)
    yhat_stft = stft(yhat, fft_size, hop_size, win_length, window=window)

    # print(f'stft_metrics y_stft.shape = {y_stft.shape}')
    l1_metric = torch.norm(y_stft - yhat_stft, p="fro") / torch.norm(y_stft, p="fro")
    log_metric = torch.nn.functional.l1_loss(torch.log(yhat_stft), torch.log(y_stft))
    return l1_metric, log_metric


###############################################################################
# functional representation of loss via functorch package
###############################################################################
def loss_functional(params, fmodel, buffers, x, y):
    out1 = fmodel(params, buffers, x)
    return mse_fn(out1, y)


# From here
# https://github.com/facebookresearch/denoiser/blob/main/denoiser/stft_loss.py
###############################################################################

# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


# def stft(x, fft_size, hop_size, win_length, window):
#     """Perform STFT and convert to magnitude spectrogram.
#     Args:
#         x (Tensor): Input signal tensor (B, T).
#         fft_size (int): FFT size.
#         hop_size (int): Hop size.
#         win_length (int): Window length.
#         window (str): Window function type.
#     Returns:
#         Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
#     """
#     x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
#     real = x_stft[..., 0]
#     imag = x_stft[..., 1]
#
#     # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
#     return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer("window", getattr(torch, window)(win_length, device=device))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 # fft_sizes=[1024, 2048, 512],
                 # hop_sizes=[120, 240, 50],
                 # win_lengths=[600, 1200, 240],
                 fft_sizes=[100],#, 2048, 512],
                 hop_sizes=[50],
                 win_lengths=[100],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


