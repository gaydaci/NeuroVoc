"""Module containing methods to reconstruct .wav files from Neurograms"""

import os
import pathlib
from math import gcd

import librosa
import scipy
import numpy as np

from .generate import Neurogram, min_max_scale

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def rms_db(y):
    return 20 * np.log10(rms(y))

def scale_to_target_dbfs(y, target_dbfs):
    current_dbfs = rms_db(y)
    diff = target_dbfs - current_dbfs
    gain = 10 ** (diff / 20)
    return y * gain


def reconstruct_neurogram(
    M: np.ndarray, sr: int, min_freq: int, max_freq: int, n_fft: int, n_hop: int
) -> np.ndarray:
    """
    Parameters
    ----------
    M: np.ndarray
        A neurogram structure, scaled to a power spectrum, and downsampled by a factor
        of n_hop
    sr: int
        The sampling rate of the original neurogram (before resampling with n_hop)
    min_freq: int
        The lower bound of the filter bank
    max_freq: int
        The upper bound of the filter bank
    n_hop: int
        The number of hops that were applied to M
    """

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=M.shape[-2],
        dtype=M.dtype,
        fmin=min_freq,
        fmax=max_freq,
    )
    
    # Estimate linear-frequency power spectrogram via non-negative least squares
    inverse = librosa.util.nnls(mel_basis, M)
    
    # Convert power spectrum to magnitude spectrum
    inverse = np.power(inverse, 1.0 / 2.0, out=inverse)

    # Reconstruct time-domain waveform using Griffin–Lim phase estimation
    reconstructed = librosa.feature.inverse.griffinlim(
        inverse,
        n_iter=32,
        hop_length=n_hop,
        win_length=None,
        n_fft=n_fft,
        window="hann",
        center=True,
        dtype=np.float32,
        length=None,
        pad_mode="constant",
        momentum=0.99,
        init="random",
        random_state=None,
    )
    return reconstructed


def downsample(data: np.ndarray, n_hop: int) -> np.ndarray:
    """
    Downsample the neurogram along the temporal axis using polyphase resampling
    to reduce reconstruction cost. The target length is ceil(N / n_hop), where
    n_hop is the hop size. Resampling is performed with a rational factor reduced
    to lowest terms and includes anti-aliasing filtering. Output values are
    clipped to [0, 1].
    """
    n_s = int(np.ceil(data.shape[1] / n_hop))
    g = gcd(n_s, data.shape[1])
    data = np.array(
        [scipy.signal.resample_poly(row, n_s // g, data.shape[1] // g) for row in data]
    ).clip(0, 1)
    return data


def power_scale(data, ref_db: float = 50.0):
    """
    Map normalized neurogram values to a decibel-like range [-80, 0] and
    convert to a linear power scale relative to a reference level.
    """
    data = min_max_scale(data, -80, 0, data_min=0, data_max=1)
    # Convert dB-scaled neurogram to power scale relative to a 50 dB reference 
    # ( Mel-band power representation)
    data = librosa.db_to_power(data, ref=ref_db) 
    return data


def reconstruct(
    neurogram: Neurogram | str | pathlib.Path,
    n_hop: int = 32,
    n_fft: int = 512,
    ref_db: float = 50,
    target_sr: int = 44100,
    target_db_fs: int = -20,
    **kwargs
):
    if isinstance(neurogram, (str, pathlib.Path)) and os.path.isfile(neurogram):
        neurogram = Neurogram.load(neurogram)

    data = downsample(neurogram.data, n_hop)
    data = power_scale(data, ref_db)

    reconstructed = reconstruct_neurogram(
        data,
        neurogram.sample_rate,
        neurogram.min_freq,
        neurogram.max_freq,
        n_fft,
        n_hop,
    )
    
    # Resample reconstructed waveform to the original audio sampling rate
    reconstructed = librosa.resample(
        reconstructed, orig_sr=neurogram.sample_rate, target_sr=target_sr
    )
    
    # Normalize RMS level to the target dB full scale
    reconstructed = scale_to_target_dbfs(reconstructed, target_db_fs)
    return reconstructed
