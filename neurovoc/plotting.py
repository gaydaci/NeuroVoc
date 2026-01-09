import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import matplotlib.ticker as mplticker
from matplotlib.colors import LogNorm, Normalize
import scipy.signal
from matplotlib.ticker import FormatStrFormatter

def audio_vs_reconstructed(
    audio_signal,
    reconstructed_signal,
    audio_fs,
    n_mels,
    min_freq,
    max_freq,
    title = "",
):
    if reconstructed_signal.size != audio_signal.size:
        reconstructed_signal = scipy.signal.resample(reconstructed_signal, audio_signal.size)
    
    fig = plt.figure()
    fig.suptitle(title)
    gs = plt.GridSpec(2, 2, wspace=0.25, hspace=0.25)  # 2x2 grid
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    for i, (ax, sig) in enumerate(
        zip((ax0, ax1), (audio_signal, reconstructed_signal))
    ):
        S = librosa.feature.melspectrogram(
            y=sig, sr=audio_fs, n_mels=n_mels, fmin=min_freq, fmax=max_freq
        )

        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB,
            x_axis="time",
            y_axis="mel",
            sr=audio_fs,
            fmin=min_freq,
            fmax=max_freq,
            ax=ax,
            vmax=0,
            vmin=-80,
        )
        ax.set_xlabel(None)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i == 1:
            ax.set_title("reconstructed")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            ax.set_title("original")

    t = np.arange(audio_signal.size) / audio_fs
    ax2.plot(t, audio_signal, label="original", alpha=0.6)
    ax2.plot(t, reconstructed_signal, label="reconstructed", alpha=0.6)
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("time [s]")


def frequency_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_yscale("symlog", linthresh=1000.0, base=2)
    ax.yaxis.set_major_formatter(mplticker.ScalarFormatter())
    ax.yaxis.set_major_locator(
        mplticker.SymmetricalLogLocator(ax.yaxis.get_transform())
    )
    ax.yaxis.set_label_text("frequency [Hz]")


def time_vs_freq(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("time [s]")
    frequency_ax(ax)


def plot_heatmap(
    t,
    y,
    data,
    ax=None,
    fig=None,
    show_bands: bool = False,
    pad_idx: bool = False,
    figsize=(9, 4),
    logcolors: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    norm = (
        LogNorm(vmin=data.min(), vmax=data.max())
        if logcolors
        else Normalize(vmin=data.min(), vmax=data.max())
    )

    if pad_idx:
        n_idx = np.nonzero(data.sum(axis=0))[0]
        n_idx = np.unique(np.c_[n_idx - 1, n_idx, n_idx + 1].ravel())
        n_idx = n_idx[n_idx < t.size]
        img = ax.pcolormesh(t[n_idx], y, data[:, n_idx], cmap="inferno", norm=norm)
    else:
        img = ax.pcolormesh(t[:], y, data[:, :], cmap="inferno", norm=norm)
    time_vs_freq(ax)
    ax.set_xlabel("time [s]")
    fig.colorbar(img, ax=ax)

    if show_bands:
        for f in y:
            ax.plot([0, t[-1]], [f, f], color="white", alpha=0.3)
        ax.set_xlim(0, t[-1])
