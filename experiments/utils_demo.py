import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
from IPython.display import Audio
import os
from IPython.display import Audio
import soundfile as sf

root = "../data"
fs = 17400

def get_melspectrogram(y, sr, n_fft, hops):
    S = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=64, 
        fmin=150, 
        fmax=10_500,
        n_fft=n_fft, 
        hop_length=hops
    )

    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def plot_spectro_vs_neurogram(S_db, y,neurogram_specres, sr, hops, soundname="sound"):
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(2, 2,  height_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.arange(len(y)) * (1/sr), y, label=soundname[:-4], color='#4682B4', linewidth=4)
    ax1.plot(np.arange(len(y)) * (1/sr), y, color='white', linewidth=.05)
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    t = np.arange(S_db.shape[1]) * hops * (1 / sr)
    cmap = 'cividis'

    ax2.pcolormesh(
        t, neurogram_specres.frequencies, S_db, cmap=cmap
    )

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)

    sp_data_phast = np.vstack(
        [
            scipy.signal.resample(neurogram_specres.data[i], S_db.shape[1])
            for i in range(neurogram_specres.data.shape[0])
        ]
    )

    img = ax3.pcolormesh(
        t, neurogram_specres.frequencies, sp_data_phast, cmap=cmap
    )

    ax2.set_title("Spectrogram")
    ax3.set_title("Neurogram")

    ax2.set_ylabel("[Hz]")
    ax2.set_yscale("log", base=2)
    ax2.set_yticks([pow(2, i) for i in range(9, 14)], [pow(2, i) for i in range(9, 14)])
    ax2.set_ylim(500, None)
    ax2.set_xlim(0, np.max(t))
    ax3.set_ylabel("[Hz]")

    for ax in ax2, ax3:
        ax.set_xlabel("Time [s]")
    plt.tight_layout()


def plot_and_play_reconstructed(original, reconstructed, fs, mel_scale, soundfile="original",
                                n_fft=512, hops=32, min_freq=150, max_freq=10500):
    f, axes = plt.subplots(2, 2, sharey="row", sharex=True, figsize=(13, 5), height_ratios=[1, 3])

    m = 2
    cmap = 'cividis'

    for ((ax1, ax2), y, title) in zip(axes.T, (original, reconstructed), (soundfile[:-4], "EH Vocoded")):
        t_sound = np.arange(len(y)) / fs   
        ax1.plot(t_sound, y, color='#4682B4', linewidth=4)
        ax1.plot(t_sound, y, color='white', linewidth=.05)
        spectrogram = get_melspectrogram(y, fs, n_fft * m, hops * m)
        t_spec = np.arange(spectrogram.shape[1]) * hops*m * (1 / fs)
        ax2.pcolormesh(t_spec, mel_scale, spectrogram, vmin=-80, vmax=0, cmap=cmap)
        ax1.set_title(title, fontsize=16)
        ax1.set_yticks([-.5, .5], [-.5, .5], fontsize=14)
        ax2.set_xlabel("Time [s]", fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax1.grid()
        display(Audio(data=y, rate=fs, element_id=title))


    axes[0][0].set_ylabel("Ampl.", fontsize=15)
    ax2 = axes[1][0]
    ax2.set_ylabel("Frequency [Hz]",  fontsize=15)
    ax2.set_yscale("log", base=2)
    ax2.set_yticks([pow(2, i) for i in range(8, 14)], [pow(2, i) for i in range(8, 14)], fontsize=14)
    ax2.set_ylim(min_freq, 8200)
    ax2.set_xlim(0, np.max(t_spec))

    for ax, label in zip(axes.ravel(), "ABCDEF"):
        ax.text(0.95, 0.95, label,
            transform=ax.transAxes,
            fontsize=15, fontweight='bold',
            va='top', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.15'))

    plt.tight_layout()
    # plt.savefig("choice_no_noise.png", dpi=600)
    
def rms_db(y):
    rms = np.sqrt(np.mean(y**2))
    return 20 * np.log10(rms)

def mix_v2(stim, noise, snr):
    mask = np.nonzero(stim)
    db_stim = rms_db(stim[mask])
    db_noise = rms_db(noise)
    db_tgt = db_noise + snr
    gain = 10 ** ((db_tgt - db_stim) / 20)
    stim_scaled = stim * gain
    mixed = noise + stim_scaled
    return mixed   

def scale_to_target_dbfs(y, target_dbfs):
    current_dbfs = rms_db(y)
    diff = target_dbfs - current_dbfs
    gain = 10 ** (diff / 20)
    return y * gain

def mix_sound_with_noise(soundfile, snr=-4, sr=fs):
    noise = os.path.join(root, "din/tripletnoise.wav")
    noise_y, sr = librosa.load(noise, sr=None)
    filename = os.path.join(root, soundfile)
    sound_y, sr = librosa.load(filename, sr=sr)
    noise_seg = noise_y[100:100+ len(sound_y)]
    mixed = scale_to_target_dbfs(mix_v2(sound_y, noise_seg, snr), rms_db(sound_y))
    t = np.arange(len(sound_y)) / sr

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5), sharey=True)
    ax1.plot(t, noise_seg)
    ax1.set_title("Noise")
    ax2.plot(t, sound_y)
    ax2.set_title("Original Sound")
    ax3.plot(t, mixed)
    ax3.set_title(f"Mixed Sound at {snr} dB SNR")
    f.tight_layout()
    
    display(Audio(data=mixed, rate=sr))
    soundname_4db_snr = f"{soundfile[:-4]}_with_noise_{snr}db_snr.wav"
    filename_4db_snr = os.path.join(root, soundname_4db_snr)
    sf.write(filename_4db_snr, mixed, sr)
    return soundname_4db_snr