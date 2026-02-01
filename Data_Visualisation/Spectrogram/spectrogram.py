import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns







def set_clean_ticks(ax, spec, sr, n_fft, hop_length, max_freq):
    # X-axis (time frames → seconds)
    n_frames = spec.shape[1]
    x_ticks = np.linspace(0, n_frames - 1, 6).astype(int)
    x_labels = np.round(
        (x_ticks * hop_length) / sr, 1
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")

    # Y-axis (frequency bins → Hz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]

    y_ticks = np.linspace(0, len(freqs) - 1, 6).astype(int)
    y_labels = np.round(freqs[y_ticks]).astype(int)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Frequency (Hz)")










# CONFIG
DIRS = [
    r"C:/Users/shikh/OneDrive/Desktop/Project/Data_Visualisation/Tank",
    r"C:/Users/shikh/OneDrive/Desktop/Project/Data_Visualisation/Aeroplane",
    r"C:/Users/shikh/OneDrive/Desktop/Project/Data_Visualisation/Construction",
    r"C:/Users/shikh/OneDrive/Desktop/Project/Data_Visualisation/Natural noise"
]

TITLES = [
    "Tank",
    "Aeroplane",
    "Construction",
    "Natural noise"
]

SAVE_DIR = r"C:/Users/shikh/OneDrive/Desktop/Project/spectrograms"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
MAX_FREQ = 1000   # Hz


os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style="white", context="paper", font_scale=1.2)


def average_spectrogram(directory):
    specs = []

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            path = os.path.join(directory, file)
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

            D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            specs.append(S_db)

    avg_spec = np.mean(np.stack(specs), axis=0)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_mask = freqs <= MAX_FREQ

    return avg_spec[freq_mask, :]



avg_specs = []

for d in DIRS:
    avg_specs.append(average_spectrogram(d))

# GLOBAL COLOR SCALE
vmin = min(spec.min() for spec in avg_specs)
vmax = max(spec.max() for spec in avg_specs)

# SAVE INDIVIDUAL FIGURES
for spec, title in zip(avg_specs, TITLES):
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        spec,
        cmap="mako",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "Amplitude (dB)"},
        ax=ax
    )

    ax.set_title(f"{title} – Average Spectrogram")

    # CLEAN AXES
    set_clean_ticks(
        ax,
        spec,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        max_freq=MAX_FREQ
    )

    filename = title.replace(" ", "_").lower() + "_spectrogram.png"
    save_path = os.path.join(SAVE_DIR, filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
