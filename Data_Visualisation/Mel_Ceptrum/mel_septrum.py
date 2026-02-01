import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

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

SAVE_DIR = r"C:/Users/shikh/OneDrive/Desktop/Project/mel_cepstrum"

SAMPLE_RATE = 16000
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 512


os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style="white", context="paper", font_scale=1.2)


def average_mel_cepstrum(directory):
    cepstra = []

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            path = os.path.join(directory, file)

            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )

            cepstra.append(mfcc)

    # Average across files
    avg_cepstrum = np.mean(np.stack(cepstra), axis=0)

    return avg_cepstrum


# COMPUTE ALL AVERAGES
avg_ceps = [average_mel_cepstrum(d) for d in DIRS]

# GLOBAL COLOR SCALE
vmin = min(c.min() for c in avg_ceps)
vmax = max(c.max() for c in avg_ceps)


def set_clean_ticks(ax, cep):
    # X-axis (time)
    n_frames = cep.shape[1]
    x_ticks = np.linspace(0, n_frames - 1, 6).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("Time Frames")

    # Y-axis (cepstral coefficients)
    y_ticks = np.arange(0, N_MFCC, 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_ylabel("Cepstral Coefficient Index")


# SAVE FIGURES
for cep, title in zip(avg_ceps, TITLES):
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        cep,
        cmap="mako",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "MFCC Amplitude"},
        ax=ax
    )

    ax.set_title(f"{title} – Average Mel Cepstrum")
    set_clean_ticks(ax, cep)

    filename = title.replace(" ", "_").lower() + "_mel_cepstrum.png"
    save_path = os.path.join(SAVE_DIR, filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")
