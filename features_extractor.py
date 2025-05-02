import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pathlib
from tqdm.auto import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize(data):
    xmax, xmin = data.max(), data.min()
    if xmax == xmin:
        return np.zeros_like(data)
    return 2 * ((data - xmin) / (xmax - xmin)) - 1

def extract_rms_features(dataset_path, features_folder):
    print("Extracting RMS features...")
    save_path = os.path.join(features_folder, "rms")
    files = sorted(list(pathlib.Path(dataset_path).rglob("*.wav")))

    for f in tqdm(files):
        signal, sr = librosa.load(f, duration=5.0)
        S, _ = librosa.magphase(librosa.stft(signal))
        # OVER MAGNITUDE SPECTRUM
        rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=512, center=True, pad_mode='constant')
        # OVER TIME SIGNAL with frame_length=1024
        # rms = librosa.feature.rms(y=signal, frame_length=1024, hop_length=512, center=True)
        class_folder = os.path.join(save_path, f.parts[-2])
        ensure_dir(class_folder)
        np.save(os.path.join(class_folder, f.stem + ".npy"), rms)

def extract_zcr_features(dataset_path, features_folder):
    print("Extracting ZCR features...")
    save_path = os.path.join(features_folder, "zcr")
    files = sorted(list(pathlib.Path(dataset_path).rglob("*.wav")))

    for f in tqdm(files):
        signal, sr = librosa.load(f, duration=5.0)
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512, center=True)
        class_folder = os.path.join(save_path, f.parts[-2])
        ensure_dir(class_folder)
        np.save(os.path.join(class_folder, f.stem + ".npy"), zcr)

def extract_melspectrogram_images(dataset_path, features_folder):
    print("Extracting Mel-spectrogram images...")
    save_path = os.path.join(features_folder, "mel_spectrogram")
    files = sorted(list(pathlib.Path(dataset_path).rglob("*.wav")))

    for f in tqdm(files):
        signal, sr = librosa.load(f, duration=5.0)
        fig, ax = plt.subplots(1, 1, tight_layout=True, frameon=False, figsize=(2.56, 2.56))
        ax.specgram(signal, Fs=sr)
        ax.axis('off')
        class_folder = os.path.join(save_path, f.parts[-2])
        ensure_dir(class_folder)
        fig.savefig(os.path.join(class_folder, f.stem + ".png"))
        plt.close(fig)


def extract_mfcc_features(dataset_path, features_folder, n_mfcc=30):
    print("Extracting MFCC features...")
    save_path = os.path.join(features_folder, "mfcc")
    files = sorted(list(pathlib.Path(dataset_path).rglob("*.wav")))

    for f in tqdm(files):
        signal, sr = librosa.load(f, duration=5.0)
        mfcc = np.mean(librosa.feature.mfcc(
            y=signal, sr=sr, n_mfcc=n_mfcc, fmin=0., fmax=1000., center=True, n_mels=20, n_fft=1024), axis=0)
        mfcc_normalized = normalize(mfcc)
        class_folder = os.path.join(save_path, f.parts[-2])
        ensure_dir(class_folder)
        np.save(os.path.join(class_folder, f.stem + ".npy"), mfcc_normalized)

if __name__ == "__main__":
    DATASET_PATH = "aug_dataset"
    FEATURES_PATH = "features"

    extract_rms_features(DATASET_PATH, FEATURES_PATH)
    extract_zcr_features(DATASET_PATH, FEATURES_PATH)
    extract_mfcc_features(DATASET_PATH, FEATURES_PATH)
    # extract_melspectrogram_images(DATASET_PATH, FEATURES_PATH)

