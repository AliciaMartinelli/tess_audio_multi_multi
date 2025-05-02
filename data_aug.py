import warnings
warnings.filterwarnings('ignore')

import os
import pathlib
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise
from scipy.signal import butter, filtfilt

class audioPreprocessing:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def readAudio(self, fileName):
        signal, sr = librosa.load(fileName, sr=self.sample_rate)
        return signal

    def audioAugmentationGaussian(self, signal):
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.001, p=1)
        ])
        return augment(signal, self.sample_rate)
    
    def butter_lowpass_filter(self, signal, sr, cutoff=20, order=5):
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    
    def audioAugmentation(self, signal):
        filtered_signal = self.butter_lowpass_filter(signal, self.sample_rate, cutoff=20, order=5)
        return filtered_signal
    
obj = audioPreprocessing()
targetSampleRate = 22050

def load_data(path):
    audioFiles = sorted(list(pathlib.Path(path).rglob("*.wav")))
    classes = [str(f.parent).split(os.sep)[-1] for f in audioFiles]
    return audioFiles, classes

def augment_data(audioFileNames, classes, save_path):
    global obj

    print("before: ", len(audioFileNames))
    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        signal_org = obj.readAudio(x)
        
        class_label = classes[idx]
        
        class_save_path = os.path.join(save_path, class_label)
        pathlib.Path(class_save_path).mkdir(parents=True, exist_ok=True)
        
        augmented_signal = obj.audioAugmentation(signal_org)
        augmented_file_name = f"{x.stem}_augmented{x.suffix}"
        augmented_path = os.path.join(class_save_path, augmented_file_name)
        sf.write(augmented_path, augmented_signal, targetSampleRate)

    print("After:", len(list(pathlib.Path(save_path).rglob("*.wav"))))


if __name__ == "__main__":
    np.random.seed(42)

    X, y = load_data("dataset")
    print(f"Train: {len(X)} - {len(y)}")

    augment_data(X, y, "aug_dataset")

