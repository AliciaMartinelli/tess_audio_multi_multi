import os
import pathlib
import numpy as np
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav

output_dir = './dataset'
os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)
os.makedirs(os.path.join(output_dir, '0'), exist_ok=True)

def convert_lightcurve_to_audio(lightcurve_raw, sampling_rate=22050, duration=5):
    desired_samples = sampling_rate * duration

    time_original = np.linspace(0, 1, len(lightcurve_raw))
    time_new = np.linspace(0, 1, desired_samples)

    interpolator = interp1d(time_original, lightcurve_raw, kind='linear', fill_value='extrapolate')

    lightcurve_interpoliert = interpolator(time_new)

    audio_signal = (lightcurve_interpoliert * 32767).astype(np.int16)

    return audio_signal

def process_lightcurves(raw_folder, output_folder):
    npy_files = sorted(pathlib.Path(raw_folder).rglob("*.npy"))
    
    for npy_file in npy_files:
        data = np.load(npy_file, allow_pickle=True)
        data_dict = data.item()
        lightcurve_raw = data_dict['lightcurve']
        label = data_dict['label']

        audio_signal = convert_lightcurve_to_audio(lightcurve_raw, sampling_rate=22050, duration=5)

        output_audio_path = os.path.join(output_folder, str(label), f"{npy_file.stem}.wav")
        pathlib.Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)
        wav.write(output_audio_path, 22050, audio_signal)
        print(f"Saved: {output_audio_path}")

raw_folder = './raw'
output_folder = './dataset'

process_lightcurves(raw_folder, output_folder)
