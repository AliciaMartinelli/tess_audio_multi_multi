# TESS ZCR, RMS and MFCC (audio features) + RF, KNN, XGB and Ridge Classification

This repository contains the implementation of a machine learning pipeline for classifying exoplanet candidates using preprocessed TESS light curves. These light curves will be transformed into audio files and augmented with a lowpass filter with a cutoff frequency of 20Hz. Then three features will be extracted (ZCR, RMS and MFCC) and four different classical machine learning classifiers (RF, KNN, XGB and Ridge) will be trained.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
tess_audio_multi_multi/
├── convert_arrayfiles.py      # Convert the .npy files into audio files save in the dataset folder
├── data_aug.py                # Adds a lowpass filter with a cutoff frequency of 20Hz to the audio files and generates new augmented audio files into the aug_dataset folder
├── feature_extractor.py       # Extracts ZCR, RMS and MFCC features and saves them in the features folder
├── train_KNN.py               # KNN training and evaluation with Optuna
├── train_RF.py                # RF training and evaluation with Optuna
├── train_Ridge.py             # Ridge training and evaluation with Optuna
├── train_XGB.py               # XGB training and evaluation with Optuna
├── run_all.sh                 # Run the pipeline for all features and all models
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed TESS dataset
The preprocessed TESS dataset used in this project is based on the public release from Yu et al. (2019) and is available via the Astronet-Vetting GitHub repository [https://github.com/yuliang419/Astronet-Vetting/tree/master/astronet/tfrecords](https://github.com/yuliang419/Astronet-Vetting/tree/master/astronet/tfrecords)

Download the TFRecords from the GitHub repository, convert them into .npy files and save them in the raw folder.


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/tess_audio_multi_multi.git
    cd tess_audio_multi_multi
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `optuna`, `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Convert .npy into .wav audio files:
```bash
python convert_arrayfiles.py
```
This will generate the .way audio files for each .npy light curve and save them in the dataset folder depending on the label in a subfolder "0" or "1".

2. Add a lowpass filter to each audio signal and save the files into aug_dataset folder
```bash
python data_aug.py
```
This step will add a lowpass filter with a cutoff frequency of 20Hz to the audio signals and save them into the aug_dataset folder into subfolders "0" and "1" depending on the label.

3. Extract ZCR, RMS and MFCC features:
```bash
python features_extractor.py
```
This extracts ZCR, RMS and MFCC features of the augmented audio files and saves the results in the features folder.

4. Train the models (KNN, RF, Ridge, XGB) with the features (ZCR, RMS and MFCC) with this pipeline script and save the logs of each training into the log folder.
```bash
./run_all.sh
```
This will train all the models with all the features and saves the results in log files in the log folder.

## Thesis Context

This repository corresponds to the experiment described in:
- **Section 3.2**: Audio-based feature extraction and classification using multiple models 


**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025