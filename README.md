# African Empathy AI – Speech Emotion Recognition Dataset Pipeline

## Overview

African Empathy AI is a Speech Emotion Recognition (SER) project designed to build a clean, standardized, and high-quality dataset for training deep learning models capable of recognizing human emotions from speech.

This notebook focuses primarily on **data engineering and preprocessing**, transforming multiple public speech emotion datasets into a unified dataset suitable for training a Wav2Vec2-based emotion recognition model.

Rather than training the model directly, the notebook performs:

- Dataset parsing
- Emotion label standardization
- Audio validation
- Audio quality inspection
- Dataset cleaning
- Data visualization
- Dataset balancing checks
- Train/validation split generation
- Preparation of Hugging Face datasets
- Label encoding for machine learning

The final output is a production-ready dataset that can be used for transformer-based speech emotion recognition.

---

# Project Objectives

The notebook aims to solve several challenges commonly encountered when combining multiple speech datasets:

- Different datasets use different naming conventions.
- Emotion labels vary across datasets.
- Audio files have different lengths.
- Sample rates are inconsistent.
- Corrupted recordings may exist.
- Duplicate recordings may appear.
- Machine learning models require standardized input.

This pipeline automatically handles these issues to produce a clean master dataset.

---

# Dataset Sources

The notebook combines four well-known Speech Emotion Recognition datasets.

| Dataset | Description |
|----------|-------------|
| TESS | Toronto Emotional Speech Set |
| RAVDESS | Ryerson Audio-Visual Database of Emotional Speech and Song |
| SAVEE | Surrey Audio-Visual Expressed Emotion Dataset |
| CREMA-D | Crowd-Sourced Emotional Multimodal Actors Dataset |

Each dataset contains recordings from different speakers expressing various emotions.

---

# Supported Emotions

All datasets are converted into a unified emotion taxonomy.

| Original Labels | Standard Label |
|-----------------|----------------|
| angry, anger, ang | angry |
| happy, hap | happy |
| sad | sad |
| fear, fearful, fea | fear |
| neutral, neu | neutral |
| calm | calm |
| disgust, dis | disgust |

Standardizing labels ensures consistency regardless of the source dataset.

---

# Notebook Structure

The notebook is organized into several logical stages.

## 1. Dataset Initialization

The project begins by defining:

- Dataset locations
- Output file paths
- Target sample rate
- Maximum audio duration
- Global configuration values

Important parameters include:

```python
TARGET_SR = 16000
MAX_DURATION = 4
```

Every audio recording is ultimately converted to:

- 16 kHz
- Mono
- Maximum duration of 4 seconds

This standardization is critical for deep learning models.

---

## 2. Audio Validation

Before including an audio file in the dataset, it is validated.

Validation checks include:

- Audio can be opened
- Audio is not empty
- Minimum duration requirement
- Readability by Librosa

Files that fail validation are excluded.

---

## 3. Audio Preparation

Each recording is normalized to a fixed length.

If audio is longer than 4 seconds:

- it is trimmed

If audio is shorter than 4 seconds:

- zero-padding is applied

Benefits include:

- consistent tensor dimensions
- easier batching
- faster training
- compatibility with transformer models

---

## 4. Dataset Parsing

Each dataset has its own filename format.

Dedicated parsers extract metadata such as:

- emotion
- speaker
- gender
- dataset source
- file path

Examples include:

### SAVEE

```
DC_a01.wav
```

Extracts:

- Speaker
- Emotion

---

### CREMA-D

```
1001_IEO_SAD_LO.wav
```

Extracts:

- Speaker ID
- Emotion
- Dataset source

---

Similar parsing functions exist for:

- TESS
- RAVDESS

---

## 5. Master Dataset Construction

After scanning every dataset, all recordings are merged into one DataFrame.

Each row represents one recording.

Example structure:

| Column | Description |
|----------|-------------|
| path | Audio file location |
| emotion | Standardized emotion |
| dataset | Dataset source |
| speaker | Speaker identifier |
| gender | Speaker gender |

This DataFrame becomes the project's master dataset.

---

# Audio Quality Inspection

The notebook performs extensive quality analysis for every recording.

The following acoustic features are extracted.

## Duration

Recording length in seconds.

Used to detect:

- incomplete recordings
- unusually long recordings

---

## Sample Rate

Ensures recordings have been loaded correctly.

Expected:

```
16000 Hz
```

---

## RMS Energy

Measures signal loudness.

Useful for identifying:

- silent files
- extremely quiet recordings

---

## Zero Crossing Rate (ZCR)

Measures how often the waveform changes sign.

Can indicate:

- speech activity
- noise
- recording quality

---

## Spectral Centroid

Represents the "center of mass" of the spectrum.

Useful for identifying:

- bright sounds
- muffled recordings
- abnormal signals

---

## Silence Ratio

Calculates the proportion of silent samples.

High silence ratios may indicate:

- damaged recordings
- empty files
- recording failures

---

## Recording Status

Each recording is labeled:

```
valid
```

or

```
error
```

Corrupted files are automatically removed.

---

# Dataset Cleaning

Several cleaning operations are performed.

## Duplicate Removal

Duplicate audio paths are removed.

---

## Corrupted File Removal

Files producing errors during analysis are discarded.

---

## Dataset Shuffle

The remaining dataset is randomly shuffled.

Randomization improves training quality.

---

# Dataset Analysis

The notebook generates summary statistics.

These include:

- emotion distribution
- dataset contribution
- gender distribution
- average duration

It also produces visualizations such as:

- emotion frequency bar chart
- dataset contribution chart
- duration histogram

These help evaluate class balance before training.

---

# Train / Validation Split

After cleaning, the dataset is divided into:

- Training set
- Validation set

The resulting CSV files are saved for later model training.

Typical outputs include:

```
empathy_train.csv
empathy_valid.csv
```

---

# Notebook 2 – Model Preparation

The second section prepares the cleaned dataset for transformer-based learning.

This notebook does **not** perform model training; instead, it prepares all required inputs.

Major steps include:

- loading training CSV
- loading validation CSV
- encoding emotion labels
- building Hugging Face datasets
- preparing audio inputs
- loading the Wav2Vec2 feature extractor

---

# Label Encoding

Machine learning models require numerical labels.

Example:

| Emotion | Label |
|----------|------:|
| angry | 0 |
| calm | 1 |
| disgust | 2 |
| fear | 3 |
| happy | 4 |
| neutral | 5 |
| sad | 6 |

The fitted `LabelEncoder` is saved for inference.

```
label_encoder.pkl
```

---

# Hugging Face Dataset

The cleaned Pandas DataFrames are converted into Hugging Face `Dataset` objects.

Advantages include:

- efficient preprocessing
- streaming
- batching
- compatibility with Transformers
- simplified training pipeline

---

# Feature Extraction

The notebook loads the pretrained Wav2Vec2 feature extractor:

```
facebook/wav2vec2-large-xlsr-53
```

This model converts raw waveforms into numerical representations suitable for transformer models.

---

# Project Outputs

The notebook produces several important artifacts:

```
empathy_master.csv
```

Combined master dataset.

---

```
empathy_train.csv
```

Training split.

---

```
empathy_valid.csv
```

Validation split.

---

```
label_encoder.pkl
```

Saved emotion encoder.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- SoundFile
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- Transformers
- Wav2Vec2
- tqdm

---

# Workflow Summary

```
Raw Datasets
      │
      ▼
Dataset Parsing
      │
      ▼
Emotion Standardization
      │
      ▼
Audio Validation
      │
      ▼
Quality Inspection
      │
      ▼
Cleaning
      │
      ▼
Duplicate Removal
      │
      ▼
Visualization
      │
      ▼
Train / Validation Split
      │
      ▼
Label Encoding
      │
      ▼
Hugging Face Dataset
      │
      ▼
Wav2Vec2 Feature Extraction
      │
      ▼
Ready for Model Training
```

---

# Future Improvements

Potential enhancements include:

- Support additional African languages and dialects.
- Incorporate data augmentation techniques such as pitch shifting, noise injection, and time stretching.
- Add automatic speaker balancing.
- Include emotion intensity estimation.
- Integrate cross-validation.
- Fine-tune multilingual transformer models.
- Deploy the trained model as a real-time empathy recognition API.

---

# Author

**David Guchu**

African Empathy AI is a research initiative focused on building intelligent, empathetic speech systems capable of understanding human emotions through voice. The project emphasizes robust data preparation, standardized preprocessing, and scalable machine learning workflows for transformer-based Speech Emotion Recognition.
