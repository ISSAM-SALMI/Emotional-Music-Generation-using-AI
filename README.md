# Emotional Music Generation using AI

This project implements an advanced AI system for generating music conditioned on specific emotions (Joy, Sadness, Tension, Calm). It uses a Transformer-based architecture trained on the EMOPIA dataset.

## Project Overview

The system learns to generate MIDI sequences based on emotion labels.
- **Input**: Emotion Label (Q1, Q2, Q3, Q4)
- **Model**: Conditional Transformer (Decoder-only GPT-style)
- **Output**: Polyphonic Piano Music (MIDI) -> MP3

### Emotions Mapping
- **Q1**: Joy / High Arousal, High Valence
- **Q2**: Tension / High Arousal, Low Valence
- **Q3**: Sadness / Low Arousal, Low Valence
- **Q4**: Calm / Low Arousal, High Valence

## Prerequisites

1.  **Python 3.8+**
2.  **FluidSynth** (for MIDI to MP3 conversion)
    - **Windows**: Download from [FluidSynth Releases](https://github.com/FluidSynth/fluidsynth/releases). Add `bin` to your PATH.
    - **Linux**: `sudo apt-get install fluidsynth`
    - **MacOS**: `brew install fluidsynth`
3.  **SoundFont (.sf2)**
    - You need a SoundFont file to render MIDI to audio.
    - Download a free one like [FluidR3_GM.sf2](https://member.keymusician.com/Member/FluidR3_GM/index.html) or similar.
    - Place it in the project root and rename it to `soundfont.sf2` (or update `src/config.py`).

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

The project expects the following structure:
```
project_root/
├── EMOPIA_1.0/
│   ├── label.csv
│   └── ...
├── data/
│   └── midis/
│       ├── Q1_xxxx.mid
│       └── ...
├── src/
└── ...
```
Ensure your EMOPIA dataset is placed correctly as shown above.

## Training the Model

To train the model from scratch:

1.  Open a terminal in the project root.
2.  Run the training script:
    ```bash
    python -m src.train
    ```
3.  The script will:
    - Load MIDI files and labels.
    - Tokenize the music (Pitch, Duration, TimeShift).
    - Train the Transformer model.
    - Save checkpoints to `checkpoints/` after each epoch.

*Note: Training on CPU can be slow. A GPU is recommended.*

## Generating Music

Once you have a trained checkpoint (e.g., `checkpoints/model_epoch_20.pt`), you can generate music.

Run the generation script:

```bash
python -m src.generate --emotion Q1 --checkpoint checkpoints/model_epoch_20.pt --output my_joyful_song
```

### Arguments:
- `--emotion`: The target emotion. Options: `Q1` (Joy), `Q2` (Tension), `Q3` (Sadness), `Q4` (Calm).
- `--checkpoint`: Path to the trained model file.
- `--output`: Name of the output file (without extension).

The script will generate:
- `outputs/my_joyful_song.mid`
- `outputs/my_joyful_song.mp3` (if FluidSynth and SoundFont are set up)

## Project Structure

- `src/config.py`: Configuration settings (hyperparameters, paths).
- `src/data_loader.py`: Handles loading and processing the EMOPIA dataset.
- `src/midi_processor.py`: Converts MIDI files to token sequences and vice versa.
- `src/model.py`: Defines the Transformer architecture.
- `src/train.py`: Main training loop.
- `src/generate.py`: Inference script for music generation.
- `checkpoints/`: Stores trained model weights.
- `outputs/`: Stores generated MIDI and MP3 files.

## Technical Details

- **Tokenization**: Event-based representation (Pitch, Duration, TimeShift).
- **Model**: Transformer Encoder (acting as Decoder with causal mask).
- **Conditioning**: Emotion tokens are prepended to the sequence `[BOS, EMOTION, ... music ...]`.
- **Generation**: Nucleus Sampling (Top-p) to ensure diversity and quality.
