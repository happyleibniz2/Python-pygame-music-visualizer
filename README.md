
---

# Python Pygame Music Visualizer

A real-time audio visualizer built with Python using Pygame.
This project focuses on bringing an **Avee Player–style visualization system to desktop environments**, allowing a PC to run similar visual effects without relying on mobile applications.

---

## Project Goal

The main purpose of this project is to replicate key ideas from Avee Player in a custom Python-based environment:

* Real-time audio-reactive visuals
* Dynamic effects synchronized to music
* Flexible rendering system for experimentation
* Full control over visuals without closed-source limitations

This is not a direct clone, but an independent implementation inspired by similar visual behavior.

---

## Preview

### Letter Mode
[![Letter Mode](eyecandy/lm.png)](eyecandy/letter%20mode.mp4)

### Normal + Phonk Mode
[![Normal + Phonk Mode](eyecandy/nmapm.png)](eyecandy/normal%20mode%20and%20phonk%20mode.mp4)


## Features

### Audio Processing

* Real-time FFT using NumPy
* Logarithmic frequency band mapping (80 Hz – 18 kHz)
* Smooth interpolation and decay system
* Optional wave-based distortion

---

### Beat Detection

* Multi-band analysis:

  * sub-bass
  * bass
  * phonk kick
  * low-mid / high-mid / high
  * cowbell detection
* Rolling history buffers for adaptive thresholds
* Beat confidence scoring
* Phonk intensity scaling

---

### Visualization Modes

#### Circle Visualizer

* Radial spectrum bars
* Inner and outer rendering
* Glow layering
* Beat-synchronized pulse

#### Letter Visualizer

* Uses first letter of track title
* Bars emitted from outline
* Geometry-based extrusion

---

### Effects System

* Shake (continuous[bug] or beat-driven)
* Wave (experimental)
* Multi-layer glow rendering
* Particle system triggered by beats
* Rotation and pulse animation

---

### Phonk Mode

* Alternate color scheme
* Increased shake intensity
* More aggressive particle effects
* Additional distortion and rotation
* Cowbell band indicator

---

### Media Integration

Using Mutagen and Pillow:

* Extracts:

  * Artist
  * Cover art
* Supported formats:

  * MP3 (ID3)
  * FLAC
  * M4A / MP4 / AAC
* Background system:

  * Dimmed cover rendering
  * Beat-reactive brightness and zoom

---

### User Interface

* Built-in UI system
* File selection dialog
* Cover image loader
* Artist override input
* Mode toggles:

  * Avee mode
  * Letter mode
  * Shake / Beat shake
  * Wave bars
  * Phonk mode
  * Background loading
* Volume slider
* Toggle UI with SPACE

---

## Requirements

```bash
pip install pygame numpy soundfile pillow mutagen
```

---

## Running (latest version)

```bash
python "ver 6 beta test 2.py"
```

---

## Controls

| Input  | Action              |
| ------ | ------------------- |
| Mouse  | UI interaction      |
| SPACE  | Toggle UI           |
| OPEN   | Load audio file     |
| COVER  | Load background     |
| ARTIST | Set artist manually |

---

## How It Works

### Audio Pipeline

1. Load audio using soundfile
2. Convert to mono
3. Process in 2048-sample windows
4. Apply Hann window
5. Perform FFT
6. Map frequencies into logarithmic bands

---

### Beat Detection

* Tracks energy per frequency band
* Compares current vs historical values
* Detects spikes using ratio thresholds
* Combines bands into a confidence score

---

### Rendering Pipeline

* Background (image or fallback)
* Glow layer (alpha surface)
* Visualizer (circle or letter)
* Particle system
* UI overlay

---

## Latest File

```
ver 6 beta test 2.py
```

---

## Known Issues

* High CPU usage (FFT + rendering)
* Font compatibility may vary across systems
* Tkinter dialogs may briefly pause rendering
* track name and artist test collide with tip.
* background can only be loaded after enabling `BG LOAD` AND reopening the music file
* background loading lags the program

---

## Planned Improvements

* Improved beat detection
* Preset system (Avee-style configs)
* Video export support
* Performance optimization

---

## Author

GitHub: [https://github.com/happyleibniz2](https://github.com/happyleibniz2)

---
