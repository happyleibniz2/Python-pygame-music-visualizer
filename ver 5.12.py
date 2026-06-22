import pygame
import numpy as np
import json
import soundfile as sf
import math
import time
import random
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import io

# ------------------------
# METADATA SUPPORT
# ------------------------
try:
    from mutagen.flac import FLAC
    HAS_MUTAGEN = True
    print("Mutagen found: FLAC metadata support enabled")
except:
    HAS_MUTAGEN = False
    print("Mutagen not found: FLAC metadata support disabled")

# ------------------------
# CONFIG 
# ------------------------
AUDIO_FILE = None
VIZ_JSON = "scene2.json"

ISPHONK = True

WIDTH, HEIGHT = 900, 900
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)


BUTTON_RECT = pygame.Rect(10, 10, 140, 36)
COVER_BUTTON_RECT = pygame.Rect(10, 56, 140, 36)
SLIDER_RECT = pygame.Rect(10, 122, 140, 10)

BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER_COLOR = (80, 80, 110)
BUTTON_TEXT_COLOR = (255, 255, 255)

# ------------------------
# global vars
# ------------------------
data = None
samplerate = None
channels = 1
sound = None
title = ""
artist = ""
sample_count = 240
radius = 200

# control stuff
volume = 0.5
slider_dragging = False

# album stuff
original_cover_pil = None
cover_surface_scaled = None
cover_available = False

# status stuff
smooth_bars_left = None
smooth_bars_right = None
smooth_beat = 0.0
rotation = 0.0
shake_strength = 0.0
current_shake = np.array([0.0, 0.0]) # 新增：用于平滑震动
prev_low_energy = 0.0
prev_high_energy = 0.0
kick_strength = 0.0
cowbell_strength = 0.0
kick_energy_history = None
cowbell_energy_history = None
kick_cooldown = 0
cowbell_cooldown = 0
start_time = 0.0


nyquist = None
CHUNK_SIZE = 1024
reduced_bin_width = None
kick_bins = None
cowbell_bins = None

COOLDOWN_FRAMES = 5

font_title = None
font_artist = None
font_button = None
font_hint = None

# ------------------------
# func
# ------------------------
def get_font(size):
    fonts = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "MS Gothic", "Arial Unicode MS",
        "Segoe UI", None
    ]
    for name in fonts:
        try:
            f = pygame.font.SysFont(name, size)
            if f:
                return f
        except:
            pass
    return pygame.font.Font(None, size)

def update_fonts():
    global font_title, font_artist, font_hint, font_button
    title_size = max(24, min(72, HEIGHT // 20))
    artist_size = max(16, min(48, HEIGHT // 30))
    hint_size = max(20, min(56, HEIGHT // 25))
    button_size = 20

    font_title = get_font(title_size)
    font_artist = get_font(artist_size)
    font_hint = get_font(hint_size)
    font_button = get_font(button_size)

def load_metadata(filepath):
    t, a = None, None
    if HAS_MUTAGEN:
        try:
            meta = FLAC(filepath)
            t = meta.get("title", [None])[0]
            a = meta.get("artist", [None])[0]
        except:
            pass
    if not t:
        t = os.path.splitext(os.path.basename(filepath))[0]
    if not a:
        a = "Unknown Artist"
    return t, a

def extract_cover(filepath):
    if not HAS_MUTAGEN:
        return None
    try:
        if filepath.lower().endswith('.flac'):
            audio = FLAC(filepath)
            pics = audio.pictures
            if pics:
                return Image.open(io.BytesIO(pics[0].data))
        elif filepath.lower().endswith(('.mp3', '.m4a')):
            try:
                from mutagen.id3 import ID3
                tags = ID3(filepath)
                for tag in tags.values():
                    if tag.FrameID == 'APIC':
                        return Image.open(io.BytesIO(tag.data))
            except:
                pass
            try:
                from mutagen.mp4 import MP4
                tags = MP4(filepath)
                if 'covr' in tags:
                    return Image.open(io.BytesIO(tags['covr'][0]))
            except:
                pass
    except Exception as e:
        print(f"Cover extraction failed: {e}")
    return None

def pil_to_pygame(pil_image):
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    if mode == 'RGB':
        return pygame.image.fromstring(data, size, 'RGB')
    elif mode == 'RGBA':
        return pygame.image.fromstring(data, size, 'RGBA')
    else:
        pil_image = pil_image.convert('RGB')
        return pygame.image.fromstring(pil_image.tobytes(), size, 'RGB')

def prepare_cover_background(pil_image, target_width, target_height):
    img_ratio = pil_image.width / pil_image.height
    win_ratio = target_width / target_height
    if img_ratio > win_ratio:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    pil_image = pil_image.crop((left, top, left + target_width, top + target_height))

    pil_image = pil_image.point(lambda p: p * 0.4)  # 调暗
    return pil_to_pygame(pil_image)

def update_background_surface():
    global cover_surface_scaled, original_cover_pil
    if original_cover_pil:
        cover_surface_scaled = prepare_cover_background(original_cover_pil, WIDTH, HEIGHT)

def reload_audio(new_file):
    global AUDIO_FILE, data, samplerate, channels, sound
    global title, artist
    global smooth_bars_left, smooth_bars_right
    global kick_energy_history, cowbell_energy_history
    global nyquist, reduced_bin_width, kick_bins, cowbell_bins
    global start_time, sample_count
    global cover_available, original_cover_pil, cover_surface_scaled
    global current_shake

    AUDIO_FILE = new_file
    print(f"Loading: {AUDIO_FILE}")

    title, artist = load_metadata(AUDIO_FILE)

    original_cover_pil = extract_cover(AUDIO_FILE)
    if original_cover_pil:
        cover_available = True
        update_background_surface()
        print("Cover art loaded.")
    else:
        cover_available = False
        cover_surface_scaled = None
        print("No cover art found.")

    data, samplerate = sf.read(AUDIO_FILE)
    if len(data.shape) == 1:
        data = np.column_stack((data, data))
    channels = data.shape[1]

    pygame.mixer.quit()
    pygame.mixer.init(frequency=samplerate, size=-16, channels=2, buffer=2048)
    sound = pygame.mixer.Sound(AUDIO_FILE)
    sound.set_volume(volume)
    sound.play()

    smooth_bars_left = np.zeros(sample_count)
    smooth_bars_right = np.zeros(sample_count)
    kick_energy_history = deque(maxlen=43)
    cowbell_energy_history = deque(maxlen=43)
    current_shake = np.array([0.0, 0.0]) # Reset shake
    start_time = time.time()

    nyquist = samplerate / 2
    reduced_bin_width = nyquist / sample_count

    kick_bin_start = max(0, int(40 / reduced_bin_width))
    kick_bin_end = min(sample_count - 1, int(120 / reduced_bin_width))
    kick_bins = slice(kick_bin_start, kick_bin_end + 1)

    cowbell_bin_start = max(0, int(2000 / reduced_bin_width))
    cowbell_bin_end = min(sample_count - 1, int(5000 / reduced_bin_width))
    cowbell_bins = slice(cowbell_bin_start, cowbell_bin_end + 1)

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.flac *.wav *.mp3 *.ogg"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def open_image_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Cover Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def get_fft(chunk):
    window = np.hanning(len(chunk))
    fft = np.fft.rfft(chunk * window)
    return np.abs(fft)

# ------------------------
# load
# ------------------------
try:
    with open(VIZ_JSON, "r", encoding="utf-8") as f:
        viz = json.load(f)
    audio_provider = viz["compositions"][0]["elements"][0]
    raw = audio_provider.get("sampleOutCount", 240)
    if isinstance(raw, dict):
        sample_count = int(float(raw.get("v", 240)))
    else:
        sample_count = int(float(raw))
    sample_count = max(8, min(sample_count, 512))
except:
    print(f"Could not read sampleOutCount from {VIZ_JSON}, using default {sample_count}.")

# ------------------------
# PYGAME init
# ------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Phonk Visualizer")
clock = pygame.time.Clock()

update_fonts()

# ------------------------
# main loop
# ------------------------
running = True
button_hover = False
cover_button_hover = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h
            CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            if cover_available and original_cover_pil:
                update_background_surface()
            radius = min(WIDTH, HEIGHT) * 0.22
            update_fonts()
            
        elif event.type == pygame.MOUSEMOTION:
            button_hover = BUTTON_RECT.collidepoint(event.pos)
            cover_button_hover = COVER_BUTTON_RECT.collidepoint(event.pos)
            if slider_dragging:
                rel_x = event.pos[0] - SLIDER_RECT.left
                volume = max(0.0, min(1.0, rel_x / SLIDER_RECT.width))
                if sound:
                    sound.set_volume(volume)
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BUTTON_RECT.collidepoint(event.pos):
                new_file = open_file_dialog()
                if new_file:
                    reload_audio(new_file)
            elif COVER_BUTTON_RECT.collidepoint(event.pos):
                new_img = open_image_dialog()
                if new_img:
                    try:
                        original_cover_pil = Image.open(new_img)
                        cover_available = True
                        update_background_surface()
                    except Exception as e:
                        print(f"Failed to load image: {e}")
            else:
                thumb_x = SLIDER_RECT.left + int(volume * SLIDER_RECT.width)
                thumb_rect = pygame.Rect(thumb_x - 10, SLIDER_RECT.centery - 10, 20, 20)
                if thumb_rect.collidepoint(event.pos) or SLIDER_RECT.collidepoint(event.pos):
                    slider_dragging = True
                    rel_x = event.pos[0] - SLIDER_RECT.left
                    volume = max(0.0, min(1.0, rel_x / SLIDER_RECT.width))
                    if sound:
                        sound.set_volume(volume)

    # background stuffs
    if cover_available and cover_surface_scaled:
        if data is not None:
            beat_factor = 1.0 + smooth_beat * 0.05
            scaled_w = int(WIDTH * beat_factor)
            scaled_h = int(HEIGHT * beat_factor)
            offset_x = (WIDTH - scaled_w) // 2
            offset_y = (HEIGHT - scaled_h) // 2
            bg_dynamic = pygame.transform.scale(cover_surface_scaled, (scaled_w, scaled_h))
            screen.blit(bg_dynamic, (offset_x, offset_y))
        else:
            screen.blit(cover_surface_scaled, (0, 0))
    else:
        screen.fill((0, 0, 0))

    # UI stuff
    btn_color = BUTTON_HOVER_COLOR if button_hover else BUTTON_COLOR
    pygame.draw.rect(screen, btn_color, BUTTON_RECT, border_radius=8)
    pygame.draw.rect(screen, (150, 150, 180), BUTTON_RECT, width=2, border_radius=8)
    btn_text = font_button.render(" Open File", True, BUTTON_TEXT_COLOR)
    screen.blit(btn_text, btn_text.get_rect(center=BUTTON_RECT.center))

    c_btn_color = BUTTON_HOVER_COLOR if cover_button_hover else BUTTON_COLOR
    pygame.draw.rect(screen, c_btn_color, COVER_BUTTON_RECT, border_radius=8)
    pygame.draw.rect(screen, (150, 150, 180), COVER_BUTTON_RECT, width=2, border_radius=8)
    c_btn_text = font_button.render("Add Cover", True, BUTTON_TEXT_COLOR)
    screen.blit(c_btn_text, c_btn_text.get_rect(center=COVER_BUTTON_RECT.center))

    vol_text = font_button.render(f"Vol: {int(volume * 100)}%", True, BUTTON_TEXT_COLOR)
    screen.blit(vol_text, (SLIDER_RECT.left, SLIDER_RECT.top - 22))
    pygame.draw.line(screen, (100, 100, 100), (SLIDER_RECT.left, SLIDER_RECT.centery), (SLIDER_RECT.right, SLIDER_RECT.centery), 6)
    fill_width = int(volume * SLIDER_RECT.width)
    if fill_width > 0:
        pygame.draw.line(screen, (150, 150, 255), (SLIDER_RECT.left, SLIDER_RECT.centery), (SLIDER_RECT.left + fill_width, SLIDER_RECT.centery), 6)
    thumb_x = SLIDER_RECT.left + fill_width
    pygame.draw.circle(screen, (200, 200, 255) if slider_dragging else (255, 255, 255), (thumb_x, SLIDER_RECT.centery), 8)

    # proc
    if data is None:
        hint_surf = font_hint.render("No audio loaded — Click 'Open File'", True, (180, 180, 180))
        screen.blit(hint_surf, hint_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
    else:
        t = time.time() - start_time
        idx = int(t * samplerate)

        if idx + CHUNK_SIZE < len(data):
            chunk_left = data[idx:idx + CHUNK_SIZE, 0]
            chunk_right = data[idx:idx + CHUNK_SIZE, 1] if channels >= 2 else chunk_left.copy()

            fft_left = get_fft(chunk_left)
            fft_right = get_fft(chunk_right)

            fft_left = fft_left[:len(fft_left) - (len(fft_left) % sample_count)]
            fft_right = fft_right[:len(fft_right) - (len(fft_right) % sample_count)]
            bars_left = fft_left.reshape(sample_count, -1).mean(axis=1)
            bars_right = fft_right.reshape(sample_count, -1).mean(axis=1)

            max_left = np.max(bars_left)
            if max_left > 1e-6: bars_left /= max_left
            max_right = np.max(bars_right)
            if max_right > 1e-6: bars_right /= max_right

            smooth_bars_left[:] = smooth_bars_left * 0.85 + bars_left * 0.15
            smooth_bars_right[:] = smooth_bars_right * 0.85 + bars_right * 0.15

            bars_avg = (bars_left + bars_right) * 0.5
            kick_raw = float(np.clip(np.mean(bars_avg[kick_bins]), 0.0, 1.0))
            cowbell_raw = float(np.clip(np.mean(bars_avg[cowbell_bins]), 0.0, 1.0))

            kick_energy_history.append(kick_raw)
            cowbell_energy_history.append(cowbell_raw)

            kick_mean = np.mean(kick_energy_history) if len(kick_energy_history) > 10 else 0.05
            kick_std = np.std(kick_energy_history) if len(kick_energy_history) > 10 else 0.01

            cowbell_mean = np.mean(cowbell_energy_history) if len(cowbell_energy_history) > 10 else 0.05
            cowbell_std = np.std(cowbell_energy_history) if len(cowbell_energy_history) > 10 else 0.01

            kick_onset = False
            if kick_cooldown == 0:
                kick_delta = kick_raw - prev_low_energy
                if kick_raw > (kick_mean + kick_std * 1.5) + 0.02 and kick_delta > 0.05:
                    kick_onset = True
                    kick_cooldown = COOLDOWN_FRAMES
            else:
                kick_cooldown -= 1

            cowbell_onset = False
            if cowbell_cooldown == 0:
                cowbell_delta = cowbell_raw - prev_high_energy
                if cowbell_raw > (cowbell_mean + cowbell_std * 1.8) + 0.02 and cowbell_delta > 0.05:
                    cowbell_onset = True
                    cowbell_cooldown = COOLDOWN_FRAMES
            else:
                cowbell_cooldown -= 1

            prev_low_energy = kick_raw
            prev_high_energy = cowbell_raw

            if kick_onset:
                kick_strength = 1.0
            else:
                kick_strength *= 0.85

            if cowbell_onset:
                cowbell_strength = 1.0
            else:
                cowbell_strength *= 0.80

            # --- TWEAK SHAKE STRENGTH HERE ---
            # Reduced from 35/25 down to 15/10 for a much calmer bounce
            shake_strength = kick_strength * 15 + cowbell_strength * 10
            
            smooth_beat = smooth_beat * 0.9 + kick_raw * 0.1
            rotation += 0.002 + smooth_beat * 0.006

            left_energy = np.mean(smooth_bars_left)
            right_energy = np.mean(smooth_bars_right)
            stereo_imbalance = (left_energy - right_energy) * (WIDTH * 0.09)

            if ISPHONK:
                target_shake = np.array([
                    random.uniform(-shake_strength, shake_strength),
                    random.uniform(-shake_strength, shake_strength)
                ])
                # Smooth the jitter out! (60% previous position, 40% target position)
                current_shake = current_shake * 0.6 + target_shake * 0.4
                
                stereo_offset = np.array([stereo_imbalance, 0.0])
                shake_offset = current_shake + stereo_offset
            else:
                shake_offset = np.array([0, 0])

            center = CENTER + shake_offset

            for i, (valL, valR) in enumerate(zip(smooth_bars_left, smooth_bars_right)):
                angle = (i / sample_count) * 2 * math.pi + rotation
                mix = 0.5 + 0.5 * math.cos(angle)
                val = valL if mix > 0.5 else valR

                r1 = radius
                r2 = radius + val * (min(WIDTH, HEIGHT) * 0.27)
                x1 = center[0] + math.cos(angle) * r1
                y1 = center[1] + math.sin(angle) * r1
                x2 = center[0] + math.cos(angle) * r2
                y2 = center[1] + math.sin(angle) * r2

                hue = i / sample_count
                r = int(127 + 127 * math.sin(hue * 6.283))
                g = int(127 + 127 * math.sin(hue * 6.283 + 2))
                b = int(127 + 127 * math.sin(hue * 6.283 + 4))

                if mix > 0.5:
                    brightness = 1.0 + (valR - valL) * 0.5
                else:
                    brightness = 1.0 + (valL - valR) * 0.5
                r = min(255, int(r * brightness))
                g = min(255, int(g * brightness))
                b = min(255, int(b * brightness))

                pygame.draw.line(screen, (r, g, b), (x1, y1), (x2, y2), max(1, int(min(WIDTH, HEIGHT) / 450)))

            # pulse circle
            pulse_radius = int(radius * 0.3 + smooth_beat * (min(WIDTH, HEIGHT) * 0.09))
            pulse_radius = max(radius * 0.2, min(pulse_radius, radius * 0.7))
            for i in range(3):
                alpha = int(60 / (i + 1))
                surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255, 255, 255, alpha),
                                   center.astype(int), pulse_radius + i * 10, 2)
                screen.blit(surf, (0, 0))

            title_surf = font_title.render(title, True, (255, 255, 255))
            artist_surf = font_artist.render(artist, True, (180, 180, 180))
            screen.blit(title_surf, title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - int(HEIGHT * 0.03))))
            screen.blit(artist_surf, artist_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + int(HEIGHT * 0.03))))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
