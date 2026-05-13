import pygame
import numpy as np
import soundfile as sf
import math
import time
import random
import os
import io
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, APIC
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

# ------------------------
# CONFIG
# ------------------------
SAMPLE_COUNT = 180
WIDTH, HEIGHT = 900, 900
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)


# --- UI Layout: Two columns ---
BUTTON_WIDTH = 110
BUTTON_HEIGHT = 30
BUTTON_MARGIN = 8
BUTTON_COLS = 2
BUTTON_RECTS = {
    "open":       pygame.Rect(10, 10, BUTTON_WIDTH, BUTTON_HEIGHT),
    "cover":      pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 1, BUTTON_WIDTH, BUTTON_HEIGHT),
    "artist":     pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 2, BUTTON_WIDTH, BUTTON_HEIGHT),
    "avee":       pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 3, BUTTON_WIDTH, BUTTON_HEIGHT),
    "letter":     pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 4, BUTTON_WIDTH, BUTTON_HEIGHT),
    "shake":      pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 5, BUTTON_WIDTH, BUTTON_HEIGHT),
    "beat_shake": pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 6, BUTTON_WIDTH, BUTTON_HEIGHT),
    "wave_bars":  pygame.Rect(10, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 7, BUTTON_WIDTH, BUTTON_HEIGHT),
    "phonk":      pygame.Rect(10 + BUTTON_WIDTH + BUTTON_MARGIN, 10, BUTTON_WIDTH, BUTTON_HEIGHT),
    "bg_load":    pygame.Rect(10 + BUTTON_WIDTH + BUTTON_MARGIN, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 1, BUTTON_WIDTH, BUTTON_HEIGHT),
    "genre":      pygame.Rect(10 + BUTTON_WIDTH + BUTTON_MARGIN, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 2, BUTTON_WIDTH, BUTTON_HEIGHT),
    "rgb":        pygame.Rect(10 + BUTTON_WIDTH + BUTTON_MARGIN, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 3, BUTTON_WIDTH, BUTTON_HEIGHT),
}
SLIDER_RECT = pygame.Rect(10 + BUTTON_WIDTH + BUTTON_MARGIN, 10 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 4, BUTTON_WIDTH, 10)


# ------------------------
# PRECOMPUTED CONSTANTS
# ------------------------
WINDOW = np.hanning(2048)
f_min, f_max = 80.0, 18000.0
LOG_EDGES = np.logspace(np.log10(f_min), np.log10(f_max), SAMPLE_COUNT + 1)

BEAT_BANDS = {
    'sub_bass':   (0, 3),
    'bass':       (3, 8),
    'phonk_kick': (4, 7),
    'low_mid':    (8, 15),
    'cowbell':    (25, 35),
    'high_mid':   (15, 30),
    'high':       (30, 60),
}
beat_energy_history = {band: [] for band in BEAT_BANDS}
HISTORY_SIZE = 43


# ------------------------
# GLOBAL STATE
# ------------------------
data = None
samplerate = None
sound = None
title, artist = "No Track", "Unknown"
volume = 1.0
avee_mode = False
letter_mode = False
shake_mode = False
beat_shake_mode = False
wave_bars_mode = False
phonk_mode = False
bg_load_enabled = False
show_ui = True
slider_dragging = False
start_time = 0

original_cover = None
bg_surface = None
original_cover_pil = None

bar_falloff = np.zeros(SAMPLE_COUNT)
peak_hold = np.zeros(SAMPLE_COUNT)
peak_timer = np.zeros(SAMPLE_COUNT)
smooth_bars = np.zeros(SAMPLE_COUNT)
wave_bars = np.zeros(SAMPLE_COUNT)

rotation = 0.0
smooth_beat = 0.0
pulse_radius = 0.0
tide_phase = 0.0
wave_phase = 0.0
shake_current = np.array([0.0, 0.0])
shake_target = np.array([0.0, 0.0])
letter_points = []
letter_centroid = (0.0, 0.0)

_last_beat_time = 0
beat_confidence = 0.0
cowbell_energy = 0.0
phonk_intensity = 0.0

particles = []

_font_cache = {}
_cached_title_surf = None
_cached_artist_surf = None
_cached_title_text = ""
_cached_artist_text = ""

glow_surf = None

# --- GENRE PANEL STATE ---
active_genres = set()
genre_panel_open = False
genre_auto_rect = pygame.Rect(0, 0, 0, 0)
genre_close_rect = pygame.Rect(0, 0, 0, 0)
genre_checkbox_rects = []
GENRE_LIST = [
    ("Breakcore", (120,255,120)),
    ("Speedcore", (255,255,255)),
    ("Hardcore", (255,180,60)),
    ("Frenchcore", (120,180,255)),
    ("Uptempo", (255,80,180)),
]

# --- LETTER RGB TRAIL EFFECT (Avee Player style) ---
letter_trail_mode = False  # RGB 按钮控制这个
last_trail_time = 0

# ------------------------
# CJK FONT DETECTION
# ------------------------
_CJK_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
_cjk_font_path = None
for _p in _CJK_FONT_PATHS:
    if os.path.exists(_p):
        _cjk_font_path = _p
        break

# ------------------------
# HELPERS
# ------------------------
def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        font = None
        if _cjk_font_path:
            try:
                font = pygame.font.Font(_cjk_font_path, size)
            except Exception:
                font = None
        if font is None:
            font = pygame.font.SysFont("Segoe UI,Arial,Helvetica", size, bold=bold)
        _font_cache[key] = font
    return _font_cache[key]

def ensure_title_surfaces():
    global _cached_title_surf, _cached_artist_surf, _cached_title_text, _cached_artist_text
    
    if _cached_title_surf is None or title != _cached_title_text:
        t_font = get_font(34, True)
        _cached_title_surf = t_font.render(title[:28], True, (255, 255, 255))
        _cached_title_text = title
    
    if _cached_artist_surf is None or artist != _cached_artist_text:
        a_font = get_font(19)
        a_color = (255, 100, 200) if phonk_mode else (140, 200, 255)
        _cached_artist_surf = a_font.render(artist, True, a_color)
        _cached_artist_text = artist

def draw_glow_line(surface, color, p1, p2, width=2, layers=3):
    x1, y1 = p1
    x2, y2 = p2
    for layer in range(layers, 0, -1):
        alpha = int(60 / layer)
        w = width + layer * 2
        r, g, b = color
        pygame.draw.line(surface, (r, g, b, alpha), (x1, y1), (x2, y2), w)
    pygame.draw.line(surface, color, (x1, y1), (x2, y2), width)

def draw_glow_circle(surface, color, center, radius, width=2):
    for layer in range(3, 0, -1):
        alpha = int(50 / layer)
        r2 = radius + layer * 3
        pygame.draw.circle(surface, (*color, alpha), (int(center[0]), int(center[1])), int(r2), width + layer)
    pygame.draw.circle(surface, color, (int(center[0]), int(center[1])), int(radius), width)

def bar_color_avee(i, n, intensity=1.0):
    r = int(min(255, 80 + 175 * intensity))
    g = int(min(255, 200 + 55 * intensity))
    b = int(255)
    return (r, g, b)

def bar_color_phonk(i, n, intensity=1.0):
    r = int(min(255, 180 + 75 * intensity))
    g = int(min(255, 20 + 40 * intensity))
    b = int(min(255, 200 + 55 * intensity))
    return (r, g, b)

def bar_color_letter(i, n, intensity=1.0):
    v = int(min(255, 180 + 75 * intensity))
    return (v, v, 255)

def generate_letter_points(char):
    global letter_points, letter_centroid
    size = int(min(WIDTH, HEIGHT) * 0.68)
    font = get_font(size, bold=True)
    surf = font.render(char.upper(), True, (255, 255, 255))
    mask = pygame.mask.from_surface(surf)
    full_outline = mask.outline()
    if not full_outline:
        letter_points = []
        return
    step = max(1, len(full_outline) // SAMPLE_COUNT)
    outline = mask.outline(step)
    rect = surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    points = []
    for i in range(SAMPLE_COUNT):
        idx = int(i * len(outline) / SAMPLE_COUNT)
        p = outline[idx]
        points.append((p[0] + rect.x, p[1] + rect.y))
    letter_points = points
    letter_centroid = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )

def prepare_bg(pil_img):
    global original_cover_pil
    max_dim = 720
    scale_down = max(pil_img.width, pil_img.height) / max_dim
    if scale_down > 1:
        new_w = int(pil_img.width / scale_down)
        new_h = int(pil_img.height / scale_down)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    scale = max(WIDTH / pil_img.width, HEIGHT / pil_img.height)
    new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    left, top = (pil_img.width - WIDTH) // 2, (pil_img.height - HEIGHT) // 2
    cropped = pil_img.crop((left, top, left + WIDTH, top + HEIGHT))
    original_cover_pil = cropped
    dimmed = cropped.point(lambda p: int(p * 0.25))
    return pygame.image.fromstring(dimmed.tobytes(), dimmed.size, dimmed.mode).convert()

def make_beat_bg(beat_strength):
    if original_cover_pil is None:
        return None
    brightness = 0.22 + beat_strength * 0.20
    arr = np.array(original_cover_pil, dtype=np.float32)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    lit = Image.fromarray(arr)
    
    zoom = 1.0 + beat_strength * 0.04
    if zoom > 1.001:
        zw, zh = int(WIDTH * zoom), int(HEIGHT * zoom)
        lit = lit.resize((zw, zh), Image.Resampling.BILINEAR)
        ox, oy = (zw - WIDTH) // 2, (zh - HEIGHT) // 2
        lit = lit.crop((ox, oy, ox + WIDTH, oy + HEIGHT))
    
    return pygame.image.fromstring(lit.tobytes(), lit.size, lit.mode).convert()

def extract_cover(file):
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".mp3":
            tags = ID3(file)
            for tag in tags.values():
                if isinstance(tag, APIC):
                    return Image.open(io.BytesIO(tag.data)).convert("RGB")
        elif ext == ".flac":
            audio = FLAC(file)
            if audio.pictures:
                return Image.open(io.BytesIO(audio.pictures[0].data)).convert("RGB")
        elif ext in (".m4a", ".mp4", ".aac"):
            audio = MP4(file)
            if "covr" in audio:
                return Image.open(io.BytesIO(bytes(audio["covr"][0]))).convert("RGB")
        else:
            audio = MutagenFile(file)
            if audio and hasattr(audio, "pictures") and audio.pictures:
                return Image.open(io.BytesIO(audio.pictures[0].data)).convert("RGB")
    except Exception:
        pass
    return None

def extract_artist(file):
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".mp3":
            tags = ID3(file)
            for key in ("TPE1", "TPE2", "TCOM"):
                if key in tags:
                    return str(tags[key])
        elif ext == ".flac":
            audio = FLAC(file)
            if "artist" in audio:
                return audio["artist"][0]
        elif ext in (".m4a", ".mp4", ".aac"):
            audio = MP4(file)
            if "\xa9ART" in audio:
                return audio["\xa9ART"][0]
        else:
            audio = MutagenFile(file)
            if audio:
                for key in ("artist", "ARTIST", "TPE1"):
                    if key in audio:
                        val = audio[key]
                        return str(val[0]) if isinstance(val, list) else str(val)
    except Exception:
        pass
    return None

def reload_audio(file):
    global data, samplerate, sound, title, artist, start_time, bg_surface
    global original_cover_pil, _cached_title_surf, _cached_artist_surf, _cached_title_text, _cached_artist_text
    
    data, samplerate = sf.read(file)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    pygame.mixer.quit()
    pygame.mixer.init(samplerate, -16, 2, 2048)
    sound = pygame.mixer.Sound(file)
    sound.set_volume(volume)
    sound.play()
    title = os.path.splitext(os.path.basename(file))[0]
    start_time = time.time()
    generate_letter_points(title[0] if title else "A")
    
    artist = "Unknown"
    bg_surface = None
    original_cover_pil = None
    _cached_title_surf = None
    _cached_artist_surf = None
    _cached_title_text = ""
    _cached_artist_text = ""
    
    for key in beat_energy_history:
        beat_energy_history[key] = []
    
    found_artist = extract_artist(file)
    if found_artist:
        artist = found_artist
    
    if bg_load_enabled:
        cover = extract_cover(file)
        if cover:
            bg_surface = prepare_bg(cover)

def auto_detect_genres():
    """Analyze FFT energy distribution and suggest genres."""
    global active_genres, data, samplerate, start_time
    if data is None or samplerate is None:
        return
    idx = int((time.time() - start_time) * samplerate)
    if idx + 2048 >= len(data):
        return
    chunk = data[idx:idx + 2048]
    fft_raw = np.abs(np.fft.rfft(chunk * WINDOW))
    freqs = np.fft.rfftfreq(2048, d=1.0 / samplerate)
    fft = np.zeros(SAMPLE_COUNT)
    for b in range(SAMPLE_COUNT):
        lo = np.searchsorted(freqs, LOG_EDGES[b])
        hi = np.searchsorted(freqs, LOG_EDGES[b + 1])
        if hi > lo:
            fft[b] = np.mean(fft_raw[lo:hi])
        elif lo < len(fft_raw):
            fft[b] = fft_raw[lo]
    mx_val = np.max(fft)
    if mx_val > 1e-6:
        fft /= mx_val
    
    low = np.mean(fft[:SAMPLE_COUNT//8])
    mid = np.mean(fft[SAMPLE_COUNT//8:SAMPLE_COUNT//2])
    high = np.mean(fft[SAMPLE_COUNT//2:])
    midhigh = np.mean(fft[SAMPLE_COUNT//2:int(SAMPLE_COUNT*0.85)])
    allbands = np.mean(fft)
    
    genres = set()
    if high > 0.55 and midhigh > 0.45:
        genres.add("Breakcore")
    if allbands > 0.45 and low > 0.35 and high > 0.35:
        genres.add("Speedcore")
    if low > 0.55 and low > mid and low > high:
        genres.add("Hardcore")
    if midhigh > 0.5 and midhigh > high and midhigh > low:
        genres.add("Frenchcore")
    if low > 0.7 and mid < 0.3 and high < 0.3:
        genres.add("Uptempo")
    
    active_genres.clear()
    active_genres.update(genres)

# ------------------------
# BEAT DETECTION
# ------------------------
def detect_beat_advanced(fft, smooth_beat):
    global _last_beat_time, beat_confidence, cowbell_energy, phonk_intensity
    
    on_beat = False
    beat_confidence = 0.0
    cowbell_energy = 0.0
    
    for band_name, (lo, hi) in BEAT_BANDS.items():
        band_energy = np.mean(fft[lo:min(hi, len(fft))])
        
        history = beat_energy_history[band_name]
        history.append(band_energy)
        if len(history) > HISTORY_SIZE:
            history.pop(0)
        
        if len(history) < 10:
            continue
        
        avg_energy = np.mean(history[:-1])
        
        if avg_energy > 1e-6:
            ratio = band_energy / avg_energy
            
            if band_name == 'phonk_kick' and ratio > 1.4:
                beat_confidence += 0.6 * ratio
                phonk_intensity = min(1.0, phonk_intensity + 0.3)
            elif band_name == 'sub_bass' and ratio > 1.6:
                beat_confidence += 0.4 * ratio
                phonk_intensity = min(1.0, phonk_intensity + 0.2)
            elif band_name == 'bass' and ratio > 1.5:
                beat_confidence += 0.5 * ratio
            elif band_name == 'cowbell' and ratio > 1.7:
                beat_confidence += 0.3 * ratio
                cowbell_energy = band_energy
                phonk_intensity = min(1.0, phonk_intensity + 0.4)
            elif band_name == 'high_mid' and ratio > 1.3:
                beat_confidence += 0.2 * ratio
    
    if len(beat_energy_history['phonk_kick']) >= 3:
        current = beat_energy_history['phonk_kick'][-1]
        prev = beat_energy_history['phonk_kick'][-2]
        prev_prev = beat_energy_history['phonk_kick'][-3]
        
        if current > prev > prev_prev:
            slope = (current - prev_prev) / 2
            if slope > 0.0008:
                beat_confidence += 0.35
    
    beat_confidence = min(beat_confidence / 3.5, 1.0)
    phonk_intensity *= 0.92
    
    current_time = time.time()
    min_interval = 0.10
    
    if beat_confidence > 0.40 and (current_time - _last_beat_time) > min_interval:
        on_beat = True
        _last_beat_time = current_time
        smooth_beat = smooth_beat * 0.65 + beat_confidence * 0.35
    else:
        on_beat = False
        smooth_beat *= 0.88
    
    return on_beat, smooth_beat, beat_confidence

# ------------------------
# PARTICLE SYSTEM
# ------------------------
class Particle:
    def __init__(self, x, y, angle, speed, life, color=(0, 200, 255)):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = life
        self.max_life = life
        self.color = color
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vx *= 0.96
        self.vy *= 0.96
        return self.life > 0
    
    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        r, g, b = self.color
        pygame.draw.circle(surface, (r, g, b, alpha), (int(self.x), int(self.y)), max(1, int(3 * self.life / self.max_life)))

# ------------------------
# GENRE VISUALIZER FUNCTIONS
# ------------------------
def genre_bar_color(i, n, intensity, active_genres_set, breakcore_palette):
    if "Uptempo" in active_genres_set:
        return (180 + int(60 * intensity), 60, 180 + int(60 * intensity))
    if "Breakcore" in active_genres_set:
        return random.choice(breakcore_palette)
    if "Speedcore" in active_genres_set:
        return (255, 255, 255) if i % 2 == 0 else (255, 80, 80)
    if "Hardcore" in active_genres_set:
        return (255, 180, 60 + int(120 * intensity))
    if "Frenchcore" in active_genres_set:
        return (120, 180, 255)
    if phonk_mode:
        return bar_color_phonk(i, n, intensity)
    return bar_color_avee(i, n, intensity)

def draw_circle_visualizer_genre(screen, final_bars, shake, glow_surf, active_genres_set, rot, pulse_r, avee_on, peak_arr):
    base_radius = min(WIDTH, HEIGHT) * 0.22
    cx, cy = CENTER[0] + shake[0], CENTER[1] + shake[1]
    breakcore_palette = [(80, 255, 80), (255, 80, 255), (0, 255, 120), (255, 255, 255)]
    
    for i in range(SAMPLE_COUNT):
        intensity = final_bars[i]
        h_out = intensity * (min(WIDTH, HEIGHT) * 0.28)
        h_in = intensity * (min(WIDTH, HEIGHT) * 0.12)
        angle = (i / SAMPLE_COUNT) * 2 * math.pi + rot
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = cx + cos_a * base_radius
        ry = cy + sin_a * base_radius
        ox = rx + cos_a * h_out
        oy = ry + sin_a * h_out
        ix = rx - cos_a * h_in
        iy = ry - sin_a * h_in
        
        width = 3 + intensity * 4
        if "Hardcore" in active_genres_set:
            width = int(width * 4)
        
        color = genre_bar_color(i, SAMPLE_COUNT, intensity, active_genres_set, breakcore_palette)
        r, g, b = color
        
        for glow_layer in range(3, 0, -1):
            alpha = int(35 / glow_layer)
            glow_w = int(width + glow_layer * 3)
            pygame.draw.line(glow_surf, (r, g, b, alpha), (rx, ry), (ox, oy), glow_w)
        
        pygame.draw.line(screen, color, (rx, ry), (ox, oy), int(width))
        inner_color = (r // 2, g // 2, b)
        pygame.draw.line(screen, inner_color, (rx, ry), (ix, iy), int(width * 0.5))
        
        if avee_on and peak_arr[i] > 0.1:
            ph = peak_arr[i] * (min(WIDTH, HEIGHT) * 0.28)
            pdx = cx + cos_a * (base_radius + ph)
            pdy = cy + sin_a * (base_radius + ph)
            for gl in range(3, 0, -1):
                dot_color = (255, 100, 255)
                pygame.draw.circle(glow_surf, (*dot_color, int(35 / gl)), (int(pdx), int(pdy)), 2 + gl * 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(pdx), int(pdy)), 3)

def draw_letter_visualizer_genre(screen, final_bars, shake, glow_surf, active_genres_set, avee_on, peak_arr, letter_pts, letter_cent):
    if not letter_pts:
        return
    breakcore_palette = [(80, 255, 80), (255, 80, 255), (0, 255, 120), (255, 255, 255)]
    
    if len(letter_pts) > 2:
        ghost_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ghost_color = (255, 50, 200, 18)
        pygame.draw.polygon(ghost_surf, ghost_color, letter_pts)
        screen.blit(ghost_surf, (0, 0))
    
    for i in range(SAMPLE_COUNT):
        intensity = final_bars[i]
        h = intensity * (min(WIDTH, HEIGHT) * 0.20)
        px, py = letter_pts[i]
        prev_p = letter_pts[(i - 1) % SAMPLE_COUNT]
        next_p = letter_pts[(i + 1) % SAMPLE_COUNT]
        tx = next_p[0] - prev_p[0]
        ty = next_p[1] - prev_p[1]
        out_dx = px - letter_cent[0]
        out_dy = py - letter_cent[1]
        n1x, n1y = -ty, tx
        if (n1x * out_dx + n1y * out_dy) < 0:
            n1x, n1y = ty, -tx
        dist = math.sqrt(n1x * n1x + n1y * n1y) or 1
        ndx, ndy = n1x / dist, n1y / dist
        x1 = px + shake[0]
        y1 = py + shake[1]
        ox = x1 + ndx * h
        oy = y1 + ndy * h
        ix = x1 - ndx * (h * 0.4)
        iy = y1 - ndy * (h * 0.4)
        
        color = genre_bar_color(i, SAMPLE_COUNT, intensity, active_genres_set, breakcore_palette)
        r, g, b = color
        
        pygame.draw.line(glow_surf, (r, g, b, 45), (x1, y1), (ox, oy), 9)
        pygame.draw.line(glow_surf, (r, g, b, 20), (x1, y1), (ix, iy), 6)
        
        width = 4
        if "Hardcore" in active_genres_set:
            width = 16
        pygame.draw.line(screen, color, (x1, y1), (ox, oy), width)
        pygame.draw.line(screen, color, (x1, y1), (ix, iy), 2)
        
        if avee_on and peak_arr[i] > 0.01:
            ph = peak_arr[i] * (min(WIDTH, HEIGHT) * 0.20)
            pdx = x1 + ndx * ph
            pdy = y1 + ndy * ph
            pygame.draw.circle(screen, (255, 255, 255), (int(pdx), int(pdy)), 2)

# ------------------------
# DRAWING (Standard)
# ------------------------
def draw_circle_visualizer_enhanced(screen, final_bars, shake, glow_surf):
    base_radius = min(WIDTH, HEIGHT) * 0.22
    cx, cy = CENTER[0] + shake[0], CENTER[1] + shake[1]
    
    for layer in range(4, 0, -1):
        alpha = int(12 / layer)
        r = base_radius + layer * 2
        ring_color = (255, 50, 200) if phonk_mode else (0, 200, 255)
        pygame.draw.circle(glow_surf, (*ring_color, alpha), (int(cx), int(cy)), int(r), 4 + layer)
    
    for i in range(SAMPLE_COUNT):
        intensity = final_bars[i]
        h_out = intensity * (min(WIDTH, HEIGHT) * 0.28)
        h_in = intensity * (min(WIDTH, HEIGHT) * 0.12)
        
        angle = (i / SAMPLE_COUNT) * 2 * math.pi + rotation
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        rx = cx + cos_a * base_radius
        ry = cy + sin_a * base_radius
        ox = rx + cos_a * h_out
        oy = ry + sin_a * h_out
        ix = rx - cos_a * h_in
        iy = ry - sin_a * h_in
        
        line_width = 3 + intensity * 4
        
        if phonk_mode:
            color = bar_color_phonk(i, SAMPLE_COUNT, intensity)
        else:
            color = bar_color_avee(i, SAMPLE_COUNT, intensity)
        
        r, g, b = color
        
        for glow_layer in range(3, 0, -1):
            alpha = int(35 / glow_layer)
            glow_w = int(line_width + glow_layer * 3)
            pygame.draw.line(glow_surf, (r, g, b, alpha), (rx, ry), (ox, oy), glow_w)
        
        pygame.draw.line(screen, color, (rx, ry), (ox, oy), int(line_width))
        
        inner_color = (r//2, g//2, b)
        pygame.draw.line(screen, inner_color, (rx, ry), (ix, iy), int(line_width * 0.5))
        
        if avee_mode and peak_hold[i] > 0.1:
            ph = peak_hold[i] * (min(WIDTH, HEIGHT) * 0.28)
            pdx = cx + cos_a * (base_radius + ph)
            pdy = cy + sin_a * (base_radius + ph)
            for gl in range(3, 0, -1):
                dot_color = (255, 100, 255) if phonk_mode else (255, 255, 255)
                pygame.draw.circle(glow_surf, (*dot_color, int(35/gl)), (int(pdx), int(pdy)), 2 + gl*2)
            pygame.draw.circle(screen, (255, 255, 255), (int(pdx), int(pdy)), 3)

def draw_letter_visualizer_enhanced(screen, final_bars, shake, glow_surf, on_beat):
    global last_trail_time, beat_confidence
    
    if not letter_points:
        return
    
    # --- 先画 Ghost 背景 ---
    if len(letter_points) > 2:
        ghost_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ghost_color = (255, 50, 200, 18) if phonk_mode else (100, 220, 255, 18)
        pygame.draw.polygon(ghost_surf, ghost_color, letter_points)
        screen.blit(ghost_surf, (0, 0))
    
    # --- 正常绘制 letter ---
    for i in range(SAMPLE_COUNT):
        intensity = final_bars[i]
        h = intensity * (min(WIDTH, HEIGHT) * 0.20)
        px, py = letter_points[i]
        
        prev_p = letter_points[(i - 1) % SAMPLE_COUNT]
        next_p = letter_points[(i + 1) % SAMPLE_COUNT]
        tx = next_p[0] - prev_p[0]
        ty = next_p[1] - prev_p[1]
        out_dx = px - letter_centroid[0]
        out_dy = py - letter_centroid[1]
        n1x, n1y = -ty, tx
        if (n1x * out_dx + n1y * out_dy) < 0:
            n1x, n1y = ty, -tx
        dist = math.sqrt(n1x * n1x + n1y * n1y) or 1
        ndx, ndy = n1x / dist, n1y / dist
        
        x1 = px + shake[0]
        y1 = py + shake[1]
        ox = x1 + ndx * h
        oy = y1 + ndy * h
        ix = x1 - ndx * (h * 0.4)
        iy = y1 - ndy * (h * 0.4)
        
        color = bar_color_letter(i, SAMPLE_COUNT, intensity)
        if phonk_mode:
            color = bar_color_phonk(i, SAMPLE_COUNT, intensity)
        
        r, g, b = color
        pygame.draw.line(glow_surf, (r, g, b, 45), (x1, y1), (ox, oy), 9)
        pygame.draw.line(glow_surf, (r, g, b, 20), (x1, y1), (ix, iy), 6)
        pygame.draw.line(screen, color, (x1, y1), (ox, oy), 4)
        pygame.draw.line(screen, color, (x1, y1), (ix, iy), 2)
        
        if avee_mode and peak_hold[i] > 0.01:
            ph = peak_hold[i] * (min(WIDTH, HEIGHT) * 0.20)
            pdx = x1 + ndx * ph
            pdy = y1 + ndy * ph
            pygame.draw.circle(screen, (255, 255, 255), (int(pdx), int(pdy)), 2)
    
    # --- Avee Player Style RGB/Trail Effect (最后绘制，确保在最上层) ---
    if letter_trail_mode and letter_mode and on_beat:
        # 根据 beat_confidence 动态调整强度
        intensity = min(1.0, beat_confidence * 1.2)
        
        # 三层残影：红、绿、蓝
        layers = [
            {"color": (255, 50, 80), "offset_mult": 8, "alpha": 70},   # 红
            {"color": (80, 255, 80), "offset_mult": 5, "alpha": 55},   # 绿
            {"color": (60, 100, 255), "offset_mult": 2, "alpha": 40},  # 蓝
        ]
        
        for layer in layers:
            # 偏移量随 beat 强度变化
            offset_x = random.randint(-layer["offset_mult"], layer["offset_mult"]) * intensity
            offset_y = random.randint(-layer["offset_mult"], layer["offset_mult"]) * intensity
            
            trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            
            # 只绘制字母轮廓，不画全屏
            for i in range(SAMPLE_COUNT):
                intensity_bar = final_bars[i]
                h = intensity_bar * (min(WIDTH, HEIGHT) * 0.20)
                px, py = letter_points[i]
                
                prev_p = letter_points[(i - 1) % SAMPLE_COUNT]
                next_p = letter_points[(i + 1) % SAMPLE_COUNT]
                tx = next_p[0] - prev_p[0]
                ty = next_p[1] - prev_p[1]
                out_dx = px - letter_centroid[0]
                out_dy = py - letter_centroid[1]
                n1x, n1y = -ty, tx
                if (n1x * out_dx + n1y * out_dy) < 0:
                    n1x, n1y = ty, -tx
                dist = math.sqrt(n1x * n1x + n1y * n1y) or 1
                ndx, ndy = n1x / dist, n1y / dist
                
                x1 = px + shake[0] + offset_x
                y1 = py + shake[1] + offset_y
                ox = x1 + ndx * h
                oy = y1 + ndy * h
                
                pygame.draw.line(trail_surf, layer["color"], (x1, y1), (ox, oy), 4)
            
            trail_surf.set_alpha(int(layer["alpha"] * intensity))
            screen.blit(trail_surf, (0, 0))

# ------------------------
# MAIN
# ------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("viz")
clock = pygame.time.Clock()

glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

running = True
while running:
    mx, my = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.VIDEORESIZE:
            new_w, new_h = event.w, event.h
            size_changed = abs(new_w - WIDTH) > 20 or abs(new_h - HEIGHT) > 20
            WIDTH, HEIGHT = new_w, new_h
            CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            _cached_title_surf = None
            _cached_artist_surf = None
            if size_changed and title != "No Track" and letter_points:
                generate_letter_points(title[0])

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            show_ui = not show_ui

        if event.type == pygame.MOUSEBUTTONDOWN and show_ui:
            # GENRE button opens panel
            if BUTTON_RECTS["genre"].collidepoint(event.pos):
                genre_panel_open = not genre_panel_open
            # If genre panel is open, handle its UI
            elif genre_panel_open and genre_checkbox_rects:
                # Checkboxes
                for rect, genre, color in genre_checkbox_rects:
                    if rect.collidepoint(event.pos):
                        if genre in active_genres:
                            active_genres.remove(genre)
                        else:
                            active_genres.add(genre)
                # AUTO DETECT
                if genre_auto_rect.collidepoint(event.pos):
                    auto_detect_genres()
                # CLOSE
                if genre_close_rect.collidepoint(event.pos):
                    genre_panel_open = False
            # Otherwise, normal button handling
            elif BUTTON_RECTS["open"].collidepoint(event.pos):
                p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg")])
                if p:
                    reload_audio(p)
            elif BUTTON_RECTS["letter"].collidepoint(event.pos):
                letter_mode = not letter_mode
            elif BUTTON_RECTS["shake"].collidepoint(event.pos):
                shake_mode = not shake_mode
                if shake_mode:
                    beat_shake_mode = False
            elif BUTTON_RECTS["beat_shake"].collidepoint(event.pos):
                beat_shake_mode = not beat_shake_mode
                if beat_shake_mode:
                    shake_mode = False
            elif BUTTON_RECTS["wave_bars"].collidepoint(event.pos):
                wave_bars_mode = not wave_bars_mode
            elif BUTTON_RECTS["phonk"].collidepoint(event.pos):
                phonk_mode = not phonk_mode
                _cached_artist_surf = None
            elif BUTTON_RECTS["bg_load"].collidepoint(event.pos):
                bg_load_enabled = not bg_load_enabled
            elif BUTTON_RECTS["avee"].collidepoint(event.pos):
                avee_mode = not avee_mode
            elif BUTTON_RECTS["artist"].collidepoint(event.pos):
                root = tk.Tk()
                root.withdraw()
                a = simpledialog.askstring("Artist", "Artist name:")
                if a:
                    artist = a
                    _cached_artist_surf = None
            elif BUTTON_RECTS["cover"].collidepoint(event.pos):
                p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png")])
                if p:
                    bg_surface = prepare_bg(Image.open(p).convert("RGB"))
            elif BUTTON_RECTS["rgb"].collidepoint(event.pos):
                letter_trail_mode = not letter_trail_mode
            elif SLIDER_RECT.collidepoint(event.pos):
                slider_dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False
    
    if slider_dragging:
        volume = np.clip((mx - SLIDER_RECT.x) / SLIDER_RECT.w, 0, 1)
        if sound:
            sound.set_volume(volume)
    
    # ---- Background ----
    if original_cover_pil is not None:
        beat_bg = make_beat_bg(smooth_beat)
        if beat_bg:
            screen.blit(beat_bg, (0, 0))
        else:
            screen.fill((10, 10, 16))
    elif bg_surface:
        screen.blit(bg_surface, (0, 0))
    else:
        screen.fill((10, 10, 16))
    
    glow_surf.fill((0, 0, 0, 0))
    
    if data is not None:
        idx = int((time.time() - start_time) * samplerate)
        
        if idx + 2048 < len(data):
            chunk = data[idx:idx + 2048]
            fft_raw = np.abs(np.fft.rfft(chunk * WINDOW))
            freqs = np.fft.rfftfreq(2048, d=1.0 / samplerate)
            
            fft = np.zeros(SAMPLE_COUNT)
            for b in range(SAMPLE_COUNT):
                lo = np.searchsorted(freqs, LOG_EDGES[b])
                hi = np.searchsorted(freqs, LOG_EDGES[b + 1])
                if hi > lo:
                    fft[b] = np.mean(fft_raw[lo:hi])
                elif lo < len(fft_raw):
                    fft[b] = fft_raw[lo]
            
            mx_val = np.max(fft)
            if mx_val > 1e-6:
                fft /= mx_val
            
            on_beat, smooth_beat, beat_conf = detect_beat_advanced(fft, smooth_beat)
            beat_confidence = beat_conf
            
            if avee_mode:
                bars = np.log10(1 + fft * 9) / 1.0
                for i in range(SAMPLE_COUNT):
                    if bars[i] > bar_falloff[i]:
                        bar_falloff[i] = bars[i]
                        peak_hold[i] = bars[i]
                        peak_timer[i] = 30
                    else:
                        bar_falloff[i] = max(0, bar_falloff[i] - 0.018)
                        peak_timer[i] -= 1
                        if peak_timer[i] <= 0:
                            peak_hold[i] = max(0, peak_hold[i] - 0.012)
                raw_bars = bar_falloff
            else:
                smooth_bars = smooth_bars * 0.80 + fft * 0.20
                raw_bars = smooth_bars
            
            if wave_bars_mode:
                tide_speed = 0.018 + smooth_beat * 0.04
                tide_phase += tide_speed
                tide_amplitude = 0.10 + smooth_beat * 0.35
                angles = np.linspace(0, 2 * math.pi, SAMPLE_COUNT, endpoint=False)
                tide = (
                    math.sin(tide_phase) * np.sin(angles + tide_phase) * tide_amplitude +
                    math.sin(tide_phase * 1.3) * np.sin(2 * angles + tide_phase * 0.7) * (tide_amplitude * 0.4)
                )
                raw_bars_with_tide = np.clip(raw_bars + tide, 0, 1)
            else:
                raw_bars_with_tide = raw_bars
            
            wave_bars = wave_bars * 0.72 + raw_bars_with_tide * 0.28
            final_bars = wave_bars.copy()
            
            if shake_mode:
                if on_beat:
                    shake_target[:] = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * (5.0 + phonk_intensity * 8)
                shake_current[:] = shake_current * 0.55 + shake_target * 0.45
                shake_target[:] *= 0.75
                shake = shake_current.copy()
            elif beat_shake_mode and on_beat:
                shake_target[:] = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * (smooth_beat * 10 + phonk_intensity * 15)
                shake_current[:] = shake_current * 0.55 + shake_target * 0.45
                shake_target[:] *= 0.75
                shake = shake_current.copy()
            else:
                shake_current[:] *= 0.75
                shake_target[:] *= 0.75
                shake = shake_current.copy()
            
            if on_beat and smooth_beat > 0.5:
                base_radius = min(WIDTH, HEIGHT) * 0.22
                cx, cy = CENTER[0] + shake[0], CENTER[1] + shake[1]
                particle_count = int(5 + phonk_intensity * 15)
                for _ in range(particle_count):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(2, 8) * (smooth_beat + phonk_intensity)
                    life = int(20 + phonk_intensity * 20)
                    color = (255, 50, 200) if phonk_mode else (0, 200, 255)
                    particles.append(Particle(cx, cy, angle, speed, life, color))
            
            particles = [p for p in particles if p.update()]
            for p in particles:
                p.draw(glow_surf)
            
            base_r = min(WIDTH, HEIGHT) * 0.22
            pulse_radius_target = base_r + smooth_beat * base_r * 0.12
            if phonk_mode:
                pulse_radius_target += phonk_intensity * base_r * 0.08
            pulse_radius = pulse_radius * 0.85 + pulse_radius_target * 0.15
            
            rotation += 0.004 + smooth_beat * 0.015
            if phonk_mode:
                rotation += phonk_intensity * 0.008
            
            if wave_bars_mode:
                wave_phase += 0.06 + smooth_beat * 0.10
                wave_burst = smooth_beat * 1.8 + phonk_intensity * 1.5
                wave_offsets = (
                    np.sin(np.linspace(0, 4 * math.pi, SAMPLE_COUNT) + wave_phase) * wave_burst +
                    np.sin(np.linspace(0, 8 * math.pi, SAMPLE_COUNT) + wave_phase * 1.4) * (wave_burst * 0.4)
                )
                indices = (np.arange(SAMPLE_COUNT) + wave_offsets) % SAMPLE_COUNT
                lo_idx = indices.astype(int) % SAMPLE_COUNT
                hi_idx = (lo_idx + 1) % SAMPLE_COUNT
                frac = indices - np.floor(indices)
                final_bars = final_bars[lo_idx] * (1 - frac) + final_bars[hi_idx] * frac

            # --- GENRE EFFECTS: modify bars, colors, and visuals ---
            genre_fx = active_genres.copy()
            breakcore_palette = [(80,255,80), (255,80,255), (0,255,120), (255,255,255)]
            
            if "Breakcore" in genre_fx:
                if random.random() < 0.04:
                    skip_idx = random.randint(0, SAMPLE_COUNT-1)
                    final_bars[skip_idx] = min(1.0, final_bars[skip_idx] + random.uniform(0.5, 1.0))
                for _ in range(random.randint(0,2)):
                    if random.random() < 0.03:
                        i = random.randint(0, SAMPLE_COUNT-1)
                        final_bars[i] = min(1.0, final_bars[i] + random.uniform(0.3, 0.7))
                if random.random() < 0.01:
                    final_bars *= random.uniform(0.0, 0.3)
            
            if "Speedcore" in genre_fx:
                rotation += 0.03 + beat_confidence * 0.04
                wave_bars[:] = wave_bars * 0.93 + raw_bars_with_tide * 0.07
                if on_beat and int(time.time() * 2) % 8 == 0:
                    for i in range(SAMPLE_COUNT):
                        final_bars[i] = max(final_bars[i], 0.7)
                if on_beat:
                    for _ in range(8):
                        angle = random.uniform(0, 2*math.pi)
                        speed = random.uniform(7, 14)
                        life = random.randint(18, 32)
                        color = (255,255,255) if random.random()<0.5 else (255,80,80)
                        particles.append(Particle(CENTER[0], CENTER[1], angle, speed, life, color))
            
            if "Hardcore" in genre_fx:
                if on_beat:
                    final_bars[:] = np.clip(final_bars + 0.18, 0, 1)
                if not letter_mode:
                    for gl in range(4,0,-1):
                        halo_col = (255, 180, 60, 30+gl*18)
                        pygame.draw.circle(screen, halo_col, CENTER.astype(int), int(pulse_radius*1.1+gl*8), 8+gl*2)
                mid_idx = slice(SAMPLE_COUNT//3, 2*SAMPLE_COUNT//3)
                if np.max(final_bars[mid_idx]) > 0.7:
                    glow_col = (255, 220, 80, 80)
                    pygame.draw.circle(screen, glow_col, CENTER.astype(int), int(pulse_radius*0.7), 0)
            
            if "Frenchcore" in genre_fx:
                midhi = slice(SAMPLE_COUNT//2, int(SAMPLE_COUNT*0.85))
                final_bars[midhi] = np.clip(final_bars[midhi]*1.25, 0, 1)
                if not letter_mode:
                    for i in range(0, SAMPLE_COUNT, 2):
                        if int(time.time()*20)%2 == 0:
                            final_bars[i] *= 0.7
            
            if "Uptempo" in genre_fx:
                sub_idx = slice(0, SAMPLE_COUNT//8)
                final_bars[sub_idx] = np.clip(final_bars[sub_idx]*2.2, 0, 1)
                if on_beat:
                    shake += np.random.uniform(-10, 10, 2)

            # --- Draw visualizer ---
            if genre_fx:
                if letter_mode:
                    draw_letter_visualizer_genre(screen, final_bars, shake, glow_surf, genre_fx, avee_mode, peak_hold, letter_points, letter_centroid)
                else:
                    draw_circle_visualizer_genre(screen, final_bars, shake, glow_surf, genre_fx, rotation, pulse_radius, avee_mode, peak_hold)
            else:
                if letter_mode:
                    draw_letter_visualizer_enhanced(screen, final_bars, shake, glow_surf, on_beat)
                else:
                    draw_circle_visualizer_enhanced(screen, final_bars, shake, glow_surf)

            screen.blit(glow_surf, (0, 0))

            if not letter_mode:
                ring_color = (255, 80, 200) if phonk_mode else (80, 210, 255)
                draw_glow_circle(screen, ring_color, CENTER + shake * 0.3, pulse_radius, width=2)

            if phonk_mode and cowbell_energy > 0.3:
                cw_font = get_font(16, True)
                cw_surf = cw_font.render("🔔", True, (255, 200, 50))
                screen.blit(cw_surf, (WIDTH - 40, HEIGHT - 40))

            ensure_title_surfaces()
            screen.blit(_cached_title_surf, (20, HEIGHT - 60))
            screen.blit(_cached_artist_surf, (20, HEIGHT - 30))
            
        else:
            wave_bars *= 0.92
            final_bars = wave_bars.copy()
            shake = np.array([0.0, 0.0])
            base_r = min(WIDTH, HEIGHT) * 0.22
            pulse_radius = pulse_radius * 0.97 + base_r * 0.03
            
            if letter_mode:
                draw_letter_visualizer_enhanced(screen, final_bars, shake, glow_surf, on_beat)
            else:
                draw_circle_visualizer_enhanced(screen, final_bars, shake, glow_surf)
            screen.blit(glow_surf, (0, 0))
            if not letter_mode:
                ring_color = (255, 80, 200) if phonk_mode else (80, 210, 255)
                draw_glow_circle(screen, ring_color, CENTER, pulse_radius, width=2)
            
            ensure_title_surfaces()
            screen.blit(_cached_title_surf, (20, HEIGHT - 60))
            screen.blit(_cached_artist_surf, (20, HEIGHT - 30))
    
    else:
        ensure_title_surfaces()
        screen.blit(_cached_title_surf, (20, HEIGHT - 60))
        screen.blit(_cached_artist_surf, (20, HEIGHT - 30))
    
    # ---- UI Buttons ----
    if show_ui:
        label_map = {
            "open":       "OPEN",
            "cover":      "COVER",
            "artist":     "ARTIST",
            "avee":       "AVEE Y"          if avee_mode        else "AVEE",
            "letter":     "LETTER Y"        if letter_mode      else "LETTER",
            "shake":      "SHAKE Y"         if shake_mode       else "SHAKE",
            "beat_shake": "BEAT SHAKE Y"    if beat_shake_mode  else "BEAT SHAKE",
            "wave_bars":  "WAVE BARS Y"     if wave_bars_mode   else "WAVE BARS",
            "phonk":      "! PHONK Y"       if phonk_mode       else "! PHONK",
            "bg_load":    "BG LOAD Y"       if bg_load_enabled  else "BG LOAD",
            "genre":      "GENRE",
            "rgb":        "TRAIL Y"         if letter_trail_mode else "TRAIL",
        }
        for k, r in BUTTON_RECTS.items():
            if k == "phonk":
                active = phonk_mode
            elif k == "bg_load":
                active = bg_load_enabled
            elif k == "genre":
                active = genre_panel_open
            elif k == "rgb":
                active = letter_trail_mode
            else:
                active = (
                    (k == "avee"       and avee_mode)       or
                    (k == "letter"     and letter_mode)     or
                    (k == "shake"      and shake_mode)      or
                    (k == "beat_shake" and beat_shake_mode) or
                    (k == "wave_bars"  and wave_bars_mode)
                )
            bg_col = (120, 20, 80) if (k == "phonk" and active) else ((0, 160, 160) if active else (30, 32, 48))
            if k == "genre" and active:
                bg_col = (60, 60, 60)
            if k == "rgb" and active:
                bg_col = (80, 30, 100)
            bdr_col = (255, 50, 150) if (k == "phonk" and active) else ((0, 220, 220) if active else (60, 65, 100))
            if k == "genre" and active:
                bdr_col = (180, 180, 180)
            if k == "rgb" and active:
                bdr_col = (200, 80, 255)
            pygame.draw.rect(screen, bg_col, r, border_radius=6)
            pygame.draw.rect(screen, bdr_col, r, width=1, border_radius=6)
            txt_color = (255, 200, 50) if (k == "phonk") else ((255, 255, 255) if active else (160, 170, 210))
            if k == "genre":
                txt_color = (180, 255, 180) if active else (200, 200, 200)
            if k == "rgb":
                txt_color = (255, 180, 255) if active else (200, 180, 220)
            txt = get_font(15, bold=active).render(label_map[k], True, txt_color)
            screen.blit(txt, txt.get_rect(center=r.center))

        # Draw active genre tags next to GENRE button
        genre_btn_rect = BUTTON_RECTS["genre"]
        tag_x = genre_btn_rect.right + 10
        tag_y = genre_btn_rect.centery - 10
        for genre, color in GENRE_LIST:
            if genre in active_genres:
                tag_surf = get_font(13, True).render(genre, True, color)
                tag_bg = pygame.Surface((tag_surf.get_width()+10, tag_surf.get_height()), pygame.SRCALPHA)
                tag_bg.fill((*color, 60))
                screen.blit(tag_bg, (tag_x, tag_y))
                screen.blit(tag_surf, (tag_x+5, tag_y))
                tag_x += tag_surf.get_width() + 18

        vol_fill = int(volume * SLIDER_RECT.w)
        pygame.draw.rect(screen, (30, 32, 48), SLIDER_RECT, border_radius=4)
        slider_color = (180, 30, 100) if phonk_mode else (0, 180, 180)
        pygame.draw.rect(screen, slider_color, (SLIDER_RECT.x, SLIDER_RECT.y, vol_fill, SLIDER_RECT.h), border_radius=4)
        pygame.draw.rect(screen, (60, 65, 100), SLIDER_RECT, width=1, border_radius=4)
        kx = SLIDER_RECT.x + vol_fill
        ky = SLIDER_RECT.y + SLIDER_RECT.h // 2
        knob_color = (255, 60, 160) if phonk_mode else (0, 220, 220)
        pygame.draw.circle(screen, knob_color, (kx, ky), 6)

        vol_label = get_font(12).render(f"VOL  {int(volume*100)}%", True, (120, 130, 170))
        screen.blit(vol_label, (SLIDER_RECT.x, SLIDER_RECT.y + 14))

        hint = get_font(11).render("SPACE = hide UI", True, (70, 75, 110))
        screen.blit(hint, (10, HEIGHT - 20))

        # --- GENRE PANEL FLOATING UI ---
        if genre_panel_open:
            panel_w, panel_h = 320, 270
            panel_x = genre_btn_rect.right + 20
            panel_y = max(genre_btn_rect.top - 10, 10)
            
            panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel_surf.fill((30, 30, 40, 220))
            pygame.draw.rect(panel_surf, (100, 100, 150), (0, 0, panel_w, panel_h), 1)
            
            genre_checkbox_rects = []
            cb_x, cb_y = 24, 28
            for i, (genre, color) in enumerate(GENRE_LIST):
                box_rect = pygame.Rect(cb_x, cb_y + i*38, 28, 28)
                pygame.draw.rect(panel_surf, color, box_rect, border_radius=5, width=2)
                if genre in active_genres:
                    pygame.draw.rect(panel_surf, color, box_rect.inflate(-6, -6), border_radius=4)
                label = get_font(17, True).render(genre, True, color)
                panel_surf.blit(label, (box_rect.right + 10, box_rect.y + 2))
                genre_checkbox_rects.append((pygame.Rect(panel_x + box_rect.x, panel_y + box_rect.y, box_rect.w, box_rect.h), genre, color))
            
            genre_auto_rect = pygame.Rect(panel_x + 20, panel_y + panel_h - 60, 120, 34)
            pygame.draw.rect(panel_surf, (80, 200, 120), (20, panel_h - 60, 120, 34), border_radius=7)
            auto_txt = get_font(15, True).render("AUTO DETECT", True, (255, 255, 255))
            panel_surf.blit(auto_txt, (32, panel_h - 54))
            
            genre_close_rect = pygame.Rect(panel_x + panel_w - 90, panel_y + panel_h - 60, 70, 34)
            pygame.draw.rect(panel_surf, (180, 80, 80), (panel_w - 90, panel_h - 60, 70, 34), border_radius=7)
            close_txt = get_font(15, True).render("CLOSE", True, (255, 255, 255))
            panel_surf.blit(close_txt, (panel_w - 78, panel_h - 54))
            
            title_txt = get_font(18, True).render("GENRE EFFECTS", True, (200, 255, 200))
            panel_surf.blit(title_txt, (panel_w // 2 - title_txt.get_width() // 2, 8))
            
            screen.blit(panel_surf, (panel_x, panel_y))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()