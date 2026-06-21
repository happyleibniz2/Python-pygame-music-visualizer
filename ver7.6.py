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
from PIL import Image, ImageFilter
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, APIC
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
try:
    import moviepy.editor as mpy
except Exception:
    mpy = None

# -------------------------------------------------------------------------
# CONSTANTS & LAYOUT
# -------------------------------------------------------------------------
SAMPLE_COUNT = 120
WIDTH, HEIGHT = 900, 900
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)

def get_button_rects(w, h):
    bw, bh, bm = 110, 28, 6
    col0, col1 = 20, 20 + bw + bm
    def row(r): return h - 145 + r * (bh + bm)
    return {
        "open":          pygame.Rect(col0, row(0), bw, bh),
        "toggle_bars":   pygame.Rect(col0, row(1), bw, bh),
        "toggle_image":  pygame.Rect(col0, row(2), bw, bh),
        "toggle_part":   pygame.Rect(col0, row(3), bw, bh),
        "toggle_rgb":    pygame.Rect(col0, row(4), bw, bh),
        "toggle_blur":   pygame.Rect(col1, row(0), bw, bh),
        "toggle_mblur":  pygame.Rect(col1, row(1), bw, bh),
        "toggle_mirror": pygame.Rect(col1, row(2), bw, bh),
        "toggle_color":  pygame.Rect(col1, row(3), bw, bh),
        "toggle_text":   pygame.Rect(col1, row(4), bw, bh),
    }, pygame.Rect(col1, row(5), bw, 10)

BUTTON_RECTS, SLIDER_RECT = get_button_rects(WIDTH, HEIGHT)

# -------------------------------------------------------------------------
# AUDIO DSP PIPELINE
# -------------------------------------------------------------------------
WINDOW = np.hanning(2048)
LOG_EDGES = np.logspace(np.log10(40.0), np.log10(16000.0), SAMPLE_COUNT + 1)
BEAT_BANDS = {'sub_bass': (0, 4), 'phonk_kick': (4, 9), 'low_mid': (12, 24)}
beat_energy_history = {band: [] for band in BEAT_BANDS}

_bar_lo = _bar_hi = None
def _precompute_bar_bins(samplerate):
    global _bar_lo, _bar_hi
    freqs = np.fft.rfftfreq(2048, d=1.0 / samplerate)
    _bar_lo = np.searchsorted(freqs, LOG_EDGES[:-1])
    _bar_hi = np.searchsorted(freqs, LOG_EDGES[1:])

# -------------------------------------------------------------------------
# RENDER STATE & GLOBALS
# -------------------------------------------------------------------------
class RenderState:
    def __init__(self):
        self.dt = 0.016
        self.fft = np.zeros(SAMPLE_COUNT)
        self.smooth_bars = np.zeros(SAMPLE_COUNT)
        self.peak_bars = np.zeros(SAMPLE_COUNT)
        self.peak_timer = np.zeros(SAMPLE_COUNT)
        self.is_beat = False
        self.smooth_beat = 0.0
        self.shake = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.time = 0.0
        self.original_cover = None

RS = RenderState()

data, samplerate, sound = None, None, None
title, artist = "No Track", "Unknown"
volume = 1.0
show_ui = True

_font_cache = {}
def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            try:
                _font_cache[key] = pygame.font.SysFont("Segoe UI,Arial", size, bold=bold)
            except Exception:
                # If SysFont fails (platform font list issues), try a generic SysFont
                try:
                    _font_cache[key] = pygame.font.SysFont(None, size, bold=bold)
                except Exception:
                    # Final fallback to the default Font
                    _font_cache[key] = pygame.font.Font(None, size)
        except Exception:
            # Extremely defensive: return a dummy font-like object that provides `render`
            class _DummyFont:
                def render(self, *a, **k):
                    return pygame.Surface((1, 1))
            _font_cache[key] = _DummyFont()
    return _font_cache[key]

# -------------------------------------------------------------------------
# AVEE NODE ARCHITECTURE (Element & Composition)
# -------------------------------------------------------------------------
class Element:
    def __init__(self): self.enabled = True
    def update(self, rs: RenderState): pass
    def render(self, surface: pygame.Surface, rs: RenderState): pass

class Composition(Element):
    def __init__(self):
        super().__init__()
        self.children = []
    def add(self, child): self.children.append(child)
    def update(self, rs):
        for c in self.children: 
            if c.enabled: c.update(rs)
    def render(self, surface, rs):
        for c in self.children: 
            if c.enabled: c.render(surface, rs)

# --- 1. Audio Data Provider ---
class AudioDataProviderElement(Element):
    def __init__(self):
        super().__init__()
        self.shake_target = np.array([0.0, 0.0])
        
    def update(self, rs: RenderState):
        rs.time += rs.dt
        if data is not None and sound and pygame.mixer.get_busy():
            idx = int(rs.time * samplerate)
            if idx + 2048 < len(data):
                chunk = data[idx:idx+2048]
                fft_raw = np.abs(np.fft.rfft(chunk * WINDOW))
                for b in range(SAMPLE_COUNT):
                    lo, hi = _bar_lo[b], _bar_hi[b]
                    if hi > lo: rs.fft[b] = np.mean(fft_raw[lo:min(hi, len(fft_raw))])
                    elif lo < len(fft_raw): rs.fft[b] = fft_raw[lo]
                max_v = np.max(rs.fft)
                if max_v > 1e-4: rs.fft /= max_v

        # Beat Detection
        beat_conf = 0.0
        for band, (lo, hi) in BEAT_BANDS.items():
            en = np.mean(rs.fft[lo:min(hi, len(rs.fft))])
            hist = beat_energy_history[band]
            hist.append(en)
            if len(hist) > 40: hist.pop(0)
            if len(hist) > 5 and np.mean(hist[:-1]) > 1e-6:
                ratio = en / np.mean(hist[:-1])
                if ratio > 1.35: beat_conf += 0.5 * ratio
        
        rs.is_beat = beat_conf > 0.8
        rs.smooth_beat = rs.smooth_beat * 0.4 + min(beat_conf/2.0, 1.0) * 0.6 if rs.is_beat else rs.smooth_beat * 0.86

        # Audio Smoothing & Peaks
        for i in range(SAMPLE_COUNT):
            target = rs.fft[i]
            if target > rs.smooth_bars[i]: rs.smooth_bars[i] = rs.smooth_bars[i] * 0.2 + target * 0.8
            else: rs.smooth_bars[i] -= 3.0 * rs.dt
            rs.smooth_bars[i] = max(0.0, rs.smooth_bars[i])
            
            if rs.smooth_bars[i] > rs.peak_bars[i]:
                rs.peak_bars[i] = rs.smooth_bars[i]
                rs.peak_timer[i] = 0.3
            else:
                if rs.peak_timer[i] > 0: rs.peak_timer[i] -= rs.dt
                else: rs.peak_bars[i] = max(0.0, rs.peak_bars[i] - 1.1 * rs.dt)

        # Shake & Rotation
        if rs.is_beat:
            amt = 8.0 * rs.smooth_beat
            self.shake_target = np.array([random.uniform(-amt, amt), random.uniform(-amt, amt)])
        else:
            self.shake_target *= 0.82
        rs.shake = rs.shake * 0.6 + self.shake_target * 0.4
        rs.rotation += 0.012 + rs.smooth_beat * 0.04

# --- 2. Image / LogoMark ---
class ImageElement(Element):
    def render(self, surface, rs):
        cx, cy = int(CENTER[0] + rs.shake[0]), int(CENTER[1] + rs.shake[1])
        rad = int(min(WIDTH, HEIGHT) * 0.16) + int(rs.smooth_beat * 16)
        pygame.draw.circle(surface, (10, 10, 14, 160), (cx + 4, cy + 5), rad) # Shadow
        
        if rs.original_cover:
            sz = rad * 2
            if sz > 15:
                pg_img = pygame.transform.smoothscale(rs.original_cover, (sz, sz))
                mask = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(mask, (255, 255, 255, 255), (rad, rad), rad)
                pg_img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                surface.blit(pg_img, (cx - rad, cy - rad))
        else:
            pygame.draw.circle(surface, (24, 24, 30), (cx, cy), rad)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), rad, 2)

# --- 3. Bars Segment ---
class SegmentElement(Element):
    def render(self, surface, rs):
        cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
        base_r = min(WIDTH, HEIGHT) * 0.16 + (rs.smooth_beat * 16)
        max_h = min(WIDTH, HEIGHT) * 0.26
        whalf = (0.55 * (WIDTH / 900)) * (base_r * 0.05)
        
        for i in range(SAMPLE_COUNT):
            v, p = rs.smooth_bars[i], rs.peak_bars[i]
            h = max(2.5, v * max_h)
            ang = (i / SAMPLE_COUNT) * 2 * math.pi + rs.rotation
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            
            rx, ry = cx + cos_a * base_r, cy + sin_a * base_r
            x2, y2 = rx - sin_a * whalf, ry + cos_a * whalf
            x3, y3 = rx + sin_a * whalf, ry - cos_a * whalf
            x0, y0 = x2 + cos_a * h, y2 + sin_a * h
            x1, y1 = x3 + cos_a * h, y3 + sin_a * h
            
            col = (int(max(0, min(255, 130 - 50 * v))), int(max(0, min(255, 225 + 30 * v))), 255)
            pygame.draw.polygon(surface, col, [(x0, y0), (x1, y1), (x3, y3), (x2, y2)])
            
            if p > 0.01:
                ph = p * max_h
                pygame.draw.circle(surface, (255,255,255), (int(cx + cos_a * (base_r + ph + 8)), int(cy + sin_a * (base_r + ph + 8))), 2)

# --- 4. Text / Fps Metadata ---
class TextElement(Element):
    def render(self, surface, rs):
        if show_ui: return # Hide when UI is visible
        t_font, a_font = get_font(28, True), get_font(18, False)
        t_surf = t_font.render(title[:34], True, (255, 255, 255))
        a_surf = a_font.render(artist[:40], True, (125, 205, 255))
        surface.blit(t_surf, (45, HEIGHT - 110))
        surface.blit(a_surf, (45, HEIGHT - 75))

# --- 5. Particles Engine ---
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max']
    def __init__(self, x, y, vx, vy, l): self.x, self.y, self.vx, self.vy, self.life, self.max = x, y, vx, vy, l, l

class ParticlesElement(Element):
    def __init__(self):
        super().__init__()
        self.particles = []
    def update(self, rs):
        if rs.is_beat and rs.smooth_beat > 0.4:
            cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
            for _ in range(int(3 + rs.smooth_beat * 10)):
                a = random.uniform(0, 2 * math.pi)
                s = random.uniform(50, 200) * rs.smooth_beat
                self.particles.append(Particle(cx, cy, math.cos(a)*s, math.sin(a)*s, random.uniform(1.0, 3.0)))
        
        for p in self.particles[:]:
            p.x += p.vx * rs.dt
            p.y += p.vy * rs.dt
            p.life -= rs.dt
            if p.life <= 0: self.particles.remove(p)

    def render(self, surface, rs):
        for p in self.particles:
            alpha = int(255 * (p.life / p.max))
            pygame.draw.circle(surface, (255, 255, 255, alpha), (int(p.x), int(p.y)), max(1, int(3 * (p.life/p.max))))

# --- 6. Post Processing: RGB Split ---
class RgbSplitEffectElement(Element):
    def render(self, surface, rs):
        amt = int(2 + 10 * rs.smooth_beat)
        if amt <= 0: return
        r, g, b = surface.copy(), surface.copy(), surface.copy()
        r.fill((255,0,0), special_flags=pygame.BLEND_RGBA_MULT)
        g.fill((0,255,0), special_flags=pygame.BLEND_RGBA_MULT)
        b.fill((0,0,255), special_flags=pygame.BLEND_RGBA_MULT)
        surface.fill((0,0,0,0))
        surface.blit(g, (0, 0))
        surface.blit(r, (-amt, 0), special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(b, (amt, 0), special_flags=pygame.BLEND_RGBA_ADD)

# --- 7. Post Processing: Motion Blur ---
class MotionBlurEffectElement(Element):
    def __init__(self):
        super().__init__()
        self.trail = None
    def render(self, surface, rs):
        if self.trail is None or self.trail.get_size() != surface.get_size():
            self.trail = surface.copy()
            self.trail.set_alpha(150)
        else:
            self.trail.blit(surface, (0,0), special_flags=pygame.BLEND_RGBA_MAX)
            self.trail.set_alpha(int(180 - rs.smooth_beat * 100)) # fade out on beats
            surface.blit(self.trail, (0,0))

# --- 8. Post Processing: Mirror Effect ---
class MirrorEffectElement(Element):
    def render(self, surface, rs):
        # Mirror horizontal (Left side copied to right side)
        half = surface.subsurface((0, 0, WIDTH//2, HEIGHT))
        flipped = pygame.transform.flip(half, True, False)
        surface.blit(flipped, (WIDTH//2, 0))

# --- 9. Post Processing: Blur Stack (Downscale) ---
class BlurEffectElement(Element):
    def render(self, surface, rs):
        # CPU Blur fake: Downscale heavily, upscale
        scale = max(0.1, 0.5 - (rs.smooth_beat * 0.3))
        sw, sh = max(1, int(WIDTH * scale)), max(1, int(HEIGHT * scale))
        tiny = pygame.transform.smoothscale(surface, (sw, sh))
        blurred = pygame.transform.smoothscale(tiny, (WIDTH, HEIGHT))
        surface.blit(blurred, (0,0))

# --- 10. Post Processing: Color Correction ---
class ColorCorrectionElement(Element):
    def render(self, surface, rs):
        # Tinting towards blue/cyan to match Avee template
        tint = pygame.Surface((WIDTH, HEIGHT))
        tint.fill((0, 30, 60))
        surface.blit(tint, (0,0), special_flags=pygame.BLEND_RGB_ADD)
        
        # Contrast multiplier
        contrast = pygame.Surface((WIDTH, HEIGHT))
        val = int(128 + 50 * rs.smooth_beat)
        contrast.fill((val, val, val))
        surface.blit(contrast, (0,0), special_flags=pygame.BLEND_RGB_MULT)

# -------------------------------------------------------------------------
# SCENE BUILDER
# -------------------------------------------------------------------------
Root = Composition()
CompAudio = AudioDataProviderElement()
CompParts = ParticlesElement()
CompBars = SegmentElement()
CompImage = ImageElement()
CompText = TextElement()

# Post Processors
PPE_RGB = RgbSplitEffectElement()
PPE_RGB.enabled = False
PPE_MBlur = MotionBlurEffectElement()
PPE_MBlur.enabled = False
PPE_Mirror = MirrorEffectElement()
PPE_Mirror.enabled = False
PPE_Blur = BlurEffectElement()
PPE_Blur.enabled = False
PPE_Color = ColorCorrectionElement()
PPE_Color.enabled = False

# Build Tree
Root.add(CompAudio)
Root.add(CompParts)
Root.add(CompBars)
Root.add(CompImage)
Root.add(CompText)
Root.add(PPE_Blur)
Root.add(PPE_MBlur)
Root.add(PPE_Mirror)
Root.add(PPE_Color)
Root.add(PPE_RGB)

# -------------------------------------------------------------------------
# ASSET HELPERS
# -------------------------------------------------------------------------
def extract_cover(file):
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".mp3":
            tags = ID3(file)
            for tag in tags.values():
                if isinstance(tag, APIC): return pygame.image.load(io.BytesIO(tag.data)).convert()
        elif ext == ".flac":
            audio = FLAC(file)
            if audio.pictures: return pygame.image.load(io.BytesIO(audio.pictures[0].data)).convert()
        elif ext in (".m4a", ".mp4", ".aac"):
            audio = MP4(file)
            if "covr" in audio: return pygame.image.load(io.BytesIO(bytes(audio["covr"][0]))).convert()
    except Exception: pass
    return None

def reload_audio(file):
    global data, samplerate, sound, title, RS
    try:
        data, samplerate = sf.read(file)
        if len(data.shape) > 1: data = data.mean(axis=1)
        pygame.mixer.quit()
        pygame.mixer.init(samplerate, -16, 2, 2048)
        sound = pygame.mixer.Sound(file)
        sound.set_volume(volume)
        sound.play()
        title = os.path.splitext(os.path.basename(file))[0]
        RS.time = 0.0
        RS.original_cover = extract_cover(file)
        _precompute_bar_bins(samplerate)
        for k in beat_energy_history: beat_energy_history[k].clear()
    except Exception as e:
        print(f"File load error: {e}")

# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
def main():
    global volume, show_ui, WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Avee Player - Pure Pygame Render Tree")
    clock = pygame.time.Clock()
    
    _precompute_bar_bins(44100)

    # UI State bindings
    btn_states = {
        "toggle_bars": CompBars,
        "toggle_image": CompImage,
        "toggle_part": CompParts,
        "toggle_text": CompText,
        "toggle_rgb": PPE_RGB,
        "toggle_blur": PPE_Blur,
        "toggle_mblur": PPE_MBlur,
        "toggle_mirror": PPE_Mirror,
        "toggle_color": PPE_Color
    }

    running = True
    while running:
        RS.dt = clock.tick(60) / 1000.0
        
        # 1. Update Phase (Physics, Audio, FFT)
        Root.update(RS)
        
        # 2. Render Phase
        screen.fill((5, 5, 10))
        
        # Build layer
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        Root.render(layer, RS)
        screen.blit(layer, (0,0))

        # 3. UI Overlay
        if show_ui:
            panel = pygame.Surface((WIDTH, 145), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 160))
            screen.blit(panel, (0, HEIGHT - 145))
            
            for name, rect in BUTTON_RECTS.items():
                active = False
                if name in btn_states: active = btn_states[name].enabled
                
                bg_col = (0, 140, 140) if active else (38, 38, 48)
                pygame.draw.rect(screen, bg_col, rect, border_radius=4)
                txt = get_font(11).render(name.replace("toggle_", "").upper(), True, (240, 240, 240))
                screen.blit(txt, (rect.x + 7, rect.y + 6))
                
            pygame.draw.rect(screen, (55, 55, 65), SLIDER_RECT, border_radius=3)
            fill_w = int(volume * SLIDER_RECT.width)
            if fill_w > 0:
                pygame.draw.rect(screen, (0, 195, 255), (SLIDER_RECT.x, SLIDER_RECT.y, fill_w, SLIDER_RECT.height), border_radius=3)
                
        # 4. Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h: show_ui = not show_ui
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and show_ui:
                pos = event.pos
                if SLIDER_RECT.collidepoint(pos):
                    volume = max(0.0, min(1.0, (pos[0] - SLIDER_RECT.x) / SLIDER_RECT.width))
                    if sound: sound.set_volume(volume)
                for name, rect in BUTTON_RECTS.items():
                    if rect.collidepoint(pos):
                        if name == "open":
                            root = tk.Tk(); root.withdraw()
                            f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.flac *.m4a *.wav")])
                            if f: reload_audio(f)
                        elif name in btn_states:
                            btn_states[name].enabled = not btn_states[name].enabled

        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()