# ---------------------------------------------------------------------------
# AVEE PLAYER PY • GPU EDITION
# Pixel work moved off the CPU onto OpenGL (moderngl + GLSL 330):
#   bars / peak dots / particles -> instanced quads (one draw call each)
#   cover disc + halo, background grid -> fragment shaders
#   glow, soft blur, motion blur, mirror, color correction, RGB split
#     -> single fullscreen composite pass over RGBA16F FBOs
#   UI panel -> rendered by pygame into an overlay texture (~10 Hz)
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import numpy as np
import soundfile as sf
import math
import random
import threading
import time
import os
import io
import tkinter as tk
from tkinter import filedialog
from mutagen.id3 import ID3, APIC
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
# moviepy imported previously but unused — removed to speed startup.

# -------------------------------------------------------------------------
# CONSTANTS & LAYOUT
# -------------------------------------------------------------------------
SAMPLE_COUNT = 144
WIDTH, HEIGHT = 1000, 820
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
PATH_MODES = ("LETTER", "CIRCLE", "LINE")

# audio visualization offset (ms) to compensate latency/export timing; 0 by default
AUDIO_VIS_OFFSET_MS = 0
# beat detection mode: 'flux' (spectral-flux) or 'rms' (RMS-based like Avee)
BEAT_MODE = "flux"


def get_button_rects(w, h):
    bw, bh, gap = 118, 30, 8
    panel_top = h - 206
    col0, col1, col2 = 22, 22 + bw + gap, 22 + (bw + gap) * 2

    def row(r):
        return panel_top + 54 + r * (bh + gap)

    return {
        "open": pygame.Rect(col0, row(0), bw, bh),
        "path_mode": pygame.Rect(col0, row(1), bw, bh),
        "toggle_bars": pygame.Rect(col0, row(2), bw, bh),
        "toggle_image": pygame.Rect(col0, row(3), bw, bh),
        "toggle_part": pygame.Rect(col1, row(0), bw, bh),
        "toggle_text": pygame.Rect(col1, row(1), bw, bh),
        "toggle_rgb": pygame.Rect(col1, row(2), bw, bh),
        "toggle_mblur": pygame.Rect(col1, row(3), bw, bh),
        "toggle_blur": pygame.Rect(col2, row(0), bw, bh),
        "toggle_mirror": pygame.Rect(col2, row(1), bw, bh),
        "toggle_color": pygame.Rect(col2, row(2), bw, bh),
        "toggle_glow": pygame.Rect(col2, row(3), bw, bh),
    }, pygame.Rect(w - 245, panel_top + 74, 190, 12), pygame.Rect(0, panel_top, w, h - panel_top)


BUTTON_RECTS, SLIDER_RECT, PANEL_RECT = get_button_rects(WIDTH, HEIGHT)

# -------------------------------------------------------------------------
# AUDIO DSP PIPELINE
# -------------------------------------------------------------------------
WINDOW = np.hanning(2048)
LOG_EDGES = np.logspace(np.log10(35.0), np.log10(17500.0), SAMPLE_COUNT + 1)
BEAT_BANDS = {"sub_bass": (0, 5), "kick": (5, 13), "low_mid": (14, 30), "presence": (44, 76)}
beat_energy_history = {band: [] for band in BEAT_BANDS}
_bar_lo = _bar_hi = None


def _precompute_bar_bins(rate):
    global _bar_lo, _bar_hi
    freqs = np.fft.rfftfreq(2048, d=1.0 / rate)
    _bar_lo = np.searchsorted(freqs, LOG_EDGES[:-1])
    _bar_hi = np.searchsorted(freqs, LOG_EDGES[1:])


# -------------------------------------------------------------------------
# RENDER STATE & GLOBALS
# -------------------------------------------------------------------------
class RenderState:
    def __init__(self):
        self.dt = 0.016
        self.fft = np.zeros(SAMPLE_COUNT)
        self.prev_fft = np.zeros(SAMPLE_COUNT)
        self.rms = 0.0
        self.rms_history = []
        self.smooth_bars = np.zeros(SAMPLE_COUNT)
        self.peak_bars = np.zeros(SAMPLE_COUNT)
        self.peak_timer = np.zeros(SAMPLE_COUNT)
        self.is_beat = False
        self.smooth_beat = 0.0
        self.energy = 0.0
        self.shake = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.time = 0.0
        self.original_cover = None      # raw RGBA bytes for GL upload
        self.cover_size = None          # (w, h) of that buffer
        self.path_mode_index = 1
        self.flux_history = []

    @property
    def path_mode(self):
        return PATH_MODES[self.path_mode_index % len(PATH_MODES)]


play_channel = None
play_start_monotonic = None

RS = RenderState()
data, samplerate, sound = None, None, None
play_channel = None
play_pos_ms = None
play_pos_lock = threading.Lock()
play_start_monotonic = None


def _poll_play_pos(channel):
    """Background thread: poll Channel.get_pos() at ~20Hz and cache the result."""
    global play_pos_ms
    try:
        while channel is not None and channel.get_busy():
            try:
                pos = channel.get_pos()
            except Exception:
                pos = None
            with play_pos_lock:
                play_pos_ms = pos
            time.sleep(0.05)
    finally:
        with play_pos_lock:
            play_pos_ms = None
        # leave play_start_monotonic intact; reload_audio will reset it
title, artist = "Avee Pygame", "GPU Shader Visualizer"
volume = 0.82
show_ui = True

_font_cache = {}


def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        if not pygame.font.get_init():
            pygame.font.init()
        # pygame.font.SysFont can crash on some Windows font registries when
        # scanning installed fonts. Use the bundled default font first so the
        # visualizer starts reliably on every Pygame install, then fall back to
        # SysFont only if the default loader is unavailable.
        try:
            font = pygame.font.Font(None, size)
        except Exception:
            font = pygame.font.SysFont(None, size, bold=bold)
        font.set_bold(bold)
        _font_cache[key] = font
    return _font_cache[key]


# -------------------------------------------------------------------------
# AVEE-LIKE NODE ARCHITECTURE (Element & Composition)
# Elements no longer blit pixels: they *produce GPU data* (vec4-packed
# instance buffers / textures) and the GlRenderer executes them on the card.
# -------------------------------------------------------------------------
import moderngl
import gl_shaders as SH

TAU = math.tau


class Element:
    def __init__(self):
        self.enabled = True

    def update(self, rs: RenderState):
        pass


class AudioDataProviderElement(Element):
    def __init__(self):
        super().__init__()
        self.shake_target = np.array([0.0, 0.0])

    def update(self, rs: RenderState):
        global play_pos_ms, play_start_monotonic
        pos = None
        with play_pos_lock:
            pos = play_pos_ms
        if data is not None and sound and pos is not None and pos >= 0:
            rs.time = pos / 1000.0
        elif play_start_monotonic is not None:
            rs.time = time.monotonic() - play_start_monotonic
        else:
            rs.time += rs.dt
        if data is not None and sound and pygame.mixer.get_busy():
            idx = int((rs.time + (AUDIO_VIS_OFFSET_MS / 1000.0)) * samplerate)
            if idx + 2048 < len(data):
                chunk = data[idx:idx + 2048]
                try:
                    rs.rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
                except Exception:
                    rs.rms = float(np.sqrt(np.mean((chunk) ** 2)))
                rs.rms_history.append(rs.rms)
                if len(rs.rms_history) > 46:
                    rs.rms_history.pop(0)
                fft_raw = np.abs(np.fft.rfft(chunk * WINDOW))
                lo_i, hi_i = _bar_lo[:SAMPLE_COUNT], _bar_hi[:SAMPLE_COUNT]
                valid = hi_i > lo_i
                rs.fft[valid] = [np.mean(fft_raw[l:min(h, len(fft_raw))]) for l, h in zip(lo_i[valid], hi_i[valid])]
                single = (~valid) & (lo_i < len(fft_raw))
                rs.fft[single] = fft_raw[lo_i[single]]
                max_v = np.max(rs.fft)
                if max_v > 1e-4:
                    rs.fft /= max_v
                if BEAT_MODE == "flux":
                    flux = float(np.sum(np.maximum(rs.fft - rs.prev_fft, 0.0)))
                    rs.prev_fft = rs.fft.copy()
                    rs.flux_history.append(flux)
                    if len(rs.flux_history) > 46:
                        rs.flux_history.pop(0)
                    avg_flux = float(np.mean(rs.flux_history[:-1])) if len(rs.flux_history) > 6 else 0.0
                    if avg_flux > 1e-6 and flux > max(0.02, avg_flux * 1.5):
                        rs.is_beat = True
                        rs.smooth_beat = rs.smooth_beat * 0.34 + min(flux / (avg_flux * 2.0), 1.0) * 0.66
                    else:
                        rs.is_beat = False
                        rs.smooth_beat *= 0.88
                else:
                    avg_rms = float(np.mean(rs.rms_history[:-1])) if len(rs.rms_history) > 6 else 0.0
                    if avg_rms > 1e-8 and rs.rms > max(1e-4, avg_rms * 1.5):
                        rs.is_beat = True
                        rs.smooth_beat = rs.smooth_beat * 0.34 + min(rs.rms / (avg_rms * 2.0), 1.0) * 0.66
                    else:
                        rs.is_beat = False
                        rs.smooth_beat *= 0.88
        else:
            phase = rs.time * 2.0
            xs = np.linspace(0, 1, SAMPLE_COUNT)
            rs.fft = np.maximum(0.0, np.sin(xs * 9.0 + phase) * 0.25 + np.sin(xs * 31.0 - phase * 1.7) * 0.18)

        rs.energy = rs.energy * 0.82 + float(np.mean(rs.fft)) * 0.18

        rise_a = 0.18
        fall_rate = 2.4 + np.arange(SAMPLE_COUNT) / SAMPLE_COUNT
        up = rs.fft > rs.smooth_bars
        rs.smooth_bars[up] = rs.smooth_bars[up] * rise_a + rs.fft[up] * (1 - rise_a)
        rs.smooth_bars[~up] = np.maximum(0.0, rs.smooth_bars[~up] - fall_rate[~up] * rs.dt)

        jumped = rs.smooth_bars > rs.peak_bars
        rs.peak_bars[jumped] = rs.smooth_bars[jumped]
        old_timer = rs.peak_timer.copy()
        rs.peak_timer[jumped] = 0.28
        rs.peak_timer[~jumped] -= rs.dt
        decay = (~jumped) & (old_timer <= 0)
        rs.peak_bars[decay] = np.maximum(0.0, rs.peak_bars[decay] - 1.0 * rs.dt)

        if rs.is_beat:
            amt = 9.0 * rs.smooth_beat
            self.shake_target = np.array([random.uniform(-amt, amt), random.uniform(-amt, amt)])
        else:
            self.shake_target *= 0.82
        rs.shake = rs.shake * 0.62 + self.shake_target * 0.38
        rs.rotation += 0.005 + rs.smooth_beat * 0.025


class LetterPathCache:
    def __init__(self):
        self.key = None
        self.points = None   # (count, 2) float array

    def get_points(self, text, count):
        key = (text, count, WIDTH, HEIGHT)
        if key == self.key and self.points is not None:
            return self.points
        font = get_font(186, True)
        mask_surf = font.render(text, True, (255, 255, 255))
        pad = 28
        canvas = pygame.Surface((mask_surf.get_width() + pad * 2, mask_surf.get_height() + pad * 2), pygame.SRCALPHA)
        canvas.blit(mask_surf, (pad, pad))
        try:
            mask = pygame.mask.from_surface(canvas)
            outline_pts = mask.outline()
            if len(outline_pts) < 8:
                self.points = None
                return None
            arr = np.array(outline_pts, dtype=float)
            arr[:, 0] -= canvas.get_width() / 2
            arr[:, 1] -= canvas.get_height() / 2
            scale = min(WIDTH * 0.62 / max(1, canvas.get_width()), HEIGHT * 0.36 / max(1, canvas.get_height()))
            arr *= scale
            arr += np.array([WIDTH / 2, HEIGHT / 2 + 10], dtype=float)
            deltas = np.diff(arr, axis=0, append=arr[:1])
            dists = np.hypot(deltas[:, 0], deltas[:, 1])
            cum = np.cumsum(dists)
            total = cum[-1]
            if total <= 0:
                self.points = None
                return None
            positions = np.linspace(0, total, count, endpoint=False)
            idxs = np.clip(np.searchsorted(cum, positions), 1, len(arr) - 1)
            prev_idx = idxs - 1
            seg_len = dists[prev_idx]
            t = np.where(seg_len > 0, (positions - cum[prev_idx]) / np.maximum(seg_len, 1e-9), 0.0)
            pts = arr[prev_idx] + (arr[idxs] - arr[prev_idx]) * t[:, None]
            self.points = pts
            self.key = key
            return pts
        except Exception:
            self.points = None
            return None


class SegmentElement(Element):
    """FFT bars along LETTER/CIRCLE/LINE paths -> one instanced-quad batch.
    Instance layout (14 floats): center.xy | halfext.xy | dir.xy round alpha | rgba"""

    def __init__(self):
        super().__init__()
        self.letter_cache = LetterPathCache()
        self.instances = np.zeros((SAMPLE_COUNT, 14), dtype=np.float32)
        self.dots = np.zeros((SAMPLE_COUNT, 12), dtype=np.float32)
        self.dot_count = 0

    def _letter_text(self):
        clean = "".join(ch for ch in title.upper() if ch.isalnum())
        return (clean[:5] or "AVEE")

    def build(self, rs: RenderState):
        mode = rs.path_mode
        cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
        i = np.arange(SAMPLE_COUNT, dtype=float)
        pts = self.letter_cache.get_points(self._letter_text(), SAMPLE_COUNT) if mode == "LETTER" else None
        if mode == "LETTER" and pts is not None:
            pos = pts + rs.shake
            tang = np.roll(pts, -1, axis=0) - pts
            ln = np.hypot(tang[:, 0], tang[:, 1])
            ln[ln == 0] = 1.0
            tang = tang / ln[:, None]
            norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            flip = ((pos[:, 0] - cx) * norm[:, 0] + (pos[:, 1] - cy) * norm[:, 1]) < 0
            norm[flip] *= -1
        elif mode == "LINE":
            span = WIDTH * 0.76
            pos = np.empty((SAMPLE_COUNT, 2))
            pos[:, 0] = cx - span / 2 + span * (i / max(1, SAMPLE_COUNT - 1))
            pos[:, 1] = cy + np.sin((i / SAMPLE_COUNT) * TAU * 4 + rs.time * 1.6) * (22 + 28 * rs.energy)
            norm = np.tile([0.0, -1.0], (SAMPLE_COUNT, 1))
            tang = np.tile([1.0, 0.0], (SAMPLE_COUNT, 1))
        else:
            base_r = min(WIDTH, HEIGHT) * 0.23 + (rs.smooth_beat * 24)
            ang = (i / SAMPLE_COUNT) * TAU + rs.rotation
            pos = np.stack([cx + np.cos(ang) * base_r, cy + np.sin(ang) * base_r], axis=1)
            norm = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            tang = np.stack([-np.sin(ang), np.cos(ang)], axis=1)

        max_h = min(WIDTH, HEIGHT) * (0.23 if mode != "LETTER" else 0.15)
        bar_w = 3.6 if mode == "LETTER" else 5.2
        v = rs.smooth_bars
        h = np.maximum(2.0, v * max_h)
        wobble = np.sin(rs.time * 4.5 + i * 0.17) * rs.smooth_beat * 7.0
        base = pos + norm * wobble[:, None]
        tip = base + norm * h[:, None]

        ins = self.instances
        ins[:, 0:2] = (base + tip) / 2.0                     # quad center (pixels)
        ins[:, 2] = h / 2.0                                  # local-x half extent (long axis)
        ins[:, 3] = bar_w / 2.0                              # local-y half extent
        ins[:, 4], ins[:, 5] = norm[:, 0], norm[:, 1]        # rotation: long axis = normal
        ins[:, 6] = 0.0                                      # sharp rect
        ins[:, 7] = 0.92                                     # alpha
        hue = i / SAMPLE_COUNT
        ins[:, 8] = np.clip((90 + 90 * np.sin(hue * TAU + rs.time) + 65 * v) / 255.0, 0, 1)
        ins[:, 9] = (175 + 55 * np.sin(hue * TAU + 2.2)) / 255.0
        ins[:, 10] = 1.0
        ins[:, 11] = 235 / 255.0

        p = rs.peak_bars
        show = p > 0.05
        n = int(np.count_nonzero(show))
        self.dot_count = n
        if n:
            dctr = pos[show] + norm[show] * (p[show] * max_h + 9)[:, None]
            d = self.dots[:n]
            d[:, 0:2] = dctr
            d[:, 2] = 2.0
            d[:, 3] = 2.0
            d[:, 4], d[:, 5] = 1.0, 0.0
            d[:, 6] = 1.0                       # circle SDF
            d[:, 7] = 0.82
            d[:, 8:11] = 1.0
            d[:, 11] = 210 / 255.0
        return ins


class ParticlesElement(Element):
    """Beat-spawned particles -> one instanced-circle draw call (SoA arrays)."""
    MAX_PARTICLES = 1024

    def __init__(self):
        super().__init__()
        n = self.MAX_PARTICLES
        f32 = np.float32
        self.x = np.zeros(n, f32); self.y = np.zeros(n, f32)
        self.vx = np.zeros(n, f32); self.vy = np.zeros(n, f32)
        self.life = np.zeros(n, f32); self.maxlife = np.ones(n, f32)
        self.size = np.zeros(n, f32)
        self.count = 0
        self.instances = np.zeros((n, 12), dtype=f32)

    def update(self, rs):
        dt = rs.dt
        c = self.count
        if c:
            sl = slice(0, c)
            self.x[sl] += self.vx[sl] * dt
            self.y[sl] += self.vy[sl] * dt
            self.vx[sl] *= 0.992
            self.vy[sl] *= 0.992
            self.life[sl] -= dt
            alive = self.life[:c] > 0
            if not alive.all():
                for a in (self.x, self.y, self.vx, self.vy, self.life, self.maxlife, self.size):
                    a[:c] = np.concatenate([a[:c][alive], a[c:]])
                self.count = c = int(alive.sum())
        if rs.is_beat and rs.smooth_beat > 0.35:
            spawn = min(int(5 + rs.smooth_beat * 18), self.MAX_PARTICLES - c)
            if spawn > 0:
                cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
                a = np.random.uniform(0, TAU, spawn)
                s = np.random.uniform(70, 260, spawn) * (0.4 + rs.smooth_beat)
                sl = slice(c, c + spawn)
                self.x[sl] = cx; self.y[sl] = cy
                self.vx[sl] = np.cos(a) * s
                self.vy[sl] = np.sin(a) * s
                life = np.random.uniform(0.7, 2.4, spawn)
                self.life[sl] = life; self.maxlife[sl] = life
                self.size[sl] = np.random.uniform(1.5, 4.5, spawn)
                self.count = c + spawn

    def build(self, rs):
        n = self.count
        if n == 0:
            return None, 0
        frac = self.life[:n] / self.maxlife[:n]
        ins = self.instances[:n]
        ins[:, 0] = self.x[:n]
        ins[:, 1] = self.y[:n]
        r = np.maximum(1.0, self.size[:n] * frac)
        ins[:, 2] = r
        ins[:, 3] = r
        ins[:, 4] = 1.0
        ins[:, 5] = 0.0
        ins[:, 6] = 1.0                       # circle
        ins[:, 7] = 0.82 * frac
        ins[:, 8], ins[:, 9], ins[:, 10] = 0.35, 0.86, 1.0
        ins[:, 11] = 1.0
        return ins, n


class ImageElement(Element):
    """Cover disc + halo are fully procedural in IMAGE_FS; this element only
    keeps the GL texture in sync with the loaded artwork."""

    def __init__(self):
        super().__init__()
        self.uploaded_id = None

    def sync_texture(self, rs, renderer):
        cov = id(rs.original_cover)
        if cov != self.uploaded_id:
            self.uploaded_id = cov
            renderer.upload_cover(rs.original_cover, rs.cover_size)


# -------------------------------------------------------------------------
# EFFECT TOGGLES — these no longer copy surfaces on the CPU; each one just
# drives an amount uniform inside the GPU composite shader.
# Slots match u_amounts[] in gl_shaders.COMPOSITE_FS.
# -------------------------------------------------------------------------
class EffectToggle:
    def __init__(self, slot, enabled=False):
        self.enabled = enabled
        self.slot = slot


PPE_GLOW = EffectToggle(0)
PPE_BLUR = EffectToggle(1)
PPE_MBlur = EffectToggle(2)
PPE_Color = EffectToggle(3)
PPE_Mirror = EffectToggle(4)
PPE_RGB = EffectToggle(5)
EFFECTS = {"toggle_glow": PPE_GLOW, "toggle_blur": PPE_BLUR, "toggle_mblur": PPE_MBlur,
           "toggle_color": PPE_Color, "toggle_mirror": PPE_Mirror, "toggle_rgb": PPE_RGB}


# -------------------------------------------------------------------------
# SCENE BUILDER
# -------------------------------------------------------------------------
CompAudio = AudioDataProviderElement()
CompParts = ParticlesElement()
CompBars = SegmentElement()
CompImage = ImageElement()
CompTextEnabled = True     # TEXT button toggles the title/artist overlay


def scene_update(rs):
    CompAudio.update(rs)
    if CompParts.enabled:
        CompParts.update(rs)


# -------------------------------------------------------------------------
# ASSET HELPERS
# -------------------------------------------------------------------------
def extract_cover(file):
    """Return (rgba_bytes, (w, h)) ready for a direct GL texture upload."""
    try:
        ext = os.path.splitext(file)[1].lower()
        raw = None
        if ext == ".mp3":
            tags = ID3(file)
            for tag in tags.values():
                if isinstance(tag, APIC):
                    raw = tag.data
                    break
        elif ext == ".flac":
            audio = FLAC(file)
            if audio.pictures:
                raw = audio.pictures[0].data
        elif ext in (".m4a", ".mp4", ".aac"):
            audio = MP4(file)
            if "covr" in audio:
                raw = bytes(audio["covr"][0])
        if raw is None:
            return None, None
        img = pygame.image.load(io.BytesIO(raw)).convert_alpha()
        side = 512
        img = pygame.transform.smoothscale(img, (side, side))
        return pygame.image.tostring(img, "RGBA", False), (side, side)
    except Exception:
        return None, None


def reload_audio(file):
    global data, samplerate, sound, title, artist, RS
    global play_channel, play_start_monotonic, play_pos_ms
    try:
        data, samplerate = sf.read(file)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        pygame.mixer.quit()
        pygame.mixer.init(samplerate, -16, 2, 2048)
        sound = pygame.mixer.Sound(file)
        sound.set_volume(volume)
        ch = sound.play()
        play_channel = ch
        play_start_monotonic = time.monotonic()
        with play_pos_lock:
            play_pos_ms = 0
        if ch:
            try:
                ch.set_volume(volume)
            except Exception:
                pass
            try:
                t = threading.Thread(target=_poll_play_pos, args=(ch,), daemon=True)
                t.start()
            except Exception:
                pass
        title = os.path.splitext(os.path.basename(file))[0]
        artist = "Loaded track"
        RS.time = 0.0
        RS.original_cover, RS.cover_size = extract_cover(file)
        _precompute_bar_bins(samplerate)
    except Exception as e:
        print(f"File load error: {e}")


# -------------------------------------------------------------------------
# GPU RENDERER (moderngl / OpenGL 3.3 core)
# Frame graph per tick:
#   scene FBO   : grid shader + cover shader + instanced bars/dots/particles
#   comp FBO    : one fullscreen composite pass (glow/blur/motion blur/
#                 mirror/color correction/RGB split/vignette) reading a ring
#                 of half-res history textures for the motion blur
#   framebuffer : blit(comp) then alpha-blend the UI overlay texture
# -------------------------------------------------------------------------
class GlRenderer:
    CORNERS = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
    HIST_LEN = 8

    def __init__(self, ctx, w, h):
        self.ctx = ctx
        self.w, self.h = w, h
        ctx.enable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vbo = ctx.buffer(self.CORNERS.tobytes())
        self._build_programs()
        self._build_fbos(w, h)

        # bars: 14 floats per instance
        self.inst_data = ctx.buffer(b" " * (SAMPLE_COUNT * 56), reserve=True)
        self.inst_vao = ctx.vertex_array(
            self.prog_inst,
            [(self.quad_vbo, "2f / 14f", "in_corner"),
             (self.inst_data, "4f 4f 4f / 0f", "in_center_wh", "in_dir_half", "in_color")],
        )
        # dots + particles share a 12-float layout
        self.dot_data = ctx.buffer(b" " * (SAMPLE_COUNT * 48), reserve=True)
        self.dot_vao = ctx.vertex_array(
            self.prog_inst,
            [(self.quad_vbo, "2f / 12f", "in_corner"),
             (self.dot_data, "4f 4f 4f / 0f", "in_center_wh", "in_dir_half", "in_color")],
        )
        self.part_data = ctx.buffer(b" " * (CompParts.MAX_PARTICLES * 48), reserve=True)
        self.part_vao = ctx.vertex_array(
            self.prog_inst,
            [(self.quad_vbo, "2f / 12f", "in_corner"),
             (self.part_data, "4f 4f 4f / 0f", "in_center_wh", "in_dir_half", "in_color")],
        )

        self.cover_tex = None
        self.ui_tex = None
        self.ui_dirty = True
        self._hist_write = 0
        self.prog_comp["u_scene"] = 0
        for k, name in enumerate(("u_h0", "u_h1", "u_h2", "u_h3", "u_h4", "u_h5", "u_h6", "u_h7")):
            self.prog_comp[name] = k + 1
        self.prog_blit["u_tex"] = 0

    def _build_programs(self):
        c = self.ctx
        self.prog_grid = c.program(vertex_shader=SH.VERTEX_QUAD, fragment_shader=SH.GRID_FS)
        self.prog_inst = c.program(vertex_shader=SH.INSTANCED_VS, fragment_shader=SH.INSTANCED_FS)
        self.prog_image = c.program(vertex_shader=SH.VERTEX_QUAD, fragment_shader=SH.IMAGE_FS)
        self.prog_comp = c.program(vertex_shader=SH.VERTEX_QUAD, fragment_shader=SH.COMPOSITE_FS)
        self.prog_blit = c.program(vertex_shader=SH.VERTEX_QUAD, fragment_shader=SH.BLIT_FS)
        self.quad_full = c.simple_vertex_array(self.prog_grid, self.quad_vbo, "in_pos")

    def _new_fbo(self, w, h):
        tex = self.ctx.texture((w, h), 4, dtype="f1")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return self.ctx.framebuffer(color_attachments=[tex])

    def _build_fbos(self, w, h):
        self.scene = self._new_fbo(w, h)
        self.comp_cur = self._new_fbo(w, h)
        self.hist_fbos = [self._new_fbo(max(2, w // 2), max(2, h // 2)) for _ in range(self.HIST_LEN)]
        self.hist_tex = [fb.color_attachments[0] for fb in self.hist_fbos]

    def release_fbos(self):
        for fb in [self.scene, self.comp_cur] + self.hist_fbos:
            fb.release()

    def resize(self, w, h):
        if (w, h) == (self.w, self.h):
            return
        self.release_fbos()
        self._build_fbos(w, h)
        self.w, self.h = w, h
        self.ui_dirty = True

    def upload_cover(self, buf, size):
        if self.cover_tex:
            self.cover_tex.release()
            self.cover_tex = None
        if buf and size:
            self.cover_tex = self.ctx.texture(size, 4, data=buf)
            self.cover_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def set_ui_surface(self, surf):
        """Upload a pygame RGBA surface as the UI overlay texture."""
        w, h = surf.get_size()
        raw = pygame.image.tostring(surf, "RGBA", False)
        if self.ui_tex is None or self.ui_tex.size != (w, h):
            if self.ui_tex:
                self.ui_tex.release()
            self.ui_tex = self.ctx.texture((w, h), 4, data=raw)
            self.ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.ui_tex.write(raw)
        self.ui_dirty = False

    def _quad(self, prog):
        return self.ctx.simple_vertex_array(prog, self.quad_vbo, "in_pos")

    def render_scene(self, rs):
        c = self.ctx
        self.scene.use()
        c.set_blend_func(moderngl.ONE, moderngl.ZERO)
        c.clear(0, 0, 0, 1)

        # background perspective grid — pure fragment shader
        self.prog_grid["u_time"].value = float(rs.time)
        self.prog_grid["u_aspect"].value = self.h / self.w
        self.quad_full.render()

        c.set_blend_func(moderngl.ONE, moderngl.ONE)  # premultiplied additive

        # cover disc + pulsing halo — procedural fragment shader
        if CompImage.enabled:
            CompImage.sync_texture(rs, self)
            cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
            rad = min(self.w, self.h) * 0.145 + rs.smooth_beat * 18
            p = self.prog_image
            p["u_tex"].value = 0
            p["u_hasTex"].value = 1 if self.cover_tex else 0
            p["u_size"].value = (self.w, self.h)
            p["u_center"].value = (float(cx), float(cy))
            p["u_half"].value = (float(rad), float(rad))
            p["u_beat"].value = float(rs.smooth_beat)
            p["u_energy"].value = float(rs.energy)
            if self.cover_tex:
                self.cover_tex.use(0)
            vao = self._quad(p)
            vao.render()
            vao.release()

        # FFT bars — one instanced draw call
        if CompBars.enabled:
            ins = CompBars.build(rs)
            self.inst_data.write(np.ascontiguousarray(ins).tobytes())
            self.prog_inst["u_size"].value = (self.w, self.h)
            self.inst_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=SAMPLE_COUNT)

            # peak dots — instanced circles
            n = CompBars.dot_count
            if n:
                self.dot_data.write(CompBars.dots[:n].tobytes())
                self.dot_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)

        # particles — instanced circles
        if CompParts.enabled:
            arr, n = CompParts.build(rs)
            if n:
                self.part_data.write(np.ascontiguousarray(arr).tobytes())
                self.part_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)

    def composite(self, rs):
        c = self.ctx
        self.comp_cur.use()
        c.set_blend_func(moderngl.ONE, moderngl.ZERO)
        p = self.prog_comp
        self.scene.color_attachments[0].use(0)
        p["u_size"].value = (self.w, self.h)
        p["u_time"].value = float(rs.time)
        p["u_beat"].value = float(rs.smooth_beat)
        p["u_energy"].value = float(rs.energy)
        p["u_shakeX"].value = float(rs.shake[0])
        p["u_shakeY"].value = float(rs.shake[1])
        amounts = [0.0] * 6
        for eff in EFFECTS.values():
            amounts[eff.slot] = 1.0 if eff.enabled else 0.0
        p["u_amounts"].value = amounts
        # RGB-split offset in normalized uv units (CPU version used 2..27 px)
        p["u_rgbAmt"].value = (2.0 + 18.0 * rs.smooth_beat + 7.0 * rs.energy) / self.w

        if PPE_MBlur.enabled:
            for k in range(self.HIST_LEN):
                idx = (self._hist_write - 1 - k) % self.HIST_LEN
                self.hist_tex[idx].use(k + 1)
        else:
            self.hist_tex[0].use_range(1, self.HIST_LEN)
        vao = self._quad(p)
        vao.render()
        vao.release()

        # record frame into the history ring (GPU-side downsampled blit)
        slot = self._hist_write % self.HIST_LEN
        self._hist_write += 1
        self.hist_fbos[slot].use()
        pb = self.prog_blit
        self.comp_cur.color_attachments[0].use(0)
        vao = self._quad(pb)
        vao.render()
        vao.release()

    def present(self):
        c = self.ctx
        c.reset_viewport()
        c.set_blend_func(moderngl.ONE, moderngl.ZERO)
        c.clear(0, 0, 0, 1)
        pb = self.prog_blit
        self.comp_cur.color_attachments[0].use(0)
        vao = self._quad(pb)
        vao.render()
        vao.release()
        if show_ui and self.ui_tex:
            c.set_blend_func(moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ui_tex.use(0)
            vao = self._quad(pb)
            vao.render()
            vao.release()


RENDERER = None


# -------------------------------------------------------------------------
# UI DRAWING — rendered by pygame into an offscreen surface at most ~10x/sec
# and uploaded as a GL texture; the GPU blends it over the composited scene.
# -------------------------------------------------------------------------
def build_ui_surface(clock, btn_states, fps_text):
    surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    panel = pygame.Surface((PANEL_RECT.width, PANEL_RECT.height), pygame.SRCALPHA)
    panel.fill((6, 9, 18, 218))
    pygame.draw.rect(panel, (0, 210, 255, 75), panel.get_rect(), 1, border_radius=18)
    surf.blit(panel, PANEL_RECT.topleft)

    header = get_font(21, True).render("AVEE PLAYER PY \u2022 GPU/OPENGL SHADER VISUALIZER", True, (240, 252, 255))
    surf.blit(header, (24, PANEL_RECT.y + 16))
    subtitle = get_font(13).render(
        "Instanced quads + FBO post FX on the GPU: glow/blur/motion blur/mirror/rgb split via GLSL",
        True, (120, 205, 230))
    surf.blit(subtitle, (24, PANEL_RECT.y + 40))

    labels = {
        "open": "OPEN", "path_mode": f"PATH {RS.path_mode}", "toggle_bars": "BARS", "toggle_image": "IMAGE",
        "toggle_part": "PARTICLES", "toggle_text": "TEXT", "toggle_rgb": "RGB SPLIT", "toggle_mblur": "MOTION BLUR",
        "toggle_blur": "SOFT BLUR", "toggle_mirror": "MIRROR", "toggle_color": "COLOR", "toggle_glow": "GLOW",
    }
    for name, rect in BUTTON_RECTS.items():
        if name == "open" or name == "path_mode":
            active = True
        elif name == "toggle_image":
            active = CompImage.enabled
        elif name == "toggle_text":
            active = CompTextEnabled
        else:
            active = btn_states[name].enabled
        bg_col = (0, 150, 170) if active else (30, 34, 48)
        hi_col = (55, 240, 255) if active else (82, 88, 108)
        pygame.draw.rect(surf, bg_col, rect, border_radius=8)
        pygame.draw.rect(surf, hi_col, rect, 1, border_radius=8)
        txt = get_font(11, True).render(labels[name], True, (245, 252, 255) if active else (176, 184, 198))
        surf.blit(txt, txt.get_rect(center=rect.center))

    pygame.draw.rect(surf, (38, 44, 58), SLIDER_RECT, border_radius=5)
    fill_w = int(volume * SLIDER_RECT.width)
    if fill_w > 0:
        pygame.draw.rect(surf, (0, 210, 255), (SLIDER_RECT.x, SLIDER_RECT.y, fill_w, SLIDER_RECT.height), border_radius=5)
    vol_label = get_font(13, True).render(f"VOLUME {int(volume * 100)}%", True, (220, 245, 255))
    surf.blit(vol_label, (SLIDER_RECT.x, SLIDER_RECT.y - 24))
    fps = get_font(13).render(fps_text, True, (145, 180, 195))
    surf.blit(fps, (SLIDER_RECT.x, SLIDER_RECT.y + 20))

    if CompTextEnabled:
        t_font, a_font = get_font(30, True), get_font(18, False)
        t_surf = t_font.render(title[:34], True, (255, 255, 255))
        a_surf = a_font.render(f"{artist[:40]}  \u2022  {RS.path_mode} path  \u2022  H hides UI", True, (125, 220, 255))
        surf.blit(t_surf, (46, HEIGHT - 112))
        surf.blit(a_surf, (48, HEIGHT - 74))
    return surf


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
def _create_gl_context():
    """Create the OpenGL context through pygame/SDL; fall back to a moderngl
    standalone (offscreen) context when no display is available so the app
    still runs headless."""
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        return moderngl.create_context(), False
    except Exception as e:
        print(f"[gpu] windowed GL unavailable ({e}); standalone offscreen context")
        return moderngl.create_standalone_context(), True


def main():
    global volume, show_ui, WIDTH, HEIGHT, CENTER, BUTTON_RECTS, SLIDER_RECT, PANEL_RECT
    global RENDERER
    pygame.init()
    try:
        pygame.mixer.init(-16, 2, 2048)
    except Exception:
        pass
    ctx, offscreen = _create_gl_context()
    print(f"[gpu] {ctx.info['GL_VERSION'].strip()} \u2014 {ctx.info['GL_RENDERER'].strip()}")
    pygame.display.set_caption("Avee Player - GPU Shader Visualizer (OpenGL)")
    clock = pygame.time.Clock()
    _precompute_bar_bins(44100)
    RENDERER = GlRenderer(ctx, WIDTH, HEIGHT)

    btn_states = {
        "toggle_bars": CompBars,
        "toggle_image": CompImage,
        "toggle_part": CompParts,
        **EFFECTS,
    }

    running = True
    ui_timer = 0.0
    while running:
        RS.dt = min(0.05, clock.tick(60) / 1000.0)
        scene_update(RS)
        RENDERER.render_scene(RS)
        RENDERER.composite(RS)

        # refresh the UI overlay texture ~10x/sec (or immediately when dirty)
        ui_timer -= RS.dt
        if show_ui and (ui_timer <= 0.0 or RENDERER.ui_dirty):
            ui_timer = 0.1
            RENDERER.set_ui_surface(build_ui_surface(
                clock, btn_states, f"{clock.get_fps():05.1f} FPS  \u2022  GPU ON  \u2022  H hide UI"))
        RENDERER.present()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not offscreen:
                WIDTH, HEIGHT = max(720, event.w), max(560, event.h)
                CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
                BUTTON_RECTS, SLIDER_RECT, PANEL_RECT = get_button_rects(WIDTH, HEIGHT)
                RENDERER.resize(WIDTH, HEIGHT)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    show_ui = not show_ui
                    RENDERER.ui_dirty = True
                elif event.key == pygame.K_TAB:
                    RS.path_mode_index = (RS.path_mode_index + 1) % len(PATH_MODES)
                    RENDERER.ui_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and show_ui:
                pos = event.pos
                if SLIDER_RECT.collidepoint(pos):
                    volume = max(0.0, min(1.0, (pos[0] - SLIDER_RECT.x) / SLIDER_RECT.width))
                    if play_channel:
                        try:
                            play_channel.set_volume(volume)
                        except Exception:
                            pass
                    elif sound:
                        sound.set_volume(volume)
                    RENDERER.ui_dirty = True
                for name, rect in BUTTON_RECTS.items():
                    if rect.collidepoint(pos):
                        if name == "open":
                            root = tk.Tk()
                            root.withdraw()
                            f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.flac *.m4a *.aac *.wav *.ogg")])
                            root.destroy()
                            if f:
                                reload_audio(f)
                        elif name == "path_mode":
                            RS.path_mode_index = (RS.path_mode_index + 1) % len(PATH_MODES)
                        elif name == "toggle_text":
                            CompTextEnabled = not CompTextEnabled
                        elif name in btn_states:
                            btn_states[name].enabled = not btn_states[name].enabled
                        RENDERER.ui_dirty = True

        if offscreen:
            # headless smoke test: save one composited frame and exit
            raw = ctx.framebuffer(color_attachments=[ctx.texture((WIDTH, HEIGHT), 4)])
            ctx.copy_framebuffer(raw, RENDERER.comp_cur)
            surf = pygame.image.frombuffer(raw.color_attachments[0].read(), (WIDTH, HEIGHT), "RGBA")
            pygame.image.save(surf, "/tmp/gpu_frame.png")
            print("[gpu] wrote /tmp/gpu_frame.png (headless smoke test)")
            raw.release()
            running = False

        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
