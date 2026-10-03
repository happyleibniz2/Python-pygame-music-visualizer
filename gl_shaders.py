"""GLSL 330 core shader sources for the GPU visualizer pipeline.

Everything that used to be a per-pixel CPU loop in pygame (glow, RGB slit,
motion blur, mirror, soft blur, color correction) is now a fragment shader
pass over an RGBA16F ping-pong framebuffer.  Bars/particles are instanced
quads; the cover art and the whole UI panel are textured quads.
"""

VERTEX_QUAD = """#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

GRID_FS = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform float u_time;
uniform float u_aspect;

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.016, 0.020, 0.039);
    float horizon = 0.42 + sin(u_time * 0.7) * 0.015;
    if (uv.y < horizon) {
        col += vec3(0.0, 0.012, 0.03) * (horizon - uv.y) / horizon;
    } else {
        float t = (uv.y - horizon) / (1.0 - horizon);
        // perspective scroll: rows bunch up near the horizon
        float flow = fract(t * 7.0 - u_time * 0.35);
        float rowline = smoothstep(0.96, 1.0, flow) * t;
        float cx = (uv.x - 0.5) * (1.0 - t) + 0.5;
        float skew = sin(u_time * 0.25) * 0.08;
        float cols = abs(fract((cx - 0.5 - skew * t) * 11.0) - 0.5);
        float colline = (1.0 - smoothstep(0.0, 0.02, cols)) * t;
        col += vec3(0.0, 0.5, 0.75) * (rowline * 0.5 + colline * 0.22);
    }
    frag_color = vec4(col, 1.0);
}
"""

INSTANCED_VS = """#version 330
in vec2 in_corner;          // unit quad corner (-1..1)
in vec4 in_center_wh;       // pixel-space center x,y + half width/height
in vec4 in_dir_half;        // long-axis direction xy, z unused, w = alpha
in vec4 in_color;           // rgba
out vec2 v_local;
out vec4 v_color;
out float v_round;
uniform vec2 u_size;

void main() {
    vec2 l = in_corner * vec2(in_center_wh.z, in_center_wh.w);
    mat2 rot = mat2(in_dir_half.x, -in_dir_half.y, in_dir_half.y, in_dir_half.x);
    vec2 p = in_center_wh.xy + rot * l;
    v_local = in_corner;
    v_color = in_color;
    v_round = in_dir_half.z;   // 0 = sharp rect, 1 = rounded/circle SDF
    vec2 clip = (p / u_size) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}
"""

INSTANCED_FS = """#version 330
in vec2 v_local;
in vec4 v_color;
in float v_round;
out vec4 frag_color;

float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    float a = v_color.a;
    if (v_round > 0.5) {
        // rounded-rect SDF in unit-corner space; r==0.98 -> circle
        float d = sdRoundBox(v_local, vec2(1.0), 0.98);
        a *= 1.0 - smoothstep(-0.04, 0.04, d);
    }
    if (a <= 0.002) discard;
    frag_color = vec4(v_color.rgb * a, a);   // premultiplied for additive blend
}
"""

IMAGE_FS = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_tex;
uniform int u_hasTex;
uniform vec2 u_size;         // scene size in pixels
uniform vec2 u_center;       // pixels, y-down
uniform vec2 u_half;         // radius in pixels
uniform float u_beat;
uniform float u_energy;

void main() {
    vec2 p = gl_FragCoord.xy;
    p.y = u_size.y - p.y;                       // to y-down pixel coords
    vec2 d = (p - u_center) / u_half;           // normalized disk space
    float r = length(d);
    // pulsing halo glow around the disc
    float gA = exp(-pow(max(r - 1.0, 0.0) * 2.6, 2.0)) * (0.22 + 0.16 * u_beat);
    vec4 col = vec4(vec3(0.0, 0.82, 1.0) * gA, gA);

    if (r <= 1.0) {
        float edge = 1.0 - smoothstep(0.985, 1.0, r);
        if (u_hasTex == 1) {
            vec2 cuv = d * 0.5 + 0.5;
            vec3 t = texture(u_tex, cuv).rgb;
            col = vec4(t * edge, edge);
        } else {
            col = vec4(vec3(0.078, 0.086, 0.117) * edge, edge);
            float ring = abs(r - 0.94);
            if (ring < 0.035) col.rgb += vec3(0.0, 0.92, 1.0) * (1.0 - ring / 0.035) * edge;
        }
    }
    if (col.a <= 0.002) discard;
    frag_color = col;
}
"""

COMPOSITE_FS = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_scene;
uniform sampler2D u_h0;
uniform sampler2D u_h1;
uniform sampler2D u_h2;
uniform sampler2D u_h3;
uniform sampler2D u_h4;
uniform sampler2D u_h5;
uniform sampler2D u_h6;
uniform sampler2D u_h7;
uniform vec2 u_size;
uniform float u_time;
uniform float u_beat;
uniform float u_energy;
uniform float u_shakeX;
uniform float u_shakeY;
uniform float u_amounts[6];   // glow, blur, mblur, mirror, color, rgb
uniform float u_rgbAmt;

vec3 sampleS(vec2 uv) {
    return texture(u_scene, clamp(uv, vec2(0.002), vec2(0.998))).rgb;
}

void main() {
    vec2 uv = v_uv;
    float ar = u_size.y / u_size.x;
    vec2 offPx = vec2(1.0) / u_size;

    float mirror = u_amounts[4];
    if (mirror > 0.001) {
        vec2 muv = uv;
        muv.x = 1.0 - muv.x;
        vec3 c = mix(sampleS(uv), sampleS(muv), mirror);
        frag_color = vec4(c, 1.0);
        return;
    }

    vec3 base = sampleS(uv);

    // soft blur: 9-tap gaussian, strength beat-reactive
    float blur = u_amounts[1];
    if (blur > 0.001) {
        vec2 s = offPx * 3.2;
        vec3 b = base * 4.0;
        b += (sampleS(uv + vec2(s.x, 0)) + sampleS(uv - vec2(s.x, 0)) +
              sampleS(uv + vec2(0, s.y * ar)) + sampleS(uv - vec2(0, s.y * ar))) * 2.0;
        b += (sampleS(uv + s) + sampleS(uv - s) +
              sampleS(uv + vec2(s.x, -s.y * ar)) + sampleS(uv - vec2(s.x, -s.y * ar)));
        base = mix(base, b / 16.0, blur);
    }

    // bloom/glow: cheap dual-radius tap-lum blur added back
    float glow = u_amounts[0];
    if (glow > 0.001) {
        vec3 g = vec3(0.0);
        for (int i = 0; i < 8; i++) {
            float fi = float(i);
            vec2 dir = vec2(cos(fi * 0.7853981634), sin(fi * 0.7853981634));
            vec2 o1 = dir * offPx * vec2(7.0, 7.0 * ar);
            vec2 o2 = dir * offPx * vec2(16.0, 16.0 * ar);
            g += sampleS(uv + o1) * 0.55 + sampleS(uv + o2) * 0.45;
        }
        g /= 8.0;
        float lum = dot(g, vec3(0.2126, 0.7152, 0.0722));
        g *= smoothstep(0.18, 0.85, lum);
        base += g * glow * (0.75 + 0.9 * u_beat);
    }

    // motion blur: additive history frames with shake-based offset
    float mb = u_amounts[2];
    if (mb > 0.001) {
        vec2 sh = vec2(u_shakeX, -u_shakeY) / u_size;
        for (int i = 0; i < 8; i++) {
            float fi = float(i) + 1.0;
            float wgt = (9.0 - fi) / 8.0;
            vec2 huv = clamp(uv - sh * 0.18 * fi, vec2(0.002), vec2(0.998));
            vec3 h;
            if (i == 0)      h = texture(u_h0, huv).rgb;
            else if (i == 1) h = texture(u_h1, huv).rgb;
            else if (i == 2) h = texture(u_h2, huv).rgb;
            else if (i == 3) h = texture(u_h3, huv).rgb;
            else if (i == 4) h = texture(u_h4, huv).rgb;
            else if (i == 5) h = texture(u_h5, huv).rgb;
            else if (i == 6) h = texture(u_h6, huv).rgb;
            else             h = texture(u_h7, huv).rgb;
            base += h * wgt * mb * 0.32;
        }
    }

    // color correction: cyan tint + beat-pumped contrast/saturation
    float cc = u_amounts[3];
    if (cc > 0.001) {
        vec3 tinted = base + vec3(0.0, 0.09, 0.22) * cc * (0.5 + u_beat);
        float k = 1.0 + cc * (0.35 + 0.5 * u_beat);
        vec3 cont = (tinted - 0.5) * k + 0.5;
        float luma = dot(cont, vec3(0.2126, 0.7152, 0.0722));
        base = mix(vec3(luma), cont, 1.0 + cc * 0.35);
    }

    // RGB split: three chroma taps in different directions
    float rgb = u_amounts[5] * u_rgbAmt;
    if (rgb > 0.0005) {
        vec2 amt = vec2(rgb, rgb * 0.22);
        float rC = sampleS(clamp(uv - amt, vec2(0.002), vec2(0.998))).r;
        float bC = sampleS(clamp(uv + amt, vec2(0.002), vec2(0.998))).b;
        base.r = rC;
        base.b = bC;
    }

    // vignette
    vec2 vd = (uv - 0.5) * vec2(1.0, ar);
    base *= 1.0 - dot(vd, vd) * 0.55;

    frag_color = vec4(base, 1.0);
}
"""

BLIT_FS = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_tex;
void main() {
    frag_color = texture(u_tex, v_uv);
}
"""
