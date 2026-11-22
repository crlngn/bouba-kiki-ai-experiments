"""Draw every shape set used in phase 1 and write stimuli/shapes.csv.

Sets:
  chen     Radial frequency outlines rebuilt from the equations in Chen, Huang, Woods & Spence (2016):
           r(t) = r_mean * (1 + A * x(t)), x(t) = sum_{k=0..n} (-1)^k sin((2k+1) w t) / (2k+1)^2,
           with frequency w in 4..9, amplitude A in 0.1..0.4 and spikiness n in {0, 1, 30}.
           Used for the graded task, so results compare directly with their US and Taiwan data.
  classic  Round/sharp pairs for the two-choice image task:
             kohler       redrawn after Köhler's (1929) maluma/takete figure, as reproduced in the PerForm thesis
             chen_fig1    the two shapes in Chen et al. (2016) Fig. 1, cropped from the CC BY 4.0 figure
             maurer_*     our own pairs in the style of Maurer, Pathman & Mondloch (2006): an irregular
                          jagged star and a wavy blob drawn through the same random points
  thesis   The six outlines from PerForm (Rozendo Xavier dos Santos, 2019) Fig. 26, redrawn, with the
           24 participants' sharpness ratings (stimuli/thesis_shape_ratings.csv).
  mixed    The earlier filled radial shapes whose spikes sit on a round body (generated in make_stimuli.py).

Classic, chen and thesis shapes are black outlines on white, like the published stimuli.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SHAPES = ROOT / "stimuli" / "shapes"
SIZE = 384
SS = 4
STROKE = 4

CHEN_FREQ = [4, 5, 6, 7, 8, 9]
CHEN_AMP = [0.1, 0.2, 0.3, 0.4]
CHEN_SPIKE = [0, 1, 30]

KOHLER_TAKETE = [(690, 620), (1045, 422), (900, 460), (1090, 588), (760, 517), (1063, 492), (690, 620)]
KOHLER_MALUMA = [((480, 522), 140, 82, -8), ((548, 502), 92, 150, 14)]

THESIS_SHAPES = ["q1_circle", "q5_rounded_star", "q6_leaf", "q3_s_curve", "q4_star", "q2_tall_star"]


def _canvas():
    """A supersampled white canvas and its drawing context."""
    img = Image.new("L", (SIZE * SS, SIZE * SS), 255)
    return img, ImageDraw.Draw(img)


def _save(img, path):
    """Downsample and save as RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.resize((SIZE, SIZE), Image.LANCZOS).convert("RGB").save(path)


def _fit(points, margin=0.14):
    """Scale and centre a list of (x, y) points (y up) to fill the canvas, keeping aspect ratio."""
    pts = np.asarray(points, dtype=float)
    lo, hi = pts.min(0), pts.max(0)
    scale = (1 - 2 * margin) * SIZE * SS / (hi - lo).max()
    centre = (lo + hi) / 2
    out = (pts - centre) * scale
    out[:, 1] *= -1
    return [tuple(p) for p in out + SIZE * SS / 2]


def _outline(draw, pts, closed=True):
    """Draw a polyline outline with round joins."""
    pts = list(pts) + ([pts[0]] if closed else [])
    draw.line(pts, fill=0, width=STROKE * SS, joint="curve")
    r = STROKE * SS / 2
    for x, y in (pts[0], pts[-1]):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=0)


def chen_points(freq, amp, spike, n=4000):
    """Outline points of a Chen et al. (2016) RF pattern."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    phase = freq * t + math.pi / 2
    x = sum((-1) ** k * np.sin((2 * k + 1) * phase) / (2 * k + 1) ** 2 for k in range(spike + 1))
    r = 1 + amp * x
    return np.c_[r * np.cos(t), r * np.sin(t)]


def draw_chen():
    """All 72 Chen RF outlines at a common base radius."""
    rows = []
    for f in CHEN_FREQ:
        for a in CHEN_AMP:
            for s in CHEN_SPIKE:
                img, d = _canvas()
                pts = chen_points(f, a, s) * (SIZE * SS * 0.30)
                pts[:, 1] *= -1
                _outline(d, [tuple(p) for p in pts + SIZE * SS / 2])
                sid = f"chen_f{f}_a{int(a * 100):02d}_n{s:02d}"
                path = SHAPES / "chen" / f"{sid}.png"
                _save(img, path)
                rows.append({"shape_id": sid, "set": "chen", "file": str(path.relative_to(ROOT)), "frequency": f,
                             "amplitude": a, "spikiness": s, "role": "grid", "pair": "", "source": "Chen et al. (2016) equations"})
    return rows


def _ellipse(center, rx, ry, rot, n=400):
    """Points of a rotated ellipse."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.c_[center[0] + c * x - s * y, center[1] + s * x + c * y]


def draw_kohler():
    """Redraw of Köhler's maluma (two overlapping loops) and takete (zigzag star)."""
    loops = [_ellipse(*e) for e in KOHLER_MALUMA]
    allpts = np.vstack(loops)
    allpts[:, 1] *= -1
    fitted = np.asarray(_fit(allpts))
    img, d = _canvas()
    i = 0
    for loop in loops:
        _outline(d, [tuple(p) for p in fitted[i:i + len(loop)]])
        i += len(loop)
    round_path = SHAPES / "classic" / "kohler_round.png"
    _save(img, round_path)
    img, d = _canvas()
    pts = [(x, -y) for x, y in KOHLER_TAKETE]
    _outline(d, _fit(pts), closed=False)
    sharp_path = SHAPES / "classic" / "kohler_sharp.png"
    _save(img, sharp_path)
    src = "redrawn after Köhler (1929), via PerForm Fig. 2"
    return [
        {"shape_id": "kohler_round", "set": "classic", "file": str(round_path.relative_to(ROOT)), "role": "round", "pair": "kohler", "source": src},
        {"shape_id": "kohler_sharp", "set": "classic", "file": str(sharp_path.relative_to(ROOT)), "role": "sharp", "pair": "kohler", "source": src},
    ]


def draw_chen_fig1():
    """Crop the round (left) and angular (right) shapes from Chen et al. Fig. 1 onto square canvases."""
    src_img = Image.open(SHAPES / "sources" / "chen2016_fig1.jpg").convert("L")
    w, h = src_img.size
    rows = []
    for role, box in (("round", (0, 0, w // 2, h)), ("sharp", (w // 2, 0, w, h))):
        crop = src_img.crop(box)
        bbox = Image.eval(crop, lambda v: 255 - v).point(lambda v: 255 if v > 40 else 0).getbbox()
        crop = crop.crop(bbox)
        side = int(max(crop.size) / 0.72)
        canvas = Image.new("L", (side, side), 255)
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        path = SHAPES / "classic" / f"chen_fig1_{role}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.resize((SIZE, SIZE), Image.LANCZOS).convert("RGB").save(path)
        rows.append({"shape_id": f"chen_fig1_{role}", "set": "classic", "file": str(path.relative_to(ROOT)), "role": role,
                     "pair": "chen_fig1", "source": "Chen et al. (2016) Fig. 1, CC BY 4.0"})
    return rows


def _smooth_blob(angles, radii, k, n=2048):
    """Blur the star's radius-by-angle profile so lobes keep their places but tips and valleys turn round."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    a = np.asarray(angles)
    r = np.interp(t, np.r_[a - 2 * math.pi, a, a + 2 * math.pi], np.r_[radii, radii, radii])
    sigma = n / (2 * k) * 0.28
    kernel = np.exp(-0.5 * (np.minimum(np.arange(n), n - np.arange(n)) / sigma) ** 2)
    smooth = np.real(np.fft.ifft(np.fft.fft(r) * np.fft.fft(kernel / kernel.sum())))
    smooth = smooth.mean() + (smooth - smooth.mean()) * 0.75 * np.ptp(r) / np.ptp(smooth)
    return np.c_[smooth * np.cos(t), smooth * np.sin(t)]


def draw_maurer_style(n_pairs=4, seed=2006):
    """Irregular jagged stars and wavy blobs drawn through the same random alternating radii."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_pairs):
        k = rng.randint(5, 8)
        angles = [2 * math.pi * j / (2 * k) + rng.uniform(-0.25, 0.25) * math.pi / k for j in range(2 * k)]
        radii = [rng.uniform(0.85, 1.25) if j % 2 == 0 else rng.uniform(0.35, 0.6) for j in range(2 * k)]
        pts = np.array([(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)])
        for role, outline in (("sharp", pts), ("round", _smooth_blob(angles, radii, k))):
            img, d = _canvas()
            _outline(d, _fit(outline))
            path = SHAPES / "classic" / f"maurer_style_{i}_{role}.png"
            _save(img, path)
            rows.append({"shape_id": f"maurer_style_{i}_{role}", "set": "classic", "file": str(path.relative_to(ROOT)),
                         "role": role, "pair": f"maurer_style_{i}", "source": "our own, in the style of Maurer et al. (2006)"})
    return rows


def _star(inner, top_stretch=1.0, n=5):
    """Vertices of a star; the top point can be stretched upward."""
    out = []
    for j in range(2 * n):
        a = math.pi / 2 + j * math.pi / n
        r = 1.0 if j % 2 == 0 else inner
        if j == 0:
            r *= top_stretch
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def _bezier(p0, p1, p2, p3, n=200):
    """Points along a cubic Bézier curve."""
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def thesis_outline(name):
    """Approximate redraws of the six PerForm Fig. 26 outlines."""
    if name == "q1_circle":
        t = np.linspace(0, 2 * math.pi, 600, endpoint=False)
        return np.c_[np.cos(t), np.sin(t)]
    if name == "q5_rounded_star":
        return chen_points(5, 0.22, 0)
    if name == "q6_leaf":
        a, b = (-0.55, -1.0), (0.55, 1.0)
        left = _bezier(a, (-1.25, -0.55), (-0.55, 1.15), b)
        right = _bezier(b, (1.25, 0.55), (0.55, -1.15), a)
        return np.vstack([left, right[1:-1]])
    if name == "q3_s_curve":
        centre = _bezier((-0.7, -1.0), (1.3, -1.0), (-1.3, 1.0), (0.7, 1.0), n=300)
        tangent = np.gradient(centre, axis=0)
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        width = 0.16 * np.sin(np.linspace(0, math.pi, len(centre)))[:, None]
        return np.vstack([centre + width * normal, (centre - width * normal)[::-1][1:-1]])
    if name == "q4_star":
        return np.array(_star(0.45))
    if name == "q2_tall_star":
        return np.array(_star(0.5, top_stretch=1.7))
    raise ValueError(name)


def draw_thesis():
    """The six thesis outlines."""
    rows = []
    for name in THESIS_SHAPES:
        img, d = _canvas()
        _outline(d, _fit(thesis_outline(name), margin=0.2))
        path = SHAPES / "thesis" / f"{name}.png"
        _save(img, path)
        rows.append({"shape_id": f"thesis_{name}", "set": "thesis", "file": str(path.relative_to(ROOT)), "role": name.split("_")[0],
                     "pair": "", "source": "redrawn from PerForm (2019) Fig. 26"})
    return rows


def make_all():
    """Draw the chen, classic and thesis sets and return their rows."""
    return draw_chen() + draw_kohler() + draw_chen_fig1() + draw_maurer_style() + draw_thesis()
