#!/usr/bin/env python3
"""
generate_banner.py — Kaarthi S GitHub Profile Banner Generator
Produces dark.svg and light.svg (1180×610) with:
  • Floyd–Steinberg dithered portrait (300×340, serpentine)
  • Dark mode: background-segmented dots on #0A101F
  • Light mode: dots draw dark parts on #F0F4FF
  • Intro animation (~60 scattered random groups, ~2s fade-in)
  • Loop (~94 drift bands, ~900 traveller dots, optimal-transport logo morph)
  • Info panel with dotted leaders, textLength, pulsing LIVE badge

Usage:
    python generate_banner.py --photo profile.jpg

Requirements:
    pip install pillow numpy scipy
"""

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# --- PALETTE -----------------------------------------------------------------
PALETTE = {
    "bg_dark":        "#0A101F",
    "bg_light":       "#F0F4FF",
    "dot_dark":       "#A78BFA",   # portrait dots dark mode
    "dot_light":      "#4C1D95",   # portrait dots light mode
    "chrome":         "#22D3EE",
    "chrome2":        "#0891B2",
    "accent":         "#10B981",
    "live_red":       "#EF4444",
    "pill_bg":        "#1E293B",
    "text_primary":   "#E2E8F0",
    "text_secondary": "#94A3B8",
    "leader":         "#334155",
    "logo_code":      "#22D3EE",
    "logo_azure":     "#0891B2",
    "logo_python":    "#A78BFA",
}

PORTRAIT_W, PORTRAIT_H = 300, 340
BANNER_W, BANNER_H = 1180, 610
INFO_X = 440
DOT_R  = 1.0

# --- PROFILE DATA ------------------------------------------------------------
PROFILE = {
    "subject":    "Kaarthi S",
    "role":       "AI & Automation Engineer",
    "origin":     "Thiruchengode, Tamil Nadu, IN",
    "education":  "B.Tech IT · Anna Univ · 2027",
    "status":     "Building + Learning + Shipping",
    "toolchain":  "VS Code · Git · ChatGPT · n8n",
    "lang":       "Java · C · Python · JavaScript",
    "frontend":   "HTML · CSS · Web Dev",
    "backend":    "Python · Java",
    "database":   "MySQL · DBMS",
    "infra":      "Azure AI · Cloud · n8n",
    "mail":       "kaarthisenthilkumar@gmail.com",
    "portfolio":  "coming soon",
    "linkedin":   "linkedin.com/in/kaarthi-s-a7a057318",
    "github":     "github.com/kaarthi2005",
    "handle":     "kaarthi2005",
}

# --- IMAGE PROCESSING --------------------------------------------------------

def load_and_prep_photo(photo_path: str) -> Image.Image:
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    crop_h = int(h * 0.82)
    top = 0
    left = max(0, (w - crop_h) // 2)
    right = min(w, left + crop_h)
    img = img.crop((left, top, right, crop_h))
    img = img.resize((PORTRAIT_W, PORTRAIT_H), Image.LANCZOS)
    return img


def enhance_photo(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.3)
    return img


def segment_background(img: Image.Image) -> np.ndarray:
    """
    For headshot photos where the subject is centred, use a soft elliptical
    subject mask rather than colour-distance (which fails when background
    and skin tones are similar in lab/office shots).

    The mask covers the centre ~72% of width and ~90% of height,
    with a smooth Gaussian fade at the boundary so error diffusion
    doesn't get a hard-cut edge.

    Returns bool array (True = foreground/subject).
    """
    from scipy import ndimage
    h, w = PORTRAIT_H, PORTRAIT_W

    # Ellipse radii (fraction of image)
    rx = w * 0.40   # horizontal semi-axis
    ry = h * 0.47   # vertical semi-axis
    cx = w * 0.50   # centred horizontally
    cy = h * 0.48   # slightly upper-centre (head is higher)

    yy, xx = np.mgrid[0:h, 0:w]
    # Normalised distance from ellipse centre
    ellipse_dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # Hard mask at 1.0 radius, with a ~15% soft margin
    mask = ellipse_dist <= 1.0

    # Refine: also exclude pixels in the outer 10% corners that are clearly
    # background (optional colour-assist at very generous threshold)
    arr = np.array(img, dtype=np.float32)
    # Edge pixels for colour reference
    edge_ref = np.concatenate([
        arr[0, :].reshape(-1, 3),
        arr[-1, :].reshape(-1, 3),
        arr[:, 0].reshape(-1, 3),
        arr[:, -1].reshape(-1, 3),
    ])
    bg_mean = edge_ref.mean(0)
    colour_dist = np.linalg.norm(arr - bg_mean, axis=2)
    # Very generous: only exclude pixels extremely similar to background AND outside ellipse
    likely_bg = (colour_dist < 18) & (~mask)
    mask = mask & (~likely_bg)

    # Clean up
    struct = ndimage.generate_binary_structure(2, 2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=4)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool)


def floyd_steinberg_dither(img: Image.Image, mask=None, dark_mode: bool = True) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    out = np.zeros((h, w), dtype=bool)

    for y in range(h):
        ltr = (y % 2 == 0)
        xs = range(w) if ltr else range(w - 1, -1, -1)
        for x in xs:
            if dark_mode and mask is not None and not mask[y, x]:
                gray[y, x] = 1.0
                continue
            old = gray[y, x]
            new = 0.0 if old < 0.5 else 1.0
            out[y, x] = (new == 0.0)
            err = old - new

            def diffuse_masked(dy, dx, factor):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if dark_mode and mask is not None and not mask[ny, nx]:
                        return
                    gray[ny, nx] += err * factor

            if ltr:
                diffuse_masked(0,  1, 7/16)
                diffuse_masked(1, -1, 3/16)
                diffuse_masked(1,  0, 5/16)
                diffuse_masked(1,  1, 1/16)
            else:
                diffuse_masked(0, -1, 7/16)
                diffuse_masked(1,  1, 3/16)
                diffuse_masked(1,  0, 5/16)
                diffuse_masked(1, -1, 1/16)

    return out


# --- EVENNESS METRIC ---------------------------------------------------------

def evenness_metric(groups, n_bins=20):
    h_bins = np.zeros((n_bins, n_bins), dtype=float)
    for g in groups:
        arr = np.array(g)
        if len(arr) == 0:
            continue
        gx = arr % PORTRAIT_W
        gy = arr // PORTRAIT_W
        bx = (gx * n_bins // PORTRAIT_W).clip(0, n_bins - 1)
        by_ = (gy * n_bins // PORTRAIT_H).clip(0, n_bins - 1)
        for i in range(len(arr)):
            h_bins[by_[i], bx[i]] += 1
    h_bins /= (h_bins.sum() + 1e-9)
    expected = 1.0 / (n_bins * n_bins)
    return float(np.std(h_bins) / (expected + 1e-9))


# --- SCATTER GROUPS ----------------------------------------------------------

def make_scatter_groups(dot_indices: np.ndarray, n_groups=60):
    """
    Assign dots to n_groups using spatial interleaving:
    1. Add per-dot noise (sigma~4) to positions before sorting
    2. Sort by a space-filling curve approximation (Morton/Z-order)
    3. Interleave: dot i → group (i % n_groups)
    This ensures every group has members spread across the full portrait.
    """
    arr = np.array(dot_indices)
    xs = (arr % PORTRAIT_W).astype(float) + np.random.normal(0, 4, len(arr))
    ys = (arr // PORTRAIT_W).astype(float) + np.random.normal(0, 4, len(arr))

    # Morton interleave: bit-interleave quantised x,y for space-filling order
    # Quantise to 0..511
    qx = np.clip((xs / PORTRAIT_W * 512).astype(int), 0, 511)
    qy = np.clip((ys / PORTRAIT_H * 512).astype(int), 0, 511)

    def morton_2d(x, y):
        # Interleave bits of x and y (16-bit each)
        z = np.zeros(len(x), dtype=np.int64)
        for i in range(9):
            z |= ((x.astype(np.int64) >> i) & 1) << (2*i)
            z |= ((y.astype(np.int64) >> i) & 1) << (2*i + 1)
        return z

    morton = morton_2d(qx, qy)
    # Sort by Morton code then interleave → each group gets uniformly scattered dots
    sorted_order = np.argsort(morton)
    groups = [[] for _ in range(n_groups)]
    flat = dot_indices.tolist()
    for idx, orig_pos in enumerate(sorted_order):
        groups[idx % n_groups].append(flat[orig_pos])

    metric = evenness_metric(groups)
    print(f"  Evenness metric: {metric:.4f} (target <=0.06)")
    return groups


# --- DRIFT BANDS -------------------------------------------------------------

def make_drift_bands(dot_positions: np.ndarray, n_bands=94, logo_centroid=(150, 170)):
    xs = dot_positions % PORTRAIT_W
    ys = dot_positions // PORTRAIT_W
    noisy_xs = xs + np.random.normal(0, 4, len(xs))
    noisy_ys = ys + np.random.normal(0, 4, len(xs))
    dx = noisy_xs - logo_centroid[0]
    dy_ = noisy_ys - logo_centroid[1]
    angle = np.arctan2(dy_, dx)
    bins = np.digitize(angle, np.linspace(-math.pi, math.pi, n_bands + 1))
    bins = np.clip(bins - 1, 0, n_bands - 1)
    bands = []
    for b in range(n_bands):
        idx = np.where(bins == b)[0]
        if len(idx) == 0:
            continue
        bxs = xs[idx]
        bys = ys[idx]
        mean_x = float(bxs.mean())
        mean_y = float(bys.mean())
        drift_x = (logo_centroid[0] - mean_x) * 0.42
        drift_y = (logo_centroid[1] - mean_y) * 0.42
        bands.append({
            "indices": dot_positions[idx].tolist(),
            "drift_x": round(drift_x, 2),
            "drift_y": round(drift_y, 2),
        })
    return bands


# --- LOGO DOT CLOUDS ---------------------------------------------------------

def logo_code_dots(n=900, cx=150, cy=170):
    dots = []
    scale = 55
    for t in np.linspace(0, 1, n // 4):
        ang = math.pi * 0.5 * (t - 0.5)
        x = cx - 35 + math.cos(ang + math.pi) * scale * 0.4
        y = cy + math.sin(ang + math.pi) * scale * 0.8
        dots.append([x + np.random.normal(0, 1.5), y + np.random.normal(0, 1.5)])
    for t in np.linspace(0, 1, n // 4):
        ang = math.pi * 0.5 * (t - 0.5)
        x = cx + 35 + math.cos(ang) * scale * 0.4
        y = cy + math.sin(ang) * scale * 0.8
        dots.append([x + np.random.normal(0, 1.5), y + np.random.normal(0, 1.5)])
    for t in np.linspace(0, 1, n // 2):
        x = cx - 18 + 36 * t + np.random.normal(0, 1.5)
        y = cy + 35 - 70 * t + np.random.normal(0, 1.5)
        dots.append([x, y])
    return np.array(dots[:n])


def logo_azure_dots(n=900, cx=150, cy=170):
    dots = []
    for t in np.linspace(0, 1, n // 3):
        x = cx - 45 + 25 * t + np.random.normal(0, 1.5)
        y = cy + 35 - 70 * t + np.random.normal(0, 1.5)
        dots.append([x, y])
    for t in np.linspace(0, 1, n // 3):
        x = cx - 20 + 65 * t + np.random.normal(0, 1.5)
        y = cy - 35 + 70 * t + np.random.normal(0, 1.5)
        dots.append([x, y])
    for t in np.linspace(-0.5, 0.5, n // 3):
        ang = t * math.pi
        x = cx + math.cos(ang) * 45 + np.random.normal(0, 1.5)
        y = cy + 35 + math.sin(abs(ang)) * 12 + np.random.normal(0, 1.5)
        dots.append([x, y])
    return np.array(dots[:n])


def logo_python_dots(n=900, cx=150, cy=170):
    dots = []
    for t in np.linspace(0, 1, n // 2):
        x = cx - 25 + np.random.normal(0, 1.5)
        y = cy - 35 + 70 * t + np.random.normal(0, 1.5)
        dots.append([x, y])
    for t in np.linspace(-0.5 * math.pi, 0.5 * math.pi, n // 2):
        x = cx - 25 + math.cos(t) * 30 + np.random.normal(0, 1.5)
        y = cy - 12 + math.sin(t) * 22 + np.random.normal(0, 1.5)
        dots.append([x, y])
    return np.array(dots[:n])


def optimal_transport_match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    _, indices = tree.query(src)
    return indices


# --- SVG HELPERS -------------------------------------------------------------

def dotted_leader(label, value, x, y, total_width=310, font_size=14):
    char_width = font_size * 0.58
    label_len = len(label) * char_width
    value_len = len(value) * char_width
    gap = total_width - label_len - value_len
    n_dots = max(3, int(gap / (char_width * 0.95)))
    dots = " " + ("." * n_dots) + " "

    label_x = x
    dots_x  = x + label_len
    value_x = x + total_width - value_len
    c = PALETTE["chrome"]
    lc = PALETTE["leader"]
    tp = PALETTE["text_primary"]

    return (
        f'<text x="{label_x:.1f}" y="{y:.1f}" font-size="{font_size}" '
        f'fill="{tp}" font-family="\'JetBrains Mono\',\'Fira Code\',monospace" '
        f'textLength="{label_len:.1f}" lengthAdjust="spacingAndGlyphs">{label}</text>'
        f'<text x="{dots_x:.1f}" y="{y:.1f}" font-size="{font_size}" '
        f'fill="{lc}" font-family="\'JetBrains Mono\',monospace">{dots}</text>'
        f'<text x="{value_x:.1f}" y="{y:.1f}" font-size="{font_size}" '
        f'fill="{c}" font-family="\'JetBrains Mono\',monospace" '
        f'textLength="{value_len:.1f}" lengthAdjust="spacingAndGlyphs">{value}</text>'
    )


# --- INFO PANEL --------------------------------------------------------------

def build_info_panel(dark: bool) -> str:
    tc = PALETTE["chrome"] if dark else PALETTE["chrome2"]
    bg2 = PALETTE["pill_bg"] if dark else "#DDE6F5"

    rows = [
        ("Subject",        PROFILE["subject"]),
        ("Role",           PROFILE["role"]),
        ("Origin",         PROFILE["origin"]),
        ("Education",      PROFILE["education"]),
        ("Status",         PROFILE["status"]),
        ("ToolChain",      PROFILE["toolchain"]),
        ("Core.Lang",      PROFILE["lang"]),
        ("Core.Frontend",  PROFILE["frontend"]),
        ("Core.Backend",   PROFILE["backend"]),
        ("Core.Database",  PROFILE["database"]),
        ("Core.Infra",     PROFILE["infra"]),
        ("Grid.Mail",      PROFILE["mail"]),
        ("Grid.LinkedIn",  PROFILE["linkedin"]),
        ("Grid.GitHub",    PROFILE["github"]),
        ("Grid.Portfolio", PROFILE["portfolio"]),
    ]

    px = INFO_X
    py = 25
    pw = BANNER_W - px - 25
    ph = BANNER_H - 50
    row_spacing = 33
    first_row_y = py + 90

    svg = []

    # Panel background
    svg.append(
        f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" '
        f'rx="10" fill="{bg2}" opacity="0.88"/>'
    )
    # Title bar
    svg.append(
        f'<rect x="{px}" y="{py}" width="{pw}" height="38" '
        f'rx="10" fill="{PALETTE["chrome2"]}" opacity="0.95"/>'
    )
    svg.append(
        f'<rect x="{px}" y="{py+20}" width="{pw}" height="18" '
        f'fill="{PALETTE["chrome2"]}" opacity="0.95"/>'
    )
    svg.append(
        f'<text x="{px + pw//2}" y="{py + 25}" text-anchor="middle" '
        f'font-size="13" font-family="\'JetBrains Mono\',monospace" '
        f'fill="white" font-weight="700">profile.sh --live · VISUAL.MAP</text>'
    )
    for i, col in enumerate(["#EF4444", "#F59E0B", "#10B981"]):
        svg.append(f'<circle cx="{px + 16 + i*18}" cy="{py + 19}" r="5" fill="{col}"/>')

    # SYSTEM.INFO header
    svg.append(
        f'<text x="{px + 20}" y="{py + 68}" font-size="13" '
        f'font-family="\'JetBrains Mono\',monospace" fill="{tc}" '
        f'font-weight="700" letter-spacing="2">SYSTEM.INFO</text>'
    )

    # LIVE badge
    svg.append(f'<g transform="translate({px + pw - 82},{py + 55})">')
    svg.append(
        f'<rect x="0" y="0" width="62" height="19" rx="4" fill="{PALETTE["live_red"]}">'
        f'<animate attributeName="opacity" values="1;0.3;1" dur="1.4s" repeatCount="indefinite"/>'
        f'</rect>'
        f'<text x="31" y="13.5" text-anchor="middle" font-size="12" '
        f'font-family="\'JetBrains Mono\',monospace" fill="white" font-weight="700">● LIVE</text>'
    )
    svg.append('</g>')

    # Handle pill
    svg.append(
        f'<rect x="{px + 20}" y="{py + 74}" width="145" height="22" rx="11" '
        f'fill="{tc}" opacity="0.22"/>'
    )
    svg.append(
        f'<text x="{px + 30}" y="{py + 90}" font-size="14" '
        f'font-family="\'JetBrains Mono\',monospace" fill="{tc}" '
        f'font-weight="700">@{PROFILE["handle"]}</text>'
    )

    # Rows
    for i, (label, value) in enumerate(rows):
        y = first_row_y + i * row_spacing
        if y + row_spacing > py + ph:
            break
        svg.append(dotted_leader(label, value, px + 20, y, total_width=pw - 40))

    return "\n".join(svg)


# --- PORTRAIT + ANIMATIONS ---------------------------------------------------

def build_portrait_section(dot_matrix, scatter_groups, drift_bands,
                            travellers_list, dot_colour, px, py):
    h, w = dot_matrix.shape
    svg = []
    loop_dur = 14.2
    t0, t1 = 0, 3.0
    t2 = t1 + 1.3
    t3 = t2 + 2.0
    t4 = t3 + 1.3
    t5 = t4 + 2.0
    t6 = t5 + 1.3
    t7 = t6 + 2.0
    t8 = loop_dur

    def kt(t):
        return f"{min(1.0, t / loop_dur):.4f}"

    # Helper: bool matrix -> path d
    def matrix_to_path(m, ox, oy):
        parts = []
        mh, mw = m.shape
        for y in range(mh):
            row = m[y]
            x = 0
            while x < mw:
                if row[x]:
                    xs = x
                    while x < mw and row[x]:
                        x += 1
                    parts.append(f"M{ox+xs},{oy+y}h{x-xs}")
                else:
                    x += 1
        return " ".join(parts)

    # INTRO LAYER
    intro_dur = 3.2
    delays = sorted([random.uniform(0, 2.0) for _ in range(len(scatter_groups))])
    svg.append('<g id="intro">')
    for gi, (group, delay) in enumerate(zip(scatter_groups, delays)):
        gm = np.zeros((h, w), dtype=bool)
        for fi in group:
            gy2, gx2 = fi // w, fi % w
            if 0 <= gy2 < h and 0 <= gx2 < w:
                gm[gy2, gx2] = True
        pd = matrix_to_path(gm, px, py)
        if not pd:
            continue
        t_in = delay / intro_dur
        t_end = min(1.0, (delay + 0.35) / intro_dur)
        svg.append(
            f'<path d="{pd}" fill="{dot_colour}" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="opacity" values="0;0;1" '
            f'keyTimes="0;{t_in:.4f};{t_end:.4f}" dur="{intro_dur}s" '
            f'begin="0s" fill="freeze"/>'
            f'</path>'
        )
    svg.append('</g>')

    # PORTRAIT LOOP LAYER (duplicate, needed so intro and loop can coexist)
    svg.append('<g id="portrait-loop">')
    for bi, band in enumerate(drift_bands):
        indices = band["indices"]
        dx = band["drift_x"]
        dy_ = band["drift_y"]
        bm = np.zeros((h, w), dtype=bool)
        for fi in indices:
            gy2, gx2 = fi // w, fi % w
            if 0 <= gy2 < h and 0 <= gx2 < w:
                bm[gy2, gx2] = True
        pd = matrix_to_path(bm, px, py)
        if not pd:
            continue

        op_kt = f"{kt(t0)};{kt(t1)};{kt(t2)};{kt(t3)};{kt(t4)};{kt(t5)};{kt(t6)};{kt(t7)};{kt(t8)}"
        op_v  = "1;1;0;0;0;0;0;0;1"
        tx_v  = f"0,0;0,0;{dx:.1f},{dy_:.1f};{dx:.1f},{dy_:.1f};0,0;0,0;0,0;0,0;0,0"

        svg.append(
            f'<path d="{pd}" fill="{dot_colour}" shape-rendering="crispEdges">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{tx_v}" keyTimes="{op_kt}" '
            f'dur="{loop_dur}s" repeatCount="indefinite" additive="sum" calcMode="spline" '
            f'keySplines="0 0 1 1;0.4 0 0.6 1;0 0 1 1;0.4 0 0.6 1;0 0 1 1;0 0 1 1;0.4 0 0.6 1;0 0 1 1"/>'
            f'<animate attributeName="opacity" values="{op_v}" keyTimes="{op_kt}" '
            f'dur="{loop_dur}s" repeatCount="indefinite" calcMode="spline" '
            f'keySplines="0 0 1 1;0.4 0 0.6 1;0 0 1 1;0.4 0 0.6 1;0 0 1 1;0 0 1 1;0.4 0 0.6 1;0 0 1 1"/>'
            f'</path>'
        )
    svg.append('</g>')

    # TRAVELLERS LAYER — batch into SMIL <set> + animateMotion on groups
    # to cut file size.  Each dot gets one <circle> with animateMotion path.
    svg.append('<g id="travellers">')
    for tv in travellers_list:
        src_pts = tv["src"]
        dst_pts = tv["dst"]
        colour  = tv["colour"]
        ts      = tv["t_start"]
        te      = tv["t_end"]
        op_kt   = f"0;{kt(t1)};{kt(ts)};{kt(min(te, loop_dur))};1"

        # Group dots into 30 batches; each batch shares one <g> with opacity anim
        # Individual circles still animate position via transform
        batch_size = max(1, len(src_pts) // 30)
        for bi in range(0, len(src_pts), batch_size):
            batch_src = src_pts[bi:bi+batch_size]
            batch_dst = dst_pts[bi:bi+batch_size]
            if len(batch_src) == 0:
                continue

            svg.append(
                f'<g opacity="0">'
                f'<animate attributeName="opacity" values="0;0;1;1;0" '
                f'keyTimes="{op_kt}" dur="{loop_dur}s" repeatCount="indefinite"/>'
            )
            for sp, dp in zip(batch_src, batch_dst):
                sx = px + float(np.clip(sp[0], 0, PORTRAIT_W - 1))
                sy = py + float(np.clip(sp[1], 0, PORTRAIT_H - 1))
                dx2 = px + float(np.clip(dp[0], 0, PORTRAIT_W - 1))
                dy2 = py + float(np.clip(dp[1], 0, PORTRAIT_H - 1))
                ddx = dx2 - sx
                ddy = dy2 - sy
                mv_kt = f"0;{kt(ts)};{kt(te)};1"
                svg.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="1.3" fill="{colour}">'
                    f'<animateTransform attributeName="transform" type="translate" '
                    f'values="0,0;0,0;{ddx:.1f},{ddy:.1f};0,0" '
                    f'keyTimes="{mv_kt}" dur="{loop_dur}s" repeatCount="indefinite" '
                    f'additive="sum" calcMode="spline" '
                    f'keySplines="0 0 1 1;0.4 0 0.6 1;0.4 0 0.6 1"/>'
                    f'</circle>'
                )
            svg.append('</g>')
    svg.append('</g>')

    return "\n".join(svg)


# --- FULL SVG BUILDER --------------------------------------------------------

def build_svg(dot_dark, dot_light, scatter_groups, drift_bands,
              travellers_list, dark: bool) -> str:
    bg = PALETTE["bg_dark"] if dark else PALETTE["bg_light"]
    dot_colour = PALETTE["dot_dark"] if dark else PALETTE["dot_light"]
    dot_matrix = dot_dark if dark else dot_light

    portrait_x = 28
    portrait_y = (BANNER_H - PORTRAIT_H) // 2

    grid_stroke = "#ffffff08" if dark else "#00000005"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{BANNER_W}" height="{BANNER_H}" viewBox="0 0 {BANNER_W} {BANNER_H}">',
        '<defs>',
        '<style>',
        "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');",
        '</style>',
        f'<pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">',
        f'<path d="M28,0 L0,0 0,28" fill="none" stroke="{grid_stroke}" stroke-width="0.5"/>',
        '</pattern>',
        f'<radialGradient id="pglow" cx="50%" cy="50%" r="60%">',
        f'<stop offset="0%" stop-color="{PALETTE["dot_dark"]}" stop-opacity="0.12"/>',
        f'<stop offset="100%" stop-color="{bg}" stop-opacity="0"/>',
        '</radialGradient>',
        '</defs>',
        f'<rect width="{BANNER_W}" height="{BANNER_H}" fill="{bg}"/>',
        f'<rect width="{BANNER_W}" height="{BANNER_H}" fill="url(#grid)"/>',
        f'<ellipse cx="{portrait_x + PORTRAIT_W//2}" cy="{portrait_y + PORTRAIT_H//2}" '
        f'rx="{int(PORTRAIT_W * 0.75)}" ry="{int(PORTRAIT_H * 0.75)}" fill="url(#pglow)"/>',
        # Portrait frame
        f'<rect x="{portrait_x - 2}" y="{portrait_y - 2}" '
        f'width="{PORTRAIT_W + 4}" height="{PORTRAIT_H + 4}" '
        f'rx="6" fill="none" stroke="{PALETTE["chrome"]}" stroke-width="1.5" '
        f'stroke-dasharray="6,3" opacity="0.55"/>',

        build_portrait_section(dot_matrix, scatter_groups, drift_bands,
                               travellers_list, dot_colour, portrait_x, portrait_y),

        build_info_panel(dark),

        '</svg>',
    ]

    return "\n".join(lines)


# --- MAIN --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate GitHub profile banner SVGs")
    parser.add_argument("--photo",     default="profile.jpg")
    parser.add_argument("--out-dark",  default="dark.svg")
    parser.add_argument("--out-light", default="light.svg")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not Path(args.photo).exists():
        print(f"ERROR: Photo not found: '{args.photo}'")
        sys.exit(1)

    print("[1/7] Loading & preparing photo...")
    img = load_and_prep_photo(args.photo)
    img = enhance_photo(img)

    print("[2/7] Segmenting background...")
    mask = segment_background(img)
    fg_pct = mask.mean() * 100
    print(f"  Foreground coverage: {fg_pct:.1f}%  (expect 40–70%)")

    print("[3/7] Dithering dark mode...")
    dot_dark = floyd_steinberg_dither(img, mask=mask, dark_mode=True)
    print(f"  Dark dots: {dot_dark.sum():,}")

    print("[4/7] Dithering light mode...")
    dot_light = floyd_steinberg_dither(img, mask=None, dark_mode=False)
    print(f"  Light dots: {dot_light.sum():,}")

    dark_idx  = np.where(dot_dark.ravel())[0]
    light_idx = np.where(dot_light.ravel())[0]

    print("[5/7] Building scatter groups...")
    scatter_groups = make_scatter_groups(dark_idx, n_groups=60)

    print("[6/7] Building drift bands...")
    drift_bands = make_drift_bands(dark_idx, n_bands=60,
                                   logo_centroid=(PORTRAIT_W // 2, PORTRAIT_H // 2))

    print("[7/7] Building traveller dots...")
    n_trav = 450
    logo1 = logo_code_dots(n_trav,   cx=PORTRAIT_W//2, cy=PORTRAIT_H//2)
    logo2 = logo_azure_dots(n_trav,  cx=PORTRAIT_W//2, cy=PORTRAIT_H//2)
    logo3 = logo_python_dots(n_trav, cx=PORTRAIT_W//2, cy=PORTRAIT_H//2)

    for lg in [logo1, logo2, logo3]:
        lg[:, 0] = np.clip(lg[:, 0], 0, PORTRAIT_W - 1)
        lg[:, 1] = np.clip(lg[:, 1], 0, PORTRAIT_H - 1)

    match_12 = optimal_transport_match(logo1, logo2)
    match_23 = optimal_transport_match(logo2, logo3)

    travellers_list = [
        {
            "src": logo1[:n_trav],
            "dst": logo2[match_12[:n_trav]],
            "colour": PALETTE["logo_code"],
            "t_start": 3.0 + 1.3,      # after portrait dissolve
            "t_end":   3.0 + 1.3 + 2.0,
        },
        {
            "src": logo2[:n_trav],
            "dst": logo3[match_23[:n_trav]],
            "colour": PALETTE["logo_azure"],
            "t_start": 3.0 + 1.3 + 2.0 + 1.3,
            "t_end":   3.0 + 1.3 + 2.0 + 1.3 + 2.0,
        },
    ]

    print("Assembling dark.svg ...")
    dark_svg = build_svg(dot_dark, dot_light, scatter_groups, drift_bands,
                         travellers_list, dark=True)
    Path(args.out_dark).write_text(dark_svg, encoding="utf-8")
    size_d = len(dark_svg.encode()) / 1024
    print(f"  Saved {args.out_dark}  ({size_d:.0f} KB)")
    if size_d > 1200:
        print(f"  WARNING: File is {size_d:.0f} KB — above expected ~900–1000 KB")

    print("Assembling light.svg ...")
    light_svg = build_svg(dot_dark, dot_light, scatter_groups, drift_bands,
                          travellers_list, dark=False)
    Path(args.out_light).write_text(light_svg, encoding="utf-8")
    size_l = len(light_svg.encode()) / 1024
    print(f"  Saved {args.out_light}  ({size_l:.0f} KB)")

    print()
    print("Done! Open dark.svg and light.svg in a browser to check the animation.")
    print("CDN cache trick if GitHub doesn't update: append ?v=1 to the raw URL.")


if __name__ == "__main__":
    main()
