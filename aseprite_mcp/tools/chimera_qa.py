"""Chimera pixel-art pipeline QA tools.

Project-specific (NOT general aseprite-mcp): value-readability and
colour-vision-deficiency (CVD) gates that map to The Ninth Bride's
"corruption stage" mechanic + its ~8% colourblind audience, plus a
kCentroid content-aware downscale wrapper that unifies Stage-2 of the
art pipeline behind the MCP surface.

The three audit/gate tools (value_monotonic_check, value_contrast_check,
cvd_palette_audit) are pure colour math over a palette/ramp — no Aseprite,
no numpy/PIL. Only kcentroid_downscale needs numpy+pillow (the optional
`chimera` dependency group), imported lazily so this module always loads.

Colour science:
  - luminance/contrast = WCAG 2.x relative luminance (sRGB).
  - ΔE = CIE76 in CIELAB (D65).
  - CVD simulation = libDaltonLens (Vienot 1999 for protan/deutan, Brettel
    1997 for tritan) in LINEAR RGB. Cross-validated byte-for-byte (2026-06-18)
    against the AseCvdSim Aseprite extension, so this audit and the dev's
    in-editor visual CVD check use the SAME model (the dev has normal colour
    vision and relies on AseCvdSim to see CVD output).
"""
import json
import os

from ..core.commands import reject_traversal
from .. import mcp


# --- colour helpers (pure Python, no deps) -----------------------------------

def _parse_hex(s: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' / 'RRGGBB' / '#RGB' / 'RGB' into an (r, g, b) tuple."""
    h = s.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex colour: {s!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _rel_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG relative luminance (0..1)."""
    r, g, b = (_srgb_to_linear(v / 255.0) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """WCAG contrast ratio (1..21) between two colours."""
    l1, l2 = _rel_luminance(c1), _rel_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _srgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v / 255.0) for v in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _delta_e76(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# CVD simulation = libDaltonLens (public domain, daltonlens.org): Vienot 1999
# for protan/deutan, Brettel 1997 for tritan, applied in linear RGB. This matches
# the AseCvdSim Aseprite extension (aseDaltonLens.lua) EXACTLY — verified
# byte-for-byte across 8 colours x 3 deficiencies (2026-06-18), so the audit and
# the in-editor visual CVD check stay consistent.
_CVD_TYPES = ("protanopia", "deuteranopia", "tritanopia")

_VIENOT = {
    "protanopia": (0.11238, 0.88762, 0.0, 0.11238, 0.88762, 0.0, 0.00401, -0.00401, 1.0),
    "deuteranopia": (0.29275, 0.70725, 0.0, 0.29275, 0.70725, 0.0, -0.02234, 0.02234, 1.0),
}
_BRETTEL_TRITAN = {
    "m1": (1.01277, 0.13548, -0.14826, -0.01243, 0.86812, 0.14431, 0.07589, 0.80500, 0.11911),
    "m2": (0.93678, 0.18979, -0.12657, 0.06154, 0.81526, 0.12320, -0.37562, 1.12767, 0.24796),
    "n": (0.03901, -0.02788, -0.01113),
}


def _mul3(m: tuple, r: float, g: float, b: float) -> tuple[float, float, float]:
    return (m[0] * r + m[1] * g + m[2] * b,
            m[3] * r + m[4] * g + m[5] * b,
            m[6] * r + m[7] * g + m[8] * b)


def _simulate_cvd(rgb: tuple[int, int, int], cvd_type: str) -> tuple[int, int, int]:
    r, g, b = (_srgb_to_linear(v / 255.0) for v in rgb)
    if cvd_type == "tritanopia":
        n = _BRETTEL_TRITAN["n"]
        m = _BRETTEL_TRITAN["m1"] if (r * n[0] + g * n[1] + b * n[2]) >= 0 else _BRETTEL_TRITAN["m2"]
        cr, cg, cb = _mul3(m, r, g, b)
    else:
        cr, cg, cb = _mul3(_VIENOT[cvd_type], r, g, b)
    return tuple(max(0, min(255, round(_linear_to_srgb(max(0.0, min(1.0, c))) * 255)))
                 for c in (cr, cg, cb))


def _parse_palette(colors: list[str]) -> list[tuple[int, int, int]]:
    if not colors:
        raise ValueError("empty colour list")
    return [_parse_hex(c) for c in colors]


# --- tools -------------------------------------------------------------------

@mcp.tool()
async def value_monotonic_check(colors: list[str]) -> str:
    """Verify a value ramp's luminance is strictly monotonic dark→light.

    The Tier-3 EXIT criterion: a shading ramp must change in VALUE at every
    step (not just hue) so the silhouette reads in grayscale / for CVD
    players. Reports per-step WCAG luminance and any step that flattens or
    reverses the trend.

    Args:
        colors: ordered ramp as hex strings (e.g. ["#1a1a2e", "#16213e", ...])

    Returns:
        JSON {monotonic, direction, luminances, violations, n}
    """
    try:
        pal = _parse_palette(colors)
    except ValueError as e:
        return f"Invalid input: {e}"
    lums = [_rel_luminance(c) for c in pal]
    asc = all(lums[i] > lums[i - 1] for i in range(1, len(lums)))
    desc = all(lums[i] < lums[i - 1] for i in range(1, len(lums)))
    monotonic = asc or desc
    direction = "ascending" if asc else "descending" if desc else "none"
    # When not monotonic, report the steps that break the intended trend
    # (inferred from the endpoints).
    intended = "ascending" if (len(lums) < 2 or lums[-1] >= lums[0]) else "descending"
    violations = []
    if not monotonic:
        for i in range(1, len(lums)):
            bad = lums[i] <= lums[i - 1] if intended == "ascending" else lums[i] >= lums[i - 1]
            if bad:
                violations.append({"index": i, "from": _hex(pal[i - 1]),
                                   "to": _hex(pal[i]),
                                   "lum_from": round(lums[i - 1], 4),
                                   "lum_to": round(lums[i], 4)})
    return json.dumps({
        "monotonic": monotonic,
        "direction": direction,
        "luminances": [round(l, 4) for l in lums],
        "violations": violations,
        "n": len(pal),
    })


@mcp.tool()
async def value_contrast_check(
    stage_a: list[str],
    stage_b: list[str],
    min_ratio: float = 3.0,
) -> str:
    """Gate: corresponding value-bands of two corruption stages must differ
    by ≥ min_ratio in WCAG contrast.

    The Ninth Bride's corruption mechanic requires an unmistakable VALUE jump
    between a region's clean and corrupted states (default ≥3:1) so the change
    reads instantly, including in grayscale/for CVD players. Pass the two
    stages' index-aligned ramps (or two single-colour lists).

    Args:
        stage_a: stage-A ramp as hex strings
        stage_b: stage-B ramp as hex strings (same length, index-aligned)
        min_ratio: required WCAG contrast ratio per band (default 3.0)

    Returns:
        JSON {pass, min_ratio_required, min_ratio_found, worst, bands}
    """
    try:
        a, b = _parse_palette(stage_a), _parse_palette(stage_b)
    except ValueError as e:
        return f"Invalid input: {e}"
    if len(a) != len(b):
        return (f"Invalid input: stage_a ({len(a)}) and stage_b ({len(b)}) "
                "must be the same length (index-aligned bands)")
    bands = []
    for i, (ca, cb) in enumerate(zip(a, b)):
        ratio = _contrast_ratio(ca, cb)
        bands.append({"index": i, "a": _hex(ca), "b": _hex(cb),
                      "ratio": round(ratio, 3), "pass": ratio >= min_ratio})
    worst = min(bands, key=lambda x: x["ratio"])
    return json.dumps({
        "pass": all(x["pass"] for x in bands),
        "min_ratio_required": min_ratio,
        "min_ratio_found": worst["ratio"],
        "worst": worst,
        "bands": bands,
    })


@mcp.tool()
async def cvd_palette_audit(colors: list[str], delta_threshold: float = 22.0) -> str:
    """Audit a palette for colour-vision-deficiency collisions.

    Simulates protanopia / deuteranopia (red-green) and tritanopia
    (blue-yellow) and flags any pair of colours that look DISTINCT to normal
    vision but COLLAPSE (ΔE < delta_threshold) for a CVD viewer — the
    dangerous case the dev cannot self-detect. Also lists pairs that are
    low-contrast for everyone.

    Args:
        colors: palette as hex strings
        delta_threshold: CIE76 ΔE below which two colours are "confusable" for
            a CVD viewer (default 22.0 — a gameplay/at-a-glance cut, within the
            ΔE 11-49 "more similar than opposite" band; lower it to ~12 for a
            strict side-by-side cut). Reported ΔEs let you recalibrate.

    Returns:
        JSON {cvd_safe, n_colors, delta_threshold, collisions,
              low_normal_contrast, note}
    """
    try:
        pal = _parse_palette(colors)
    except ValueError as e:
        return f"Invalid input: {e}"
    labs = [_srgb_to_lab(c) for c in pal]
    collisions = []
    low_normal = []
    for i in range(len(pal)):
        for j in range(i + 1, len(pal)):
            normal_de = _delta_e76(labs[i], labs[j])
            if normal_de < delta_threshold:
                low_normal.append({"i": i, "j": j, "color_i": _hex(pal[i]),
                                   "color_j": _hex(pal[j]),
                                   "normal_deltaE": round(normal_de, 2)})
                continue
            for t in _CVD_TYPES:
                de = _delta_e76(_srgb_to_lab(_simulate_cvd(pal[i], t)),
                                _srgb_to_lab(_simulate_cvd(pal[j], t)))
                if de < delta_threshold:
                    collisions.append({"cvd_type": t, "i": i, "j": j,
                                       "color_i": _hex(pal[i]),
                                       "color_j": _hex(pal[j]),
                                       "normal_deltaE": round(normal_de, 2),
                                       "cvd_deltaE": round(de, 2)})
    return json.dumps({
        "cvd_safe": not collisions,
        "n_colors": len(pal),
        "delta_threshold": delta_threshold,
        "collisions": collisions,
        "low_normal_contrast": low_normal,
        "note": "libDaltonLens (Vienot/Brettel); matches AseCvdSim exactly",
    })


@mcp.tool()
async def kcentroid_downscale(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    centroids: int = 2,
    quantize_colors: int = 0,
) -> str:
    """Content-aware downscale (kCentroid) — the smart Stage-2 primitive.

    Per output pixel, takes the dominant (k-means) colour of its source tile,
    preserving the hard silhouette that the value/CVD read depends on (beats
    naive nearest/lanczos, which blur or point-sample). Optionally follows
    with a PIL MAXCOVERAGE palette quantize to lock the colour count.

    Requires the optional `chimera` dependency group (numpy + pillow).

    Args:
        input_path: source image (PNG/any PIL-readable)
        output_path: destination PNG
        target_width: output width in pixels
        target_height: output height in pixels
        centroids: k-means colours per source tile (default 2)
        quantize_colors: if > 0, MAXCOVERAGE-quantize the result to N colours

    Returns:
        success string with the result's unique-colour count
    """
    if not os.path.exists(input_path):
        return f"File {input_path} not found"
    if target_width <= 0 or target_height <= 0:
        return "Invalid input: target_width and target_height must be > 0"
    if centroids < 1:
        return "Invalid input: centroids must be >= 1"
    err = reject_traversal(output_path)
    if err:
        return err
    out = output_path if output_path.lower().endswith(".png") else f"{output_path}.png"
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return ("Failed: numpy+pillow not installed. Install the optional "
                "group: `uv sync --extra chimera`")
    from itertools import product
    try:
        src = Image.open(input_path).convert("RGB")
        down = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        wf, hf = src.width / target_width, src.height / target_height
        for x, y in product(range(target_width), range(target_height)):
            tile = src.crop((x * wf, y * hf, x * wf + wf, y * hf + hf))
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")
            down[y, x, :] = max(tile.getcolors(), key=lambda c: c[0])[1]
        result = Image.fromarray(down)
        if quantize_colors > 0:
            result = result.quantize(colors=quantize_colors, method=2, dither=0).convert("RGB")
        result.save(out)
        n_colors = len(result.getcolors(maxcolors=1 << 24) or [])
    except Exception as e:  # noqa: BLE001 — surface any PIL/numpy failure as a tool error
        return f"Failed to downscale: {e}"
    return f"Downscaled to {target_width}x{target_height} ({n_colors} colours) -> {out}"
