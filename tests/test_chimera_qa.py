"""Chimera pixel-art QA tools (chimera_qa.py).

The value/CVD tools are pure colour math (no Aseprite); kcentroid uses
the optional numpy+pillow group.
"""
import json
import os

from conftest import BASE, ok, run

from aseprite_mcp.tools import chimera_qa as q


def test_value_monotonic_pass():
    r = json.loads(run(q.value_monotonic_check(
        ["#0a0a14", "#1a1a2e", "#4a4a6a", "#9a9ab0", "#e8e8f0"])))
    assert r["monotonic"] is True
    assert r["direction"] == "ascending"
    assert r["violations"] == []


def test_value_monotonic_detects_break():
    r = json.loads(run(q.value_monotonic_check(
        ["#0a0a14", "#4a4a6a", "#1a1a2e", "#e8e8f0"])))
    assert r["monotonic"] is False
    assert any(v["index"] == 2 for v in r["violations"])


def test_value_monotonic_rejects_bad_hex():
    assert run(q.value_monotonic_check(["#xyz123"])).startswith("Invalid input")


def test_value_contrast_pass():
    r = json.loads(run(q.value_contrast_check(
        ["#e8e8f0", "#9a9ab0"], ["#1a1a2e", "#0a0a14"], 3.0)))
    assert r["pass"] is True
    assert r["min_ratio_found"] >= 3.0


def test_value_contrast_fail():
    r = json.loads(run(q.value_contrast_check(["#888888"], ["#7f7f7f"], 3.0)))
    assert r["pass"] is False
    assert r["worst"]["ratio"] < 3.0


def test_value_contrast_length_mismatch():
    res = run(q.value_contrast_check(["#000000"], ["#ffffff", "#000000"], 3.0))
    assert res.startswith("Invalid input")


def test_cvd_grayscale_is_safe():
    r = json.loads(run(q.cvd_palette_audit(
        ["#000000", "#555555", "#aaaaaa", "#ffffff"])))
    assert r["cvd_safe"] is True
    assert r["collisions"] == []


def test_cvd_flags_red_green_collision():
    r = json.loads(run(q.cvd_palette_audit(["#d13030", "#308a30"])))
    assert r["cvd_safe"] is False
    assert any(c["cvd_type"] in ("deuteranopia", "protanopia")
               for c in r["collisions"])


def test_cvd_invalid_hex():
    assert run(q.cvd_palette_audit(["#zzzzzz"])).startswith("Invalid input")


def test_kcentroid_downscale(base_dir):
    from PIL import Image
    src, out = f"{BASE}/kc_src.png", f"{BASE}/kc_out.png"
    img = Image.new("RGB", (64, 64), (20, 130, 200))
    for y in range(32):
        for x in range(32):
            img.putpixel((x, y), (220, 60, 40))  # red quadrant => structure
    img.save(src)
    res = ok(run(q.kcentroid_downscale(src, out, 16, 16, 2)))
    assert os.path.exists(out)
    assert "16x16" in res
    assert Image.open(out).size == (16, 16)


def test_kcentroid_rejects_missing_input():
    assert run(q.kcentroid_downscale(
        "/nonexistent/x.png", f"{BASE}/o.png", 8, 8)).startswith("File")


def test_value_blockin_predictable(base_dir):
    from PIL import Image
    # a smooth vertical gradient — the exact case kCentroid k-means would mud up
    src, out = f"{BASE}/vbi_src.png", f"{BASE}/vbi_out.png"
    img = Image.new("L", (64, 64))
    for y in range(64):
        for x in range(64):
            img.putpixel((x, y), int(y / 63 * 255))
    img.convert("RGB").save(src)
    res = ok(run(q.value_blockin_downscale(src, out, 16, 16, levels=5, supersample=2)))
    assert "16x16" in res
    assert "invented=0" in res            # the predictability guarantee, self-reported
    im = Image.open(out)
    assert im.size == (16, 16)
    assert len(im.getcolors()) <= 5        # output is a strict subset of the 5 chosen levels


def test_value_blockin_auto_supersample(base_dir):
    from PIL import Image
    # supersample=0 -> auto K from source/target ratio; here 64/16 = 4
    src, out = f"{BASE}/vbi_auto_src.png", f"{BASE}/vbi_auto_out.png"
    img = Image.new("L", (64, 64))
    for y in range(64):
        for x in range(64):
            img.putpixel((x, y), int(y / 63 * 255))
    img.convert("RGB").save(src)
    res = ok(run(q.value_blockin_downscale(src, out, 16, 16, levels=4)))  # supersample defaults to 0
    assert "K=4" in res and "invented=0" in res
    assert len(Image.open(out).getcolors()) <= 4


def test_value_blockin_rejects_bad_levels(base_dir):
    from PIL import Image
    src = f"{BASE}/vbi_lv.png"
    Image.new("RGB", (8, 8), (128, 128, 128)).save(src)
    assert run(q.value_blockin_downscale(
        src, f"{BASE}/o.png", 8, 8, levels=1)).startswith("Invalid")


def test_value_blockin_rejects_missing_input():
    assert run(q.value_blockin_downscale(
        "/nonexistent/x.png", f"{BASE}/o.png", 8, 8)).startswith("File")
