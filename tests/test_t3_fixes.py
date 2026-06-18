"""Tier 3 fixes: B2 (pipe-safe list_slices) + B6 (group recursion in get_sprite_info)."""
import json

from conftest import ok, run

from aseprite_mcp.tools import animation, canvas, slices


def test_list_slices_handles_pipe_and_comma_in_name(sprite):
    # '|' and ',' in a slice name used to break the delimiter parse for the
    # whole sprite; JSON-per-slice makes it safe.
    ok(run(slices.create_slice(sprite, "weird|name,x", 1, 2, 10, 12)))
    ok(run(slices.create_slice(sprite, "ninepatch", 0, 0, 16, 16)))
    ok(run(slices.set_slice_center(sprite, "ninepatch", 4, 4, 8, 8)))
    by_name = {s["name"]: s for s in json.loads(ok(run(slices.list_slices(sprite))))}
    assert "weird|name,x" in by_name
    assert by_name["weird|name,x"]["x"] == 1 and by_name["weird|name,x"]["width"] == 10
    assert by_name["ninepatch"]["center"] == {"x": 4, "y": 4, "width": 8, "height": 8}


def test_get_sprite_info_enumerates_group_children(sprite):
    ok(run(canvas.add_group(sprite, "grp")))
    ok(run(canvas.add_layer(sprite, "child", "grp")))
    layers = json.loads(run(animation.get_sprite_info(sprite)))["layers"]
    names = [l["name"] for l in layers]
    assert "grp" in names
    assert "child" in names  # nested layer now enumerated (B6)
    child = next(l for l in layers if l["name"] == "child")
    assert child["parent"] == "grp"
    grp = next(l for l in layers if l["name"] == "grp")
    assert grp["is_group"] is True and grp["parent"] is None
