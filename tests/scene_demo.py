"""Run the OME-Zarr Scene prototype with the real MicroscopyNodes volume pipeline."""

import argparse
from pathlib import Path

import bpy
import numpy as np
from cmap import Colormap
from mathutils import Vector

import microscopynodes
from microscopynodes.blender_state import Dataset
from microscopynodes.file_to_array.scene import discover_scenes, scene_dataset
from microscopynodes.handle_blender_structs.node_handling import (
    get_modifier_input_socket,
    get_socket,
    set_modifier_input_socket,
)

from .scene_displacement import (
    baked_volume,
    expected_warp,
    make_displacement_scene,
    output_grid,
    sample_displacement_scene,
)

HOUSE_CHANNELS = ("house", "Wood", "Hardware", "Glass", "Frame", "Window marker", "solar_panel", "lawn")
OMERO_COLORS = {"Wood": "C98248", "Hardware": "FFD54A", "Glass": "56CDEF", "Frame": "B58CFF", "Window marker": "F04482"}
SCALAR_COLORS = {"house": "#ACADB7", "solar_panel": "#468AFE", "lawn": "#60BF72"}
ALPHA_MULTIPLIERS = {
    "house": 0.085,
    "Wood": 1,
    "Hardware": 1.4,
    "Glass": 1.9,
    "Frame": 1.4,
    "Window marker": 1.75,
    "solar_panel": 18,
    "lawn": 1.2,
}
MARKERS_XYZ = {
    "house": (350, 12.5, 350),
    "Hardware": (133, 5, 190),
    "Window marker": (234, 4, 205),
    "solar_panel": (140, 20 + 100 / np.sqrt(2), 425 + 100 / np.sqrt(2)),
    "lawn": (-50, 430, 0),
}


def marker_world(channel):
    samples = np.argwhere(channel.data.data.compute() == 255)
    assert len(samples), f"Missing bright marker in {channel.name}"
    indices = dict(zip(channel.data.axes_order, samples.mean(axis=0)))
    point = np.array([*(indices.get(axis, 0) for axis in "xyz"), 1])
    return (channel.data.affine_matrix @ point)[:3]


def check_house(source, cache_dir):
    assert tuple(source.scene.multiscale_paths) == ("house", "door", "window", "solar_panel", "lawn")
    assert all(tuple(image) == ("s0", "s1") for image in source.scene.multiscales)
    model = scene_dataset(source, world="world", cache_dir=cache_dir)
    assert tuple(channel.name for channel in model.channels) == HOUSE_CHANNELS
    assert [channel.data.source_channel_index for channel in model.channels] == [0, 0, 1, 0, 1, 2, 0, 0]
    assert [channel.data.ix for channel in model.channels] == list(range(8))
    assert all(channel.data.unit == 0.01 for channel in model.channels)

    channels = {channel.name: channel for channel in model.channels}
    for name, hex_color in OMERO_COLORS.items():
        channel = channels[name]
        assert channel.source_name == name
        assert channel.viz.cmap(1.0).hex.lstrip("#").upper()[:6] == hex_color
        assert channel.data.data.max().compute() > 0

    for name, expected in MARKERS_XYZ.items():
        channel = channels[name]
        tolerance = max(np.linalg.norm(channel.data.affine_matrix[:3, :3], axis=0)) / 2
        np.testing.assert_allclose(marker_world(channel), expected, rtol=0, atol=tolerance, err_msg=name)

    window = channels["Frame"].data
    edge = window.data.compute()[:, :, 0]
    starts = [np.flatnonzero(edge[z]).mean() for z in (0, len(edge) - 1)]
    assert (starts[1] - starts[0]) * 0.8 > 18
    for z, center_y in zip((0, len(edge) - 1), starts):
        np.testing.assert_allclose((window.affine_matrix @ [0, center_y, z, 1])[1], 4, rtol=0, atol=0.4)

    panel = channels["solar_panel"].data
    height, width = panel.data_shape
    corners = np.array([panel.affine_matrix @ [x, y, 0, 1] for x in (-0.5, width - 0.5) for y in (-0.5, height - 0.5)])
    np.testing.assert_allclose(corners[:, 2] - corners[:, 1], 405)
    assert np.all((corners[:, 1] > 0) & (corners[:, 1] < np.minimum(corners[:, 0], 400 - corners[:, 0])))

    coarse = scene_dataset(source, world="world", cache_dir=cache_dir, levels={"door": "s1", "lawn": "s1"})
    coarse_channels = {channel.name: channel for channel in coarse.channels}
    for name, shift in (("Hardware", (0.625,) * 3), ("lawn", (0, 0, 0))):
        fine, sampled = channels[name], coarse_channels[name]
        assert (
            sampled.identifier == fine.identifier
            and sampled.data.source_channel_index == fine.data.source_channel_index
        )
        assert sampled.data.internal_path.endswith("/s1")
        np.testing.assert_allclose(sampled.data.affine_matrix[:3, 3] - fine.data.affine_matrix[:3, 3], shift)
        tolerance = max(np.linalg.norm(sampled.data.affine_matrix[:3, :3], axis=0)) / 2
        np.testing.assert_allclose(marker_world(sampled), MARKERS_XYZ[name], rtol=0, atol=tolerance)
    return model, coarse


def import_volume(model, dataset=None):
    dataset = Dataset() if dataset is None else dataset
    scene = bpy.context.scene
    scene.MiN_import_scale = "METER_SCALE"
    scene.MiN_import_loc = "ZERO"
    dataset.set_state(model)
    dataset.axes.object.hide_render = True
    dataset.slicecube.object.hide_render = True
    for channel in model.channels:
        assert dataset.volume.ch_present(channel)
        node = dataset.volume.node_group.nodes[f"channel_affine_{channel.identifier}"]
        actual = np.array([socket.default_value for socket in node.inputs]).reshape((4, 4), order="F")
        np.testing.assert_allclose(actual, channel.data.affine_matrix, rtol=1e-6, atol=1e-6)
    bpy.context.view_layer.update()
    return dataset


def set_demo_display(dataset, model):
    """Use the full uint8 range and balance emission for the thin fixture layers."""
    for channel in model.channels:
        if channel.name in SCALAR_COLORS:
            channel.viz.cmap = Colormap([SCALAR_COLORS[channel.name]])
    dataset.volume.set_settings(model)
    shader_nodes = dataset.volume.object.data.materials[0].node_tree.nodes
    for channel in model.channels:
        identifier = channel.identifier
        node = dataset.volume.node_group.nodes[f"IMPORT_{identifier}"]
        node.inputs["VDB Minimum"].default_value = 0
        node.inputs["VDB Maximum"].default_value = 1
        for ramp in ("contrast_limits", "alpha_limits"):
            elements = shader_nodes[f"[{ramp}_{identifier}]"].color_ramp.elements
            elements[0].position = 0
            elements[1].position = 1
        shader_nodes[f"[volume_alpha_{identifier}]"].inputs["Alpha Multiplier"].default_value = ALPHA_MULTIPLIERS.get(
            channel.name, 0.7
        )
        shader_nodes[f"[microscopy_shading_{identifier}]"].inputs[
            "Force Alpha/Color to scale linearly"
        ].default_value = True


def check_switches(dataset, model):
    volume = dataset.volume
    by_name = {channel.name: channel for channel in model.channels}
    wood = get_socket(volume.node_group, by_name["Wood"], min_type="SWITCH")
    glass = get_socket(volume.node_group, by_name["Glass"], min_type="SWITCH")
    assert wood.identifier != glass.identifier
    for hidden, visible in ((wood, glass), (glass, wood)):
        set_modifier_input_socket(volume.gn_mod, hidden, False)
        assert not get_modifier_input_socket(volume.gn_mod, hidden)
        assert get_modifier_input_socket(volume.gn_mod, visible)
        set_modifier_input_socket(volume.gn_mod, hidden, True)


def render(dataset, model, angle, path):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.volume_step_rate = 0.3
    scene.cycles.volume_max_steps = 2048
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)

    if scene.world is None:
        scene.world = bpy.data.worlds.new("Scene demo background")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.018, 0.023, 0.03, 1)

    camera = scene.camera
    if camera is None:
        camera = bpy.data.objects.new("Scene demo camera", bpy.data.cameras.new("Scene demo camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
    mins, maxs, _ = model.intermediate_bbox
    transform = dataset.holder.object.matrix_world
    center = transform @ Vector((mins + maxs) / 2)
    extent = max(transform.to_scale()) * max(maxs - mins)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = float(extent * 1.38)
    camera.data.clip_start = min(0.01, extent / 100)
    camera.data.clip_end = max(100, extent * 10)
    camera.location = center + Vector(angle).normalized() * extent * 2
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.render.render(write_still=True)


def run_demo(house: Path, output: Path):
    output.mkdir(parents=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    microscopynodes._test_register()
    (source,) = discover_scenes(house)
    model, coarse = check_house(source, output / "vdb")
    dataset = import_volume(coarse)
    dataset = import_volume(model, dataset)
    set_demo_display(dataset, model)
    check_switches(dataset, model)
    render(dataset, model, (0, -1, 0), output / "house-front.png")
    render(dataset, model, (0.65, -1, 0.72), output / "house-three-quarter.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "house.blend"))
    dataset.volume.object.hide_render = True

    warp = output / "displacement.zarr"
    original = make_displacement_scene(warp)
    grid = output_grid()
    pixels = sample_displacement_scene(warp, grid)
    np.testing.assert_array_equal(pixels, expected_warp(original))
    np.testing.assert_array_equal(np.argwhere(pixels == 255), [[2, 4, 3]])
    model = baked_volume(output / "resampled.zarr", pixels, grid, output / "vdb")
    dataset = import_volume(model)
    set_demo_display(dataset, model)
    np.testing.assert_array_equal(model.channels[0].data.data.compute(), pixels)
    render(dataset, model, (0.4, -1, 1.3), output / "displacement.png")
    print(f"Verified and rendered both Scene prototypes in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("house", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_demo(args.house.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
