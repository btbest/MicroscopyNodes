"""Numerical acceptance tests for the headless Scene prototype."""

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from microscopynodes.file_to_array.scene import discover_scenes
from microscopynodes.file_to_array.zarr import OmeZarrLoadError

from .scene_demo import check_house
from .scene_displacement import expected_warp, make_displacement_scene, output_grid, sample_displacement_scene


def test_house_scene_channel_identity_and_placement(tmp_path):
    fixture = os.environ.get("MIN_HOUSE_SCENE")
    if fixture is None:
        pytest.skip("Set MIN_HOUSE_SCENE to the generated house_scene.zarr fixture")
    (source,) = discover_scenes(Path(fixture).resolve(strict=True))
    check_house(source, tmp_path)


def test_displacement_pull_samples_known_pixels(tmp_path):
    path = tmp_path / "displacement.zarr"
    original = make_displacement_scene(path)
    with pytest.raises(OmeZarrLoadError, match="external data are currently unsupported"):
        discover_scenes(path)
    pixels = sample_displacement_scene(path, output_grid())
    np.testing.assert_array_equal(pixels, expected_warp(original))
    np.testing.assert_array_equal(np.argwhere(pixels == 255), [[2, 4, 3]])
    assert not pixels[:, 3:6, -2:].any()
    assert not pixels[:, 6:, :1].any()


@pytest.mark.parametrize("missing", ["payload", "route"])
def test_displacement_reports_missing_inputs_separately(tmp_path, missing):
    path = tmp_path / "displacement.zarr"
    make_displacement_scene(path)
    root = zarr.open_group(path, mode="r+")
    ome = root.attrs["ome"]
    transforms = ome["scene"]["coordinateTransformations"]
    if missing == "payload":
        transforms[0]["path"] = "coordinateTransformations/missing"
        diagnostic = "Missing displacement payload"
    else:
        transforms.clear()
        diagnostic = "No pull route"
    root.attrs["ome"] = ome
    with pytest.raises(OmeZarrLoadError, match=diagnostic):
        sample_displacement_scene(path, output_grid())
