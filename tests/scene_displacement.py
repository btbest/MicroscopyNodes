"""One path-backed pull transform to probe the resampling interface."""

from pathlib import Path

import clearscale as cs
import numpy as np
import zarr

from microscopynodes.data_model import ChannelDataModel, ChannelModel, ChannelVizModel, DatasetModel
from microscopynodes.file_to_array.scene import pixel_to_world
from microscopynodes.file_to_array.zarr import OmeZarrLoadError

AXES = tuple("zyx")
SHAPE = (6, 9, 12)
PITCH = 2.0
ROW_SHIFTS = np.repeat([0, 2, -1], 3)


def output_grid() -> cs.Scale:
    return cs.Scale(
        dict(zip(AXES, SHAPE)),
        pixel_size=dict.fromkeys(AXES, PITCH),
        translation=dict.fromkeys(AXES, PITCH / 2),
        unit=dict.fromkeys(AXES, "centimeter"),
    )


def write_multiscale(group: zarr.Group, pixels: np.ndarray, scale: cs.Scale) -> str:
    multiscale = cs.Multiscale.from_single(scale)
    attrs = cs.OmeZarrGroup.from_single(multiscale).to_attrs(version="0.6")
    group.attrs.update(attrs)
    group.create_array("s0", data=pixels, chunks=pixels.shape, dimension_names=tuple(scale.shape))
    return attrs["ome"]["multiscales"][0]["coordinateSystems"][0]["name"]


def make_displacement_scene(path: Path) -> np.ndarray:
    root = zarr.open_group(path, mode="w-", zarr_format=3)
    grid = output_grid()
    pixels = np.zeros(SHAPE, dtype=np.uint8)
    pixels[..., 0] = 120
    pixels[..., 5] = 200
    pixels[2, 4, 5] = 255
    image_system = write_multiscale(root.create_group("warped"), pixels, grid)

    field = np.zeros((3, *SHAPE), dtype=np.float32)
    field[2] = ROW_SHIFTS[None, :, None] * PITCH
    field_axes = {"d": cs.ome_zarr.Axis(type="displacement", discrete=True), **dict(grid.ome_zarr_axes)}
    field_grid = cs.Scale(
        {"d": 3, **dict(grid.shape)},
        pixel_size={"d": 1, **dict(grid.pixel_size)},
        translation={"d": 0, **dict(grid.translation)},
        ome_zarr_axes=field_axes,
    )
    field_path = "coordinateTransformations/warp"
    write_multiscale(root.create_group(field_path), field, field_grid)
    root.attrs.update(
        {
            "ome": {
                "version": "0.6",
                "scene": {
                    "coordinateSystems": [
                        {
                            "name": "world",
                            "axes": [axis.to_ome_zarr(version="0.6") for axis in grid.ome_zarr_axes.values()],
                        }
                    ],
                    "coordinateTransformations": [
                        {
                            "type": "displacements",
                            "path": field_path,
                            "interpolation": "nearest",
                            "input": {"name": "world"},
                            "output": {"path": "warped", "name": image_system},
                        }
                    ],
                },
            }
        }
    )
    return pixels


def read_fixture_level(group: zarr.Group) -> tuple[zarr.Array, cs.Scale, str]:
    array = group["s0"]
    (multiscale,) = cs.OmeZarrGroup.from_attrs(group.attrs, shape_source={"s0": array.shape}).multiscales
    name = multiscale.to_ome_zarr(version="0.6")["coordinateSystems"][0]["name"]
    return array, multiscale["s0"], name


def nearest_indices(positions, grid: cs.Scale):
    return tuple(
        np.floor((p - grid.translation[axis]) / grid.pixel_size[axis] + 0.5).astype(np.intp)
        for p, axis in zip(positions, AXES)
    )


def in_bounds(indices, shape):
    z, y, x = ((i >= 0) & (i < size) for i, size in zip(indices, shape))
    return z & y & x


def sample_displacement_scene(path: Path, output: cs.Scale) -> np.ndarray:
    """Load just the direct, nearest-neighbor pull edge used by this fixture."""
    root = zarr.open_group(path, mode="r")
    scene = root.attrs["ome"]["scene"]
    image, source_grid, source_system = read_fixture_level(root["warped"])
    edges = [
        edge
        for edge in scene["coordinateTransformations"]
        if edge["input"] == {"name": "world"} and edge["output"] == {"path": "warped", "name": source_system}
    ]
    if not edges:
        raise OmeZarrLoadError("No pull route from 'world' to image 'warped'")
    if len(edges) != 1 or edges[0]["type"] != "displacements" or edges[0].get("interpolation") != "nearest":
        raise OmeZarrLoadError("Fixture expects one direct nearest-neighbor displacement edge")
    payload_path = edges[0]["path"]
    try:
        field, field_grid, _ = read_fixture_level(root[payload_path])
    except KeyError as error:
        raise OmeZarrLoadError(f"Missing displacement payload at {payload_path!r}") from error

    if (
        tuple(field_grid.shape) != ("d", *AXES)
        or field.shape[0] != len(AXES)
        or field_grid.ome_zarr_axes["d"].type != "displacement"
    ):
        raise OmeZarrLoadError("Fixture expects a dzyx field with z, y and x vector components")
    for grid in (output, source_grid, field_grid.with_axes(AXES)):
        if tuple(grid.shape) != AXES or set(grid.unit.values()) != {"centimeter"}:
            raise OmeZarrLoadError("Fixture expects zyx grids in centimeters")

    centers = np.ix_(
        *(np.arange(output.shape[axis]) * output.pixel_size[axis] + output.translation[axis] for axis in AXES)
    )
    field_indices = nearest_indices(centers, field_grid)
    if not in_bounds(field_indices, field.shape[1:]).all():
        raise OmeZarrLoadError("Displacement field does not cover the output grid")
    vectors = field[:][(slice(None), *field_indices)]
    source_indices = nearest_indices(tuple(p + offset for p, offset in zip(centers, vectors)), source_grid)
    safe_indices = tuple(np.clip(i, 0, size - 1) for i, size in zip(source_indices, image.shape))
    return np.where(in_bounds(source_indices, image.shape), image[:][safe_indices], 0)


def expected_warp(pixels: np.ndarray) -> np.ndarray:
    expected = np.zeros_like(pixels)
    for row, shift in enumerate(ROW_SHIFTS):
        for x in range(pixels.shape[2]):
            if 0 <= x - shift < pixels.shape[2]:
                expected[:, row, x - shift] = pixels[:, row, x]
    return expected


def baked_volume(path: Path, pixels: np.ndarray, grid: cs.Scale, cache_dir: Path) -> DatasetModel:
    root = zarr.open_group(path, mode="w-", zarr_format=3)
    write_multiscale(root, pixels, grid)
    channel = ChannelModel(
        cache_path=str(cache_dir / "displacement"),
        data=ChannelDataModel(
            dataset_resolution=0,
            ix=0,
            axes_order="zyx",
            source=str(path),
            internal_path="s0",
            unit=0.01,
            affine=pixel_to_world(cs.SameSpace(), grid, AXES, 0).tolist(),
        ),
        viz=ChannelVizModel(name="Warped sample"),
    )
    return DatasetModel(name="Displacement fallback", channels=[channel])
