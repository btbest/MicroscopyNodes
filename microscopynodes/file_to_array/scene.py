"""Headless OME-Zarr Scene adapter for the MicroscopyNodes volume pipeline."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping

import clearscale as cs
import numpy as np
from cmap import Color, Colormap

from ..data_model import ChannelDataModel, ChannelModel, ChannelVizModel, DatasetModel, sanitize_channel_name
from .zarr import OmeZarrLoadError, ZarrLoader, _loadable_multiscale


@dataclass(frozen=True)
class SceneSource:
    uri: str
    scene: cs.Scene


def discover_scenes(uri: str | Path) -> tuple[SceneSource, ...]:
    """Resolve each Scene at this group. Its multiscale_paths list the linked images and their levels."""
    uri = str(uri)
    group = ZarrLoader().open_zarr(uri)
    metadata = cs.OmeZarrGroup.from_attrs(group.attrs, shape_source="singletons")
    for invalid in metadata.invalid_objects:
        if invalid.kind == "scene":
            raise OmeZarrLoadError(invalid.error)

    multiscales = {}
    for scene in metadata.scenes:
        for path in scene.unresolved_multiscale_paths:
            if path not in multiscales:
                try:
                    child = group[path]
                    child_metadata = cs.OmeZarrGroup.from_attrs(child.attrs, shape_source=lambda key: child[key].shape)
                    multiscales[path] = _loadable_multiscale(child_metadata)
                except (KeyError, ValueError) as error:
                    raise OmeZarrLoadError(f"Could not resolve Scene image {path!r}: {error}") from error

    return tuple(SceneSource(uri, scene.with_resolved(multiscales)) for scene in metadata.scenes)


def scene_dataset(
    source: SceneSource,
    *,
    world: str,
    cache_dir: Path,
    levels: Mapping[str, str] | None = None,
) -> DatasetModel:
    """Build one Blender dataset with a flat list of independently selectable scene layers."""
    systems = source.scene.to_ome_zarr(version="0.6").get("coordinateSystems", [])
    axes = next((system["axes"] for system in systems if system["name"] == world), ())
    if (
        len(axes) != 3
        or {axis["name"] for axis in axes} != set("xyz")
        or any(axis.get("type") != "space" for axis in axes)
    ):
        raise OmeZarrLoadError(f"Scene system {world!r} must have x, y and z space axes")
    units = {axis.get("unit") for axis in axes}
    if len(units) != 1 or not next(iter(units)):
        raise OmeZarrLoadError(f"Scene system {world!r} must use one defined spatial unit")
    world_axes = tuple(axis["name"] for axis in axes)
    unit = next(iter(units))
    meters_per_unit = cs.Scale({"x": 1}, unit={"x": unit}).with_unit_converted({"x": "meter"}).pixel_size["x"]

    levels = {} if levels is None else levels
    unknown_paths = levels.keys() - source.scene.multiscale_paths.keys()
    if unknown_paths:
        raise OmeZarrLoadError(f"Level selection refers to unknown Scene images: {sorted(unknown_paths)}")

    channels = []
    display_names = set()
    for path, image in source.scene.multiscale_paths.items():
        level_keys = tuple(image)
        level = levels.get(path, level_keys[0])
        if level not in image:
            raise OmeZarrLoadError(f"Scene image {path!r} has no level {level!r}")
        scale = image[level]
        relation = source.scene.relation_from(image, to=world)
        if relation is None:
            raise OmeZarrLoadError(f"Scene image {path!r} has no affine placement in {world!r}")

        source_axes = "".join(image.axes)
        image_name = PurePosixPath(path).name
        omero = image.ome.omero
        channel_count = scale.shape.get("c", 1)
        cache_key = sha256(f"{source.uri}\n{path}".encode()).hexdigest()[:16]

        for local_index in range(channel_count):
            slot = len(channels)
            channel = omero.channels[local_index] if omero and local_index < len(omero.channels) else None
            label = channel.label if channel else None
            fallback = image_name if channel_count == 1 else f"{image_name} {local_index + 1}"
            name = sanitize_channel_name(label or fallback)
            while name in display_names:
                name = f"{name} {slot + 1}"
            display_names.add(name)
            color = Colormap([Color(f"#{channel.color}")]) if channel and channel.color else None
            channels.append(
                ChannelModel(
                    source_name=label,
                    cache_path=str(cache_dir / cache_key),
                    data=ChannelDataModel(
                        dataset_resolution=level_keys.index(level),
                        ix=slot,
                        source_channel_index=local_index,
                        axes_order=source_axes.replace("c", ""),
                        source_axes_order=source_axes,
                        affine=pixel_to_world(relation, scale, world_axes, local_index).tolist(),
                        unit=meters_per_unit,
                        source=source.uri,
                        internal_path=str(PurePosixPath(path) / level),
                    ),
                    viz=ChannelVizModel(ix=slot, name=name, cmap=color),
                )
            )

    return DatasetModel(name=f"{Path(source.uri).name} {world}", channels=channels)


def pixel_to_world(
    relation: cs.SpatialRelation, scale: cs.Scale, world_axes: tuple[str, ...], channel_index: int
) -> np.ndarray:
    """Compose a channel's array-index-to-world transform in Blender's xyz order."""
    axes = tuple(scale.shape)
    spatial = set(axes) & set("xyz")
    if len(spatial) < 2 or set(axes) - set("cxyz"):
        raise OmeZarrLoadError(
            f"Scene prototype supports 2D or 3D spatial data and an optional channel axis, got {axes}"
        )

    # Each row maps a Blender xyz pixel center to one intrinsic coordinate.
    rows = np.zeros((len(axes), 4))
    for i, axis in enumerate(axes):
        rows[i, 3] = scale.translation[axis]
        if axis == "c":
            rows[i, 3] += channel_index * scale.pixel_size[axis]
        else:
            rows[i, "xyz".index(axis)] = scale.pixel_size[axis]

    steps = relation.relations if isinstance(relation, cs.SpatialRelationSequence) else (relation,)
    for step in steps:
        target_axes = step.target_axes(axes)
        if isinstance(step, cs.Affine):
            linear, offset = step.to_matrix_and_offset()
            rows = np.asarray(linear) @ rows
            rows[:, 3] += offset
        elif isinstance(step, (cs.ProjectionTo, cs.PermutationTo, cs.AxisRearrangementTo)):
            rows = np.asarray([rows[axes.index(axis)] if axis in axes else np.zeros(4) for axis in target_axes])
        elif not isinstance(step, (cs.SameSpace, cs.AxisRenaming)):
            raise OmeZarrLoadError(f"Unsupported placement step {step!r}")
        axes = target_axes

    if axes != world_axes:
        raise OmeZarrLoadError(f"Placement reaches {axes}, expected {world_axes}")
    matrix = np.eye(4)
    matrix[:3] = rows[[axes.index(axis) for axis in "xyz"]]
    for missing in set("xyz") - spatial:
        column = "xyz".index(missing)
        a = matrix[:3, (column + 1) % 3]
        b = matrix[:3, (column + 2) % 3]
        normal = np.cross(a, b)
        length = np.linalg.norm(normal)
        if length == 0:
            raise OmeZarrLoadError("Scene plane collapsed to fewer than two dimensions")
        # to_xyz adds one voxel centered at zero on the absent axis.
        matrix[:3, column] = normal * (min(np.linalg.norm(a), np.linalg.norm(b)) / length)

    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix[:3, :3]) < 3:
        raise OmeZarrLoadError("Scene affine has a singular or nonfinite Blender placement")
    return matrix
