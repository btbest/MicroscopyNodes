# Headless OME-Zarr Scene prototype

Run from this MicroscopyNodes checkout with the `micnodes` environment and the clearscale revision that wrote the house fixture.

```powershell
python -m tests.scene_demo C:/path/to/house_scene.zarr C:/path/to/new-demo-output
```

The demo uses the existing MicroscopyNodes volume loader. It checks the placements and channel switches, then writes two house views and an OpenVDB-backed Blender file. The demo uses the fixture's full 8-bit display range so automatic contrast does not suppress the walls or glass. The Blender file references the generated volumes at their output paths.

The demo also creates a tiny path-backed displacement Scene, checks its resampled pixels against an independent expected array, and renders that result.

To run the numerical checks separately, set `MIN_HOUSE_SCENE` to the fixture path and run `python -m pytest tests/test_scene_import.py`.

## What the consumer needed

| Operation | Prototype call site | Clearscale API implication |
|---|---|---|
| Discover scenes and levels | `discover_scenes(uri)` reads `OmeZarrGroup.scenes`, looks up each linked group and resolves the paths. `Scene.multiscale_paths` then exposes the images and their levels. | The existing objects cover the model. A callback-based resolver and public access to named Scene axes and units would remove the most adapter code. |
| Collect channels | `scene_dataset(source, world=..., levels=..., cache_dir=...)` combines child image channels in Scene order and builds each `ChannelDataModel` separately. | A useful iterator would yield each image path, local channel index and OMERO metadata. Blender owns slot IDs, display-name disambiguation, and cache paths. |
| Place affine channels | `pixel_to_world(relation, scale, world_axes, channel_index)` composes the pixel-center grid and resolved relation. It fixes the input channel and returns a matrix for the existing Blender pipeline. | A general helper could supply a rectangular forward matrix and its ordered input and output axes, with explicit fixed-axis coordinates and an optional source grid. Blender can reorder to `xyz` and choose a thickness for 2D planes. |
| Resample a non-affine route | `sample_displacement_scene(path, output_grid)` evaluates a world-to-source pull edge and samples the data on a caller-specified grid. The loader consumes the baked array using the output grid’s affine. | Route lookup needs to distinguish a non-affine mapping from a missing route or payload. A dependency-free mapping descriptor with a caller-provided payload resolver would let the consumer choose interpolation, array backend, and caching. |

The displacement reader in `tests/scene_displacement.py` handles the direct nearest-neighbor edge in its fixture. It isolates the path-backed experiment while the public transform payload API is designed. Scene loading in the Blender file picker is a separate follow-up.
