"""Rebuild the Figure 6a schematic anatomy from archived DevCCF geometry."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("VTK_SMP_MAX_THREADS", "2")

import numpy as np
import pandas as pd
import pyvista as pv
from skimage.measure import marching_cubes

from feast_plate import FINAL
import native_geometry as native

sys.path.insert(0, str(FINAL / "reproduce_code/07_3d_transfer"))
from workflow import read_nifti_gz

AGE = "E18.5"
MASK = "E18.5/E18-5_MRI-MaskBrain_40um.nii.gz"


def geometry_frame(path):
    from anndata.experimental import read_elem
    import h5py
    with h5py.File(path) as file:
        frame = pd.DataFrame(file["obsm/spatial_3d"][:], columns=["x", "y", "z"])
        frame["region"] = read_elem(file["obs/region"])
        for axis in "ijk":
            frame[f"voxel_{axis}"] = file[f"obs/voxel_{axis}"][:]
    return frame


def brain_mask(path, image_path):
    with TemporaryDirectory() as temp:
        with ZipFile(path) as source:
            local = Path(temp) / Path(MASK).name
            local.write_bytes(source.read(MASK))
        volume, affine = read_nifti_gz(local)
    vertices, faces, _, _ = marching_cubes((volume > 0).astype(np.uint8), .5, step_size=2)
    vertices = vertices @ affine[:3, :3].T + affine[:3, 3]
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = 1.18 * max(np.max(np.abs((vertices - center) @ native.UP)),
                       np.max(np.abs((vertices - center) @ native.RIGHT)) / native.ASPECT)
    mesh = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    native.render({"center": center, "meshes": {"brain": mesh}}, {"brain": "#949ea4"}, scale, image_path)


def render(zenodo, output):
    output.mkdir(parents=True, exist_ok=True)
    results = zenodo / "results/07_3d_transfer"
    inputs = zenodo / "inputs/07_3d_transfer"
    atlas = results / "reconstruction/unified" / f"{AGE}_FEAST_3D.h5ad"
    frame = geometry_frame(atlas)
    geom = native.load_geometry(frame)
    cut = native.cutaway(frame, geom)
    regions = pd.read_csv(inputs / "ccc/region_schema.tsv", sep="\t")
    colors = dict(zip(regions.region_label, regions.hex_color))
    anatomical_extents = []
    for age in native.AGES:
        other = atlas.with_name(f"{age}_FEAST_3D.h5ad")
        # The archived camera scale was shared over the extents of both stages.
        import h5py
        with h5py.File(other) as file:
            xyz = file["obsm/spatial_3d"][:]
        delta = xyz - (xyz.max(axis=0) + xyz.min(axis=0)) / 2
        anatomical_extents.append(max(float(np.abs(delta @ native.UP).max()),
                                      float(np.abs(delta @ native.RIGHT).max()) / native.ASPECT))
    scale = 1.34 * max(anatomical_extents)
    native.render(cut, colors, scale, output / "cutaway.png")
    exterior = pv.Plotter(off_screen=True, window_size=native.WINDOW)
    exterior.ren_win.SetMultiSamples(0)
    exterior.ren_win.SetAlphaBitPlanes(1)
    exterior.set_background("white")
    exterior.enable_parallel_projection()
    exterior.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0)
    for label, mesh in geom["meshes"].items():
        exterior.add_mesh(mesh, color=colors[label], opacity=.30, smooth_shading=True,
                          ambient=.55, diffuse=.45, show_scalar_bar=False)
    center = geom["center"]
    lo, hi = np.asarray(geom["bounds"])
    plane = pv.Plane(center=(center[0], cut["cut_y_mm"], center[2]), direction=(0,1,0),
                     i_size=(hi[0]-lo[0])*1.10, j_size=(hi[2]-lo[2])*1.08)
    exterior.add_mesh(plane, color="#f4bd9b", opacity=.37, lighting=False)
    exterior.camera_position = [center+30*native.VIEW, center, native.UP]
    exterior.camera.parallel_scale = scale
    exterior.reset_camera_clipping_range()
    native.transparent_screenshot(exterior, output / "translucent.png")
    exterior.close()
    raw = zenodo / "inputs/raw/DevCCFv1_figshare_26377171/components/E18.5.zip"
    brain_mask(raw, output / "coordinate_system.png")
    callouts = {label: native.project(anchor, cut["center"], scale)[0].tolist()
                for label, anchor in cut["anchors"].items()}
    entry = {"atlas": str(atlas.relative_to(zenodo)), "whole_brain_mask": str(raw.relative_to(zenodo)),
             "mask_member": MASK, "age": AGE, "parallel_scale_mm": scale, "camera_view": native.VIEW.tolist(),
             "camera_up": native.UP.tolist(), "cut_y_mm": cut["cut_y_mm"], "retained_voxels": cut["retained_positions"],
             "cutaway_callouts_axes_xy": callouts,
             "display": "Schematic views of anatomical labels and DevCCF geometry."}
    (output / "provenance.json").write_text(json.dumps(entry, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=FINAL / "Zenodo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.zenodo, args.output or args.zenodo / "results/07_3d_transfer/figure06_sources/artwork")
