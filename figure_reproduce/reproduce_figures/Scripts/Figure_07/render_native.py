"""Render the eight native endpoint views using the archived route framing."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Figure_06"))
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("VTK_SMP_MAX_THREADS", "2")

import numpy as np
import pandas as pd
from PIL import Image

from feast_plate import FINAL
from layout import NATIVE_VIEW_HEIGHT
import native_routes as routes

CASES = ("Wnt7a--Fzd2_Lrp5", "Wnt7a--Fzd8_Lrp6")


def render(sources, output):
    output.mkdir(parents=True, exist_ok=True)
    common = routes.common
    frames = {age: pd.read_parquet(sources / f"{age}_geometry.parquet") for age in common.AGES}
    for frame in frames.values():
        assert np.array_equal(frame.global_row, np.arange(len(frame)))
    geometry = {age: common.load_geometry(frame) for age, frame in frames.items()}
    support = pd.read_csv(sources / "spatial_support.csv").set_index(["stage", "LR", "role"])
    metrics = pd.read_csv(sources / "top5_LR_metrics.csv").set_index("LR")
    edges, refs, totals, calibration = {}, {}, {}, []
    for lr, row in metrics.iterrows():
        key = f"{row.ligand}-{row.receptor}"
        values = []
        for age in common.AGES:
            edges[age, lr] = pd.read_parquet(sources / f"{age}_{key}_interface_edges.parquet")
            marginals = pd.read_parquet(sources / f"{age}_{key}_inter_region_marginals.parquet")
            totals[age, lr] = support.loc[(age, lr, "sender"), "inter_region_mass"]
            for role in ("sender", "receiver"):
                score = marginals[f"{role}_score"].to_numpy()
                np.testing.assert_allclose(score.sum(), totals[age, lr], rtol=1e-12)
                values.append(100 * score[score > 0] / totals[age, lr])
        refs[lr] = np.sort(np.concatenate(values))
        for p in (0, 25, 50, 75, 95, 99, 100):
            calibration.append({"LR": lr, "percentile": p, "pct_of_stage_inter_region_mass":
                                float(np.quantile(refs[lr], p / 100)), "reference_size": len(refs[lr])})
    local = routes.interface_geometry(frames, geometry, edges, sources)
    directions = []
    for lr in CASES:
        a, b = routes.FOCUS[lr]
        group = local[f"{a}_{b}"]
        scale = group["scale"]
        bounds = []
        records = []
        for mode, sender, receiver in (("forward", a, b), ("reverse", b, a)):
            for age in common.AGES:
                path = output / f"{lr.replace('--', '-')}_{age}_{mode}.png"
                stats = routes.local_scene(frames[age], group[age], edges[age, lr],
                                          sender, receiver, totals[age, lr], refs[lr], scale, path, False)
                with Image.open(path) as image:
                    assert image.mode == "RGBA" and image.size == routes.WINDOW
                    bounds.append(image.getchannel("A").getbbox())
                records.append({"LR": lr, "stage": age, "mode": mode, "sender_region": sender,
                                "receiver_region": receiver, "source": path.name, "render_pixels": routes.WINDOW,
                                "inter_region_mass": totals[age, lr], "view": common.VIEW.tolist(),
                                "up": common.UP.tolist(), "parallel_scale_mm": scale,
                                "midline_x_mm": group[age]["midline_x_mm"], "center_mm": group[age]["center"].tolist(),
                                "world_bounds_before_projection": group[age]["bounds"], **stats})
                print(f"{age} {lr} {sender} → {receiver}: {stats['n_edges_drawn']:,} right-hemisphere edges", flush=True)
        crop = (max(0, min(box[0] for box in bounds)-24), max(0, min(box[1] for box in bounds)-24),
                min(routes.WINDOW[0], max(box[2] for box in bounds)+24),
                min(routes.WINDOW[1], max(box[3] for box in bounds)+24))
        world_width = 2 * scale * (crop[2] - crop[0]) / routes.WINDOW[1]
        aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
        display_width = NATIVE_VIEW_HEIGHT * aspect
        directions.extend({**record, "crop_pixels": crop, "cropped_world_width_mm": world_width,
                           "display_width_pt": display_width, "scale_bar_pt": display_width/world_width,
                           "scale_bar_mm": 1.0} for record in records)
    (output / "provenance.json").write_text(json.dumps({
        "input": str(sources.name), "framing_lrs": metrics.index.tolist(), "voxel_mm": common.SPACING,
        "routes": directions, "denominator": "full atlas, all cross-region weights for the LR and stage",
        "color_reference": "positive full-atlas sender and receiver marginals, pooled across stages per LR",
    }, indent=2) + "\n")
    pd.DataFrame(calibration).to_csv(output / "pooled_color_reference.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=FINAL / "Zenodo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.zenodo / "results/07_3d_transfer/ccc_atlas"
    render(root / "top5_spatial", args.output or root / "native")
