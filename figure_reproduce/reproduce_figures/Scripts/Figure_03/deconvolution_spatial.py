"""Study 03 spatial primitives from the registered public renderer."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Wedge
from scipy.spatial import cKDTree

RESOLUTIONS = ('0.25', '0.1')

SOURCE_LABELS = {'truth': 'Ground truth', 'cell2location': 'Cell2location', 'rctd': 'RCTD'}

BASE_CELL_TYPES = (('01 IT-ET Glut', '#3B73A5'), ('02 NP-CT-L6b Glut', '#5264C3'), ('03 OB-CR Glut', '#D66F36'), ('04 DG-IMN Glut', '#D997CC'), ('05 OB-IMN GABA', '#4C8B4A'), ('06 CTX-CGE GABA', '#8BAE3E'), ('07 CTX-MGE GABA', '#B94945'), ('08 CNU-MGE GABA', '#C9686A'), ('09 CNU-LGE GABA', '#8066A8'), ('10 LSX GABA', '#7189C2'), ('11 CNU-HYa GABA', '#8C6352'), ('12 HY GABA', '#A76B78'), ('13 CNU-HYa Glut', '#B25B9F'), ('14 HY Glut', '#CC7A9A'), ('18 TH Glut', '#6C7072'), ('19 MB Glut', '#577F5E'), ('20 MB GABA', '#857526'), ('21 MB Dopa', '#C9A43A'), ('22 MB-HB Sero', '#3E9FA6'), ('23 P Glut', '#3E9FA6'), ('24 MY Glut', '#B9B43B'), ('26 P GABA', '#9361B0'), ('27 MY GABA', '#3B9C87'), ('28 CB GABA', '#586EAE'), ('29 CB Glut', '#A84C79'), ('30 Astro-Epen', '#32A36F'), ('31 OPC-Oligo', '#3F7181'), ('32 OEC', '#69758F'), ('33 Vascular', '#7A4E39'), ('34 Immune', '#A84C79'))

ADDITIONAL_DISPLAY_TYPES = {'050': (('15 HY Gnrh1 Glut', '#267F9C'),), '100': (('16 HY MM Glut', '#267F9C'), ('25 Pineal Glut', '#8F6D4B'))}

OTHER_CELL_CLASS = ('Other', '#8A8A8A')

FULL_CELL_TYPES = tuple(sorted(BASE_CELL_TYPES + tuple((item for additions in ADDITIONAL_DISPLAY_TYPES.values() for item in additions)), key=lambda item: int(item[0].split(' ', maxsplit=1)[0])))

FULL_CELL_TYPE_COLORS = {label: colour for label, colour in FULL_CELL_TYPES}

FULL_RADIUS_SCALE = {'0.25': 0.37, '0.1': 0.38}

ROI_RADIUS_SCALE = 0.41

FULL_WEDGE_LINEWIDTH = 0.0

ROI_WEDGE_LINEWIDTH = 0.18

DISPLAY_THRESHOLD = 0.005

PROPORTION_EPSILON = 1e-09

ROI_X_FRACTION = (0.5, 0.72)

ROI_Y_FRACTION = (0.22, 0.5)

def nearest_neighbor_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        raise ValueError('At least two spatial centres are required')
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))

def common_limits(reference_xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = reference_xy.min(axis=0)
    upper = reference_xy.max(axis=0)
    padding = 0.018 * (upper - lower)
    return ((float(lower[0] - padding[0]), float(upper[0] + padding[0])), (float(lower[1] - padding[1]), float(upper[1] + padding[1])))

def roi_bounds(x_limits: tuple[float, float], y_limits: tuple[float, float]) -> tuple[float, float, float, float]:
    return (x_limits[0] + ROI_X_FRACTION[0] * (x_limits[1] - x_limits[0]), x_limits[0] + ROI_X_FRACTION[1] * (x_limits[1] - x_limits[0]), y_limits[0] + ROI_Y_FRACTION[0] * (y_limits[1] - y_limits[0]), y_limits[0] + ROI_Y_FRACTION[1] * (y_limits[1] - y_limits[0]))

def format_spatial_axis(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(y_limits[1], y_limits[0])
    ax.set_aspect('equal')
    ax.set_axis_off()

def pie_collection(xy: np.ndarray, proportions: np.ndarray, radius: float, palette: dict[str, str], *, crop: tuple[float, float, float, float] | None=None, linewidth: float=FULL_WEDGE_LINEWIDTH) -> PatchCollection:
    """Draw one named, clockwise pie glyph per location.

    The display threshold removes only wedges below 0.5% and the retained
    components are renormalized.  This makes sub-pixel slivers legible without
    changing the stored composition data or the order of the remaining wedges.
    """
    patches: list[Wedge] = []
    colors: list[str] = []
    if crop is None:
        selected = np.ones(len(xy), dtype=bool)
    else:
        x0, x1, y0, y1 = crop
        selected = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    for coordinate, values in zip(xy[selected], proportions[selected], strict=True):
        total = float(values.sum())
        if total <= PROPORTION_EPSILON:
            continue
        displayed = np.asarray(values, dtype=float) / total
        displayed[displayed < DISPLAY_THRESHOLD] = 0.0
        displayed_total = float(displayed.sum())
        if displayed_total <= PROPORTION_EPSILON:
            displayed[np.argmax(values)] = 1.0
        else:
            displayed /= displayed_total
        start_angle = 90.0
        for value, cell_type in zip(displayed, palette, strict=True):
            if value <= PROPORTION_EPSILON:
                continue
            end_angle = start_angle - 360.0 * float(value)
            patches.append(Wedge(tuple(coordinate), radius, end_angle, start_angle))
            colors.append(palette[cell_type])
            start_angle = end_angle
    return PatchCollection(patches, facecolor=colors, edgecolor='white', linewidth=linewidth, alpha=1.0, antialiased=True, rasterized=True, zorder=2)

def add_roi_rectangle(parent: plt.Axes, roi: tuple[float, float, float, float]) -> None:
    """Mark the fixed ROI without annotating or obscuring the tissue."""
    x0, x1, y0, y1 = roi
    parent.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#D95F4A', linewidth=0.68, zorder=5))
