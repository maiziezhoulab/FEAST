"""The archived regional network layout and width mapping."""
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as BezierPath
import numpy as np

WIDTH_PER_UNIT_SHARE = 30.0
BALANCED_WIDTH_FACTOR = .55


def layout(regions):
    angles = np.pi / 2 - np.arange(8) * np.pi / 4
    return {region: .93 * np.array([np.cos(angle), np.sin(angle)])
            for region, angle in zip(regions, angles)}


def edge_width(share, display_style):
    if display_style == 'balanced':
        return BALANCED_WIDTH_FACTOR * np.sqrt(100 * share)
    return WIDTH_PER_UNIT_SHARE * share


def network(ax, table, regions, colors, positions, display_style='linear'):
    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.85, 1.85)
    ax.set_aspect('equal'); ax.set_axis_off()
    nodes = {}
    for region in regions:
        nodes[region] = Circle(positions[region], .095, facecolor=colors[region],
                               edgecolor='white', linewidth=.7, zorder=4)
        ax.add_patch(nodes[region])
    # Every positive W is drawn; no top-edge, p-value or minimum-width filter.
    for edge in table.loc[table.share > 0].sort_values('share').itertuples():
        sender, receiver = edge.sender_region, edge.receiver_region
        width = edge_width(edge.share, display_style)
        kwargs = {'arrowstyle': '-|>', 'mutation_scale': max(2.5, 1.8 * width),
                  'linewidth': width, 'color': colors[sender], 'alpha': .75,
                  'zorder': 2, 'shrinkA': 0, 'shrinkB': 0, 'capstyle': 'round'}
        if sender == receiver:
            center = positions[sender]
            radial = center / np.linalg.norm(center)
            tangent = np.array([-radial[1], radial[0]])
            bulge, spread = (.32, .25) if display_style == 'balanced' else (.55, .43)
            points = [center + .066 * radial + .068 * tangent,
                      center + bulge * radial + spread * tangent,
                      center + bulge * radial - spread * tangent,
                      center + .066 * radial - .068 * tangent]
            path = BezierPath(points, [BezierPath.MOVETO, BezierPath.CURVE4, BezierPath.CURVE4, BezierPath.CURVE4])
            if display_style == 'balanced':
                arrow = PathPatch(path, facecolor='none', edgecolor='#818d93', alpha=.55,
                                  linewidth=width, capstyle='round', zorder=1)
            else:
                arrow = FancyArrowPatch(path=path, **kwargs)
        else:
            if display_style == 'balanced':
                kwargs.update({'mutation_scale': 5.0, 'alpha': .88})
            arrow = FancyArrowPatch(positions[sender], positions[receiver],
                                    patchA=nodes[sender], patchB=nodes[receiver],
                                    connectionstyle='arc3,rad=0.15', **kwargs)
        ax.add_patch(arrow)
    for region in regions:
        radial = positions[region] / .93
        point = radial * (1.45 if display_style == 'balanced' else 1.58)
        ha = 'left' if radial[0] > .3 else 'right' if radial[0] < -.3 else 'center'
        va = 'bottom' if radial[1] > .3 else 'top' if radial[1] < -.3 else 'center'
        ax.text(*point, region, ha=ha, va=va, fontsize=6.55, color='#263238', zorder=5)
