"""Render matched marker sections and expression-derived clusters on native coordinates."""
import json

import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from feast_plate import axes_box, limits, style_3d, text_at

DATA = "results/06_3d_stack/figure_s5"
GENES = ("Slc17a7","Gad2")
CONDITIONS = (("Observed","Observed"),("FEAST_gap3","Gap 3"),("FEAST_gap5","Gap 5"),("FEAST_gap10","Gap 10"))
CAMERA = {"elevation":25.,"azimuth":-55.,"roll":90.,"zoom":1.7}


def expression(fig,zenodo):
    """Draw two markers at every native position in the three matched gap-5 sections."""
    root = zenodo / DATA
    frame = pd.read_csv(root / "section_data.csv.gz",dtype={"cell_id":str})
    report = json.loads((root/"provenance.json").read_text())
    text_at(fig,277,4,"Section expression at gap 5",ha="center",va="top",size=10.9,weight="bold")
    text_at(fig,277,24,f"{report['gap5_measured_references']} reference sections",ha="center",va="top",size=8.4)
    for col,sid in enumerate(sorted(frame.target_slice.unique())):
        left = 36 + 156.5*col
        subset = frame.loc[frame.target_slice.eq(sid)]
        text_at(fig,left+73,43,f"Section {sid} · {subset.z.iloc[0]:.2f} mm",ha="center",va="top",size=8.35)
        low = subset[["x","y"]].min().to_numpy()
        high = subset[["x","y"]].max().to_numpy()
        pad = np.maximum(.025*(high-low),.01)
        for index,method in enumerate(("Observed","FEAST")):
            x = left + 78*index
            part = subset.loc[subset.method.eq(method)]
            text_at(fig,x+34,61,method,ha="center",va="top",size=8.3)
            for row,gene in enumerate(GENES):
                ax = axes_box(fig,x,74+84*row,69.5,74)
                positive = part.loc[part[gene].gt(0)].sort_values(gene,kind="stable")
                zero = part.loc[part[gene].eq(0)]
                ax.scatter(zero.x,zero.y,s=.23,c="#d6dce0",linewidths=0,rasterized=True)
                ax.scatter(positive.x,positive.y,s=.29,c=np.log1p(positive[gene]),cmap="viridis",
                           vmin=0,vmax=report["marker_log_count_caps"][gene],linewidths=0,rasterized=True)
                ax.set_xlim(low[0]-pad[0],high[0]+pad[0])
                ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
                ax.set_aspect("equal")
                ax.set_axis_off()
                if index==1:
                    start = low+pad
                    length = 1.
                    ax.plot([start[0],start[0]+length],[start[1],start[1]],color="#222222",lw=.69)
                    ax.text(start[0]+length/2,start[1]-.13,"1 mm",ha="center",va="top",size=7.6)
    for j,gene in enumerate(GENES):
        cap = report["marker_log_count_caps"][gene]
        top = 74+84*j
        text_at(fig,11,top+37,gene,ha="center",va="center",rotation=90,size=9.6,style="italic")
        norm = mpl.colors.Normalize(0,cap,clip=True)
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap="viridis"),cax=axes_box(fig,515,top,5,71),
                           ticks=np.arange(int(cap)+1))
        bar.ax.tick_params(length=1.5,labelsize=7.6,pad=1.6)
        bar.outline.set_linewidth(.5)
    text_at(fig,541.5,148,"log1p counts",ha="center",va="center",rotation=90,size=8)


def tissue(fig,points,top,bounds,triad):
    ax = axes_box(fig,329,top,185,99,projection="3d")
    ax.scatter(points.x,points.y,points.z,c=points.plot_color,s=.26,linewidths=0,alpha=.75,
               depthshade=False,rasterized=True,clip_on=False)
    style_3d(ax,bounds,CAMERA["elevation"],CAMERA["azimuth"])
    ax.view_init(elev=CAMERA["elevation"],azim=CAMERA["azimuth"],roll=CAMERA["roll"])
    ax.set_box_aspect(np.ptp(bounds,axis=1),zoom=CAMERA["zoom"])
    if triad:
        origin = np.array(bounds)[:,0]
        for i,label in enumerate("xyz"):
            endpoint = origin + np.eye(3)[i]*2.
            ax.plot(*np.stack([origin,endpoint]).T,c="#333333",lw=.65,clip_on=False)
            position = origin+1.25*(endpoint-origin)
            ax.text(*position,label,fontsize=7.6,color="#333333")


def clusters(fig,zenodo):
    """Show one common UMAP and native tissue view per independent expression solution."""
    root = zenodo / DATA
    data = pd.read_csv(root / "cluster_data.csv.gz",dtype={"cell_id":str,"leiden":str,"reference_leiden":str})
    summary = pd.read_csv(root / "condition_summary.csv").set_index("condition",verify_integrity=True)
    observed = summary.loc["Observed"]
    bounds = limits(data)
    umap = data[["umap_1","umap_2"]].to_numpy()
    low = umap.min(axis=0)
    high = umap.max(axis=0)
    pad = .04*(high-low)
    text_at(fig,277,4,"Expression clusters in embedding and tissue",ha="center",va="top",size=10.9,weight="bold")
    text_at(fig,277,23,f"{observed.n_cells:,} matched cells · {observed.n_sections} held-out sections",
            ha="center",va="top",size=8.5)
    text_at(fig,181,40,"UMAP",ha="center",va="top",size=9,weight="bold")
    text_at(fig,420,40,"3D cluster map",ha="center",va="top",size=9,weight="bold")
    for row,(condition,heading) in enumerate(CONDITIONS):
        top = 54+106*row
        frame = data.loc[data.condition.eq(condition)]
        text_at(fig,7,top+38,heading,ha="left",va="center",size=8.9,weight="bold")
        text_at(fig,7,top+54,f"{summary.loc[condition,'n_clusters']} clusters",ha="left",va="center",size=8.1)
        points = frame.iloc[np.random.default_rng(2026).permutation(len(frame))]
        ax = axes_box(fig,88,top,188,98)
        ax.scatter(points.umap_1,points.umap_2,c=points.plot_color,s=.21,linewidths=0,alpha=.65,rasterized=True)
        ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),aspect="equal")
        ax.set_axis_off()
        tissue(fig,points,top,bounds,triad=row==0)
    text_at(fig,181,484,"Observed-fitted UMAP",ha="center",va="top",size=8)
    text_at(fig,420,484,"Target coordinates (mm)",ha="center",va="top",size=8)
    labels = data[["display_label","plot_color"]].drop_duplicates().sort_values("display_label")
    width,height=fig.get_size_inches()*72
    for i,row in enumerate(labels.itertuples(index=False)):
        y,x = 510+(i//20)*10.6,8+(i%20)*26.4
        fig.add_artist(Rectangle((x/width,1-(y+2)/height),3.8/width,3.8/height,
                                 facecolor=row.plot_color,edgecolor="none",transform=fig.transFigure))
        text_at(fig,x+5.2,y,row.display_label,ha="left",va="center",size=6.6)


PLOTS = {"a":expression,"b":clusters}
