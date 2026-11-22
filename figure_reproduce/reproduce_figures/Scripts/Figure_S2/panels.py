"""Mean/variance ARI, per-class deconvolution and matched alignment diagnostics."""
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from feast_plate import BLUE, axes_box, text_at

CLUSTER = "results/01_clustering/s2"
DECONV = "results/03_deconvolution/s2"
ALIGN = "results/02_alignment/s2"
METHODS = {
    "GraphST":("GraphST",BLUE,"o"),
    "STAGATE_mclust":("STAGATE + mclust","#e69f00","s"),
    "Leiden_unsupervised":("Label-free Leiden","#009e73","^"),
}
METHOD_ALIGNMENT = {"paste2":("PASTE2","#165597","o"), "spateo":("Spateo","#b54440","s")}
ALTERATIONS = (("baseline","Unaltered"), ("mean_0.5","Mean ×0.5"),
               ("variance_2.0","Variance ×2"), ("sparsity_0.5","Sparsity −ln2"))


def style(axis):
    axis.spines[["top","right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#555555")
    axis.grid(axis="y",c="#d9d9d9",lw=.50)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both",labelsize=7.4,pad=1.8)


def log_x(axis):
    axis.set_xscale("log")
    axis.set_xticks([.2,.5,1,2,5],["0.2","0.5","1","2","5"])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlim(.174,5.6)


def clustering(fig, zenodo, factor, top, panel_id):
    table = pd.read_csv(zenodo / CLUSTER / "clustering_scores.csv", dtype={"slice_id":str})
    diagnostics = pd.read_csv(zenodo / CLUSTER / "clustering_diagnostics.csv", dtype={"slice_id":str})
    scores = table.loc[table.panel_id.eq(panel_id)]
    ratios = diagnostics.loc[diagnostics.factor.eq(factor)]
    if panel_id == "a":
        text_at(fig,272.4,4.2,"Downstream-task diagnostics",size=10,weight="bold",ha="center",va="top")
        fig.texts[0].set_y(1-31.1/(fig.get_size_inches()[1]*72))
        handles = [Line2D([],[],c=color,lw=.85,marker=marker,ms=3,label=name)
                   for name,color,marker in METHODS.values()]
        fig.legend(handles=handles,ncols=3,loc="upper center",bbox_to_anchor=(.475,.795),fontsize=7.45,
                   handletextpad=.44,columnspacing=1.1,handlelength=1.4)
        text_at(fig,559.5,48.6,"Dashed: raw reference · Open: unaltered simulation",
                ha="right",va="center",size=7,color="#51555a")
    text_at(fig,272.4,top-35.7,f"{factor.title()} control",ha="center",va="top",size=11,weight="bold")
    lim = max(.67, float(table.ARI.max())+.025)
    for col, sid in enumerate(("151508","151670","151676")):
        ax = axes_box(fig,34.33+126.57*col,top,103.95,79.51)
        part = scores.loc[scores.slice_id.eq(sid)]
        for method, (_, color, marker) in METHODS.items():
            series = part.loc[part.method.eq(method)]
            reference = series.loc[series.input_role.eq("raw_reference"),"ARI"].item()
            neutral = series.loc[series.input_role.eq("unaltered_simulation"),"ARI"].item()
            curve = series.loc[series.input_role.ne("raw_reference")].sort_values("requested_fold_change")
            ax.axhline(reference,color=color,ls=(0,(2,2)),lw=.60,alpha=.8)
            ax.plot(curve.requested_fold_change, curve.ARI,color=color,marker=marker,ms=2.3,lw=.9)
            ax.plot([1],[neutral],color=color,marker=marker,ms=4,markerfacecolor="white",mew=.85,zorder=4)
        style(ax)
        log_x(ax)
        ax.set_ylim(min(0,float(table.ARI.min())-.018),lim)
        ax.set_yticks([0,.3,.6])
        ax.set_title(sid,size=8.5,weight="bold",pad=4)
        if col==0:
            ax.set_ylabel("Ground-truth ARI ↑",size=8.4,labelpad=2.8)
        else:
            ax.tick_params(axis="y",labelleft=False)
    ax = axes_box(fig,433.96,top,96.95,79.51)
    ax.axhline(1,color="#9c9c9c",ls=":",lw=.6)
    ax.plot([.2,5],[.2,5],c="#a8a8a8",ls="--",lw=.72)
    names = (("mean_ratio","Mean","#5b5b5b"),("variance_ratio","Variance","#7b5a91"),
             ("zero_fraction_ratio","Zero fraction","#668c7a"))
    for field,name,color in names:
        for _, sample in ratios.groupby("slice_id"):
            sample = sample.sort_values("nominal_level")
            ax.plot(sample.nominal_level,sample[field],c=color,lw=.58,alpha=.33)
        center = ratios.groupby("nominal_level")[field].mean()
        ax.plot(center.index,center,c=color,lw=1.05,marker="o",ms=2.0,label=name)
    log_x(ax)
    style(ax)
    high = float(ratios[list(n[0] for n in names)].max().max())
    ax.set_ylim(0,max(5.2, high*1.04))
    ax.set_yticks([0,2.5,5])
    ax.set_title("Count ratio to raw",size=8,weight="bold",pad=4)
    ax.legend(loc="upper left",fontsize=7.0,ncols=1,handlelength=1.28,columnspacing=.1,
              labelspacing=.15,handletextpad=.4,borderaxespad=.4)
    text_at(fig,215.8,top+100,f"Requested {factor} fold change",ha="center",va="center",size=8)
    text_at(fig,482.44,top+100,"Requested fold change",ha="center",va="center",size=7)


def mean_control(fig,zenodo):
    """Draw the nine requested mean factors and realized count ratios."""
    clustering(fig,zenodo,"mean",67.69,"a")


def variance_control(fig,zenodo):
    """Draw the nine requested variance factors and realized count ratios."""
    clustering(fig,zenodo,"variance",39.59,"b")


def class_comparison(fig,zenodo):
    """Show the 156 class-method Pearson and RMSE values with the recorded order."""
    root = zenodo / DECONV
    data = pd.read_csv(root / "deconvolution_classes.csv",dtype={"slice_id":str})
    key = pd.read_csv(root / "class_key.csv",dtype={"slice_id":str})
    text_at(fig,277.8,3,"Deconvolution by cell class",ha="center",va="top",size=10.7,weight="bold")
    text_at(fig,277.8,15,"R: RCTD · C: Cell2location · –: unavailable",ha="center",va="top",size=7)
    columns = pd.MultiIndex.from_product([[.10,.25],["rctd","cell2location"]])
    heatmaps = {}
    for col,sid in enumerate(("007","050","100")):
        subset = data.loc[data.slice_id.eq(sid)]
        labels = key.loc[key.slice_id.eq(sid)].sort_values("class_order")
        start = 74.23 + col*177.72
        text_at(fig,start+47.35,27,f"Slice {sid}",ha="center",va="top",size=10)
        for j,(metric,title,cmap,lo,hi) in enumerate((
            ("pearson","Pearson r","RdBu",-1,1),("rmse","RMSE","Blues_r",0,data.rmse.max()),
        )):
            x = start+j*55.44
            width = 48.48
            ax = axes_box(fig,x,51.11,width,7.57*len(labels))
            values = subset.pivot(index="cell_class",columns=["resolution","method"],values=metric)
            matrix = values.loc[labels.cell_class,columns].to_numpy()
            palette = mpl.colormaps[cmap].copy()
            palette.set_bad("#e3e3e3")
            heatmaps[metric] = ax.imshow(matrix,norm=mpl.colors.Normalize(lo,hi),cmap=palette,
                                         aspect="auto",interpolation="nearest")
            ax.set_title(title,weight="bold",size=8,pad=2)
            ax.set_xticks([])
            ax.set_yticks(range(len(labels)), labels.display_label if j==0 else [""]*len(labels),fontsize=7.35)
            ax.tick_params(length=0,pad=2)
            ax.spines[:].set_visible(False)
            ax.axvline(1.5,c="white",lw=1.8)
            for i,k in np.argwhere(~np.isfinite(matrix)):
                ax.text(k,i,"–",color="#555555",ha="center",va="center",size=7)
            for k, letter in enumerate("RCRC"):
                text_at(fig,x+width*(k+.5)/4,177,letter,ha="center",va="top",size=7.2)
            for k, factor in enumerate(("0.10","0.25")):
                text_at(fig,x+width*(k+.5)/2,190.1,factor,ha="center",va="top",size=7)
    for col,(name,label,ticks,fmt) in enumerate((
        ("pearson","Pearson r ↑",[-1,0,1],"{:g}"),
        ("rmse","RMSE ↓",[0,data.rmse.max()/2,data.rmse.max()],"{:.4f}"),
    )):
        cbar = fig.colorbar(heatmaps[name],cax=axes_box(fig,100.03+215.4*col,205.1,123.9,4.2),
                            orientation="horizontal",ticks=ticks)
        cbar.set_ticklabels([fmt.format(v).replace("-","−") for v in ticks])
        cbar.ax.set_title(label,size=7.35,pad=1.9,weight="bold")
        cbar.ax.tick_params(length=1.3,pad=1.0,labelsize=7.1)
        cbar.outline.set_visible(False)


def padded(values):
    span = float(values.max()-values.min())
    return float(values.min()-.03*span),float(values.max()+.03*span)


def alignment(fig,zenodo):
    """Plot the 56 true alignment scores and both matched 45-degree overlays."""
    root = zenodo / ALIGN
    scores = pd.read_csv(root / "alignment_scores.csv")
    reference = pd.read_csv(root / "alignment_reference.csv")
    overlay = pd.read_csv(root / "alignment_overlay.csv")
    text_at(fig,272.4,3.3,"Alignment under rotation and partial overlap",ha="center",va="top",size=11,weight="bold")
    handles = [Line2D([],[],c=color,marker=marker,ms=3,lw=1,label=name) for name,color,marker in METHOD_ALIGNMENT.values()]
    fig.legend(handles=handles,ncols=2,loc="upper center",bbox_to_anchor=(.41,.924),fontsize=8.6,handlelength=2)
    labels = (("mean_spatial_error_spot_units","Spatial error ↓\n(spot spacings)"),
              ("rotation_recovery_error","Rotation error (°) ↓"))
    show_angles = [5,45,75]
    retained = scores.groupby("angle_degrees").retained_fraction.first()
    for col, (condition,title) in enumerate(ALTERATIONS):
        for row,(metric,ylabel) in enumerate(labels):
            ax = axes_box(fig,40.8+91.56*col,52.38+68.03*row,75.4,51.02)
            for method,(_,color,marker) in METHOD_ALIGNMENT.items():
                group = scores.loc[scores.method.eq(method) & scores.alteration.eq(condition)].sort_values("angle_degrees")
                ax.plot(group.angle_degrees,group[metric],c=color,marker=marker,ms=2.7,lw=1)
            maximum = float(scores[metric].max())
            ax.set(xlim=(0,80),ylim=(-.04*maximum,1.11*maximum),xticks=show_angles)
            style(ax)
            if row==0:
                ax.set_title(title,fontsize=7.9,weight="bold",pad=4)
                ax.tick_params(axis="x",labelbottom=False)
            else:
                ax.set_xticklabels([f"{angle}°\n{100*retained.loc[angle]:.0f}%" for angle in show_angles],fontsize=7.1)
            ax.ticklabel_format(axis="y",style="plain",useOffset=False)
            if col==0:
                ax.set_ylabel(ylabel,size=7.65,labelpad=1.9)
            else:
                ax.tick_params(axis="y",labelleft=False)
    text_at(fig,215.8,203,"Input angle / retained overlap",ha="center",va="top",size=8)
    text_at(fig,479.8,30.5,"Unaltered, 45°",ha="center",va="top",size=8)
    x,y = (padded(np.r_[reference[k],overlay[f"aligned_{k}"]]) for k in ("x","y"))
    for row,(method,(name,color,_)) in enumerate(METHOD_ALIGNMENT.items()):
        ax = axes_box(fig,421.3,48.5+78.24*row,117,66.5)
        selected = overlay.loc[overlay.method.eq(method)]
        ax.scatter(reference.x,reference.y,s=1.2,c="#d4d4d4",linewidths=0,rasterized=True)
        ax.scatter(selected.aligned_x,selected.aligned_y,s=1,c=color,linewidths=0,rasterized=True)
        ax.set(xlim=x,ylim=y[::-1],aspect="equal")
        ax.set_axis_off()
        ax.set_title(name,fontsize=8.1,weight="bold",pad=1.0)
    n = int(len(overlay)/2)
    text_at(fig,479.8,200,f"{n:,} / {len(reference):,} ({100*n/len(reference):.1f}%) retained\nGray: complete reference",
            ha="center",va="top",size=7,linespacing=1.15)


PLOTS = {"a":mean_control,"b":variance_control,"c":class_comparison,"d":alignment}
