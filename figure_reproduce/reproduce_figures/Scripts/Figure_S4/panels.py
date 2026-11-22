"""Display the registered 2D matched-coordinate, Moran and support diagnostics."""
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from feast_plate import BLUE, axes_box, text_at

DATA = "results/05_2d_conditional_transfer/figure_s4"
Y = np.array([0,1,2.8,3.8,4.8,5.8,7.6,8.6,9.6,11.4,12.4,14.2,15.2])
DIRECTION_STYLES = {
    "151675_to_151676":(BLUE,"-"), "151676_to_151675":(BLUE,"--"),
    "151670_to_151675":("#d55e00","-"), "151675_to_151670":("#d55e00","--"),
    "151670_to_151676":("#009e73","-"), "151676_to_151670":("#009e73","--"),
    "Zhuang-ABCA-1.006_to_Zhuang-ABCA-1.007":(BLUE,"-"),
    "Zhuang-ABCA-1.007_to_Zhuang-ABCA-1.006":(BLUE,"--"),
}
COLORS = ("#0072b2", "#d55e00", "#009e73")


def short(sample):
    return str(sample).replace("Zhuang-ABCA-1.", "")


def task_label(row):
    return (f"{short(row.source)} · half-slice" if row.mode == "mask_half"
            else f"{short(row.source)} → {short(row.target)}")


def setup(zenodo, axis):
    tasks = pd.read_csv(zenodo / DATA / "task_summary.csv", dtype={"source":str,"target":str})
    axis.set(ylim=(16.1,-1.5),yticks=[])
    axis.spines[["left","right","top"]].set_visible(False)
    for first in tasks.groupby("task_group",sort=False).head(1).index[1:]:
        axis.axhline(Y[first]-1.02,color="#e8e8e8",lw=.65,zorder=0)
    axis.tick_params(axis="x",labelsize=8)
    return tasks


def pearson(fig, zenodo):
    """Render the task-wise gene Pearson distributions and finite counts."""
    text_at(fig,143,4,"Coordinate-wise\ngene agreement",ha="center",va="top",weight="bold",size=10.1,linespacing=1)
    ax = axes_box(fig,132,41,137,309)
    tasks = setup(zenodo, ax)
    genes = pd.read_csv(zenodo / DATA / "gene_data.csv.gz",usecols=["job_id","pearson","pearson_defined"])
    groups = [genes.loc[genes.job_id.eq(task) & genes.pearson_defined, "pearson"].to_numpy() for task in tasks.job_id]
    parts = ax.violinplot(groups, positions=Y, orientation="horizontal", widths=.66,
                          showmeans=False, showextrema=False, showmedians=False, points=120)
    for body in parts["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor(BLUE)
        body.set_alpha(.38)
        body.set_linewidth(.5)
    for y, group, task in zip(Y,groups,tasks.itertuples(index=False)):
        q1, median, q3 = np.quantile(group,[.25,.5,.75])
        ax.plot([q1,q3],[y,y],c=BLUE,lw=2,solid_capstyle="round")
        ax.plot(median,y,marker="|",c="#111111",ms=5,mew=.9)
        ax.text(-.035,y-.11,task_label(task),transform=ax.get_yaxis_transform(),
                ha="right",va="center",size=7.8,clip_on=False)
        ax.text(-.035,y+.29,f"{task.n_finite_pearson:,} / {task.n_gene_rows:,}",
                transform=ax.get_yaxis_transform(), ha="right",va="center",size=7.45,color="#555555",clip_on=False)
    for index, title in zip(tasks.groupby("task_group",sort=False).head(1).index,tasks.task_group.unique()):
        ax.text(-.033,Y[index]-.68,title,transform=ax.get_yaxis_transform(),
                ha="right",va="center",weight="bold",size=7.9,color="#454545",clip_on=False)
    ax.axvline(0,c="#999999",lw=.65,linestyle=(0,(2,2)),zorder=0)
    ax.set(xlim=(-1,1),xticks=[-1,0,1],xlabel="Gene Pearson r")
    ax.xaxis.label.set_size(8.8)
    text_at(fig,8,380,"Counts: finite / full-panel genes",size=7.7,va="top",color="#454545")


def moran(fig, zenodo):
    """Show one ordinary Moran-profile correlation per registered task."""
    text_at(fig,58,4,"Spatial-profile\nagreement",ha="center",va="top",weight="bold",size=10.1,linespacing=1)
    ax = axes_box(fig,11,41,88,309)
    tasks = setup(zenodo,ax)
    ax.scatter(tasks.moran_corr,Y,s=13,c=BLUE,linewidths=0,zorder=3)
    for y,value in zip(Y,tasks.moran_corr):
        ax.text(.025,y,f"{value:.3f}",va="center",size=7.6,color="#555555")
    ax.set(xlim=(0,1),xticks=[0,.5,1],xlabel="Moran-profile r")
    ax.xaxis.label.set_size(8.8)


def coverage(fig, zenodo):
    """Show retained and excluded positions out of each candidate target."""
    text_at(fig,56,4,"Retained target\ncoverage",ha="center",va="top",weight="bold",size=10.1,linespacing=1)
    ax = axes_box(fig,10,41,95,309)
    tasks = setup(zenodo,ax)
    ax.barh(Y-.08,100,height=.27,color="#e1e1e1",lw=0)
    ax.barh(Y-.08,tasks.target_retained_fraction*100,height=.27,color=BLUE,lw=0)
    for i, row in enumerate(tasks.itertuples(index=False)):
        ax.text(0,Y[i]+.31,f"{row.target_retained_spots:,} / {row.target_candidate_spots:,}",
                va="center",size=7.45,color="#444444")
    ax.set(xlim=(0,100), xticks=[0,50,100], xlabel="Retained spots (%)")
    ax.xaxis.label.set_size(8.8)


def ar_curves(fig,zenodo,metric,title,ylabel,ticks,limits,legend):
    text_at(fig,260,7,title,ha="center",va="top",weight="bold",size=10.3)
    ar = pd.read_csv(zenodo / DATA / "ar_data.csv",dtype={"source":str,"target":str})
    for col,dataset in enumerate(("dlpfc","merfish")):
        part = ar.loc[ar.dataset.eq(dataset)]
        ax = axes_box(fig,44+256*col,36,205,69)
        handles = []
        for direction, (color,linestyle) in DIRECTION_STYLES.items():
            group = part.loc[part.direction.eq(direction)].sort_values("assignment_randomness")
            if group.empty:
                continue
            ax.plot(group.assignment_randomness,group[metric],c=color,ls=linestyle,lw=1.08,marker="o",ms=2.7)
            handles.append(Line2D([],[],c=color,ls=linestyle,marker="o",ms=2.4,lw=1,label=task_label(group.iloc[0])))
        ax.axvline(.3,color="#909090",ls=":",lw=.65)
        low,high = min(limits[0],part[metric].min()-.001),max(limits[1],part[metric].max()+.001)
        ax.set(xlim=(-.02,.52), xticks=[0,.1,.2,.3,.5], yticks=ticks, ylim=(low,high))
        ax.set_xlabel("Assignment randomness",fontsize=8.15,labelpad=2)
        ax.set_ylabel(ylabel,fontsize=8.15,labelpad=2)
        ax.set_title(dataset.upper(),fontsize=8.8,pad=3)
        ax.tick_params(axis="both",labelsize=7.7)
        ax.grid(axis="y",color="#d7d7d7",lw=.52)
        ax.spines[["top","right"]].set_visible(False)
        if legend:
            fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.055+.564*col,.014),
                       ncols=2 if col==0 else 1,fontsize=7.4,handlelength=2,columnspacing=1.15)


def pearson_sensitivity(fig, zenodo):
    """Show fixed-seed coordinate agreement across the eight cross-section directions."""
    ar_curves(fig,zenodo,"median_gene_pearson","Assignment randomness: coordinate-wise agreement",
              "Median gene r",[0,.04,.08],(-.008,.095),False)


def moran_sensitivity(fig, zenodo):
    """Show fixed-seed Moran-profile agreement across the eight cross-section directions."""
    ar_curves(fig,zenodo,"moran_corr","Assignment randomness: spatial-profile agreement",
              "Moran-profile r",[.5,.7,.9],(.5,.98),True)


PLOTS = {"a":pearson,"b":moran,"c":coverage,"d":pearson_sensitivity,"e":moran_sensitivity}
