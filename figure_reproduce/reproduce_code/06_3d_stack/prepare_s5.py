"""Verify native marker maps and the registered matched-cell expression clusters for S5."""
import argparse
import json
from pathlib import Path

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from prepare_maps import read_slice

STUDY = "06_3d_stack"
GENES = ("Slc17a7", "Gad2")
CONDITIONS = {"Observed":0,"FEAST_gap3":3,"FEAST_gap5":5,"FEAST_gap10":10}
ROW_KEYS = ["target_slice","spot_index"]
CLUSTER_KEYS = ["condition",*ROW_KEYS]
DTYPES = {"cell_id":str,"spot_id":str,"leiden":str,"reference_leiden":str}


def keyed(frame, fields):
    return frame.set_index(fields,verify_integrity=True).sort_index()


def native_markers(zenodo, generation, arms, inventory, common):
    sites = [common[0], common[len(common)//2], common[-1]]
    assert sites == [5,78,149]
    tables, sources, counts = [], [], {}
    for sid in sites:
        measured_path = zenodo / "inputs" / STUDY / "h5ad" / inventory.loc[sid,"filename"]
        feast_path = generation / "targets" / arms[5] / measured_path.stem / "generated.h5ad"
        measured, xy, xyz, _ = read_slice(measured_path)
        counts[sid] = len(measured)
        assert len(measured)==inventory.loc[sid,"n_obs"]
        np.testing.assert_allclose(xyz[:,2],inventory.loc[sid,"z"],rtol=0,atol=1e-10)
        for method,path in (("Observed",measured_path),("FEAST",feast_path)):
            obs, points, volume_points, expression = read_slice(path,genes=GENES)
            assert obs.index.is_unique and obs.index.equals(measured.index)
            np.testing.assert_array_equal(points,xy)
            np.testing.assert_array_equal(volume_points,xyz)
            np.testing.assert_array_equal(obs["class"].astype(str),measured["class"].astype(str))
            assert np.isfinite(expression).all() and (expression>=0).all()
            block = pd.DataFrame(volume_points,columns=["x","y","z"])
            block[list(GENES)] = expression
            tables.append(block.assign(panel_id="a",method=method,gap=0 if method=="Observed" else 5,
                                       target_slice=sid,spot_index=np.arange(len(obs)),cell_id=obs.index))
            sources.append(str(path.relative_to(zenodo)))
    table = pd.concat(tables,ignore_index=True)
    assert len(table)==111322
    plot_scales = {g:float(np.quantile(np.log1p(table.loc[table[g].gt(0),g]),.99)) for g in GENES}
    return table,plot_scales,sources,counts


def check_color_matching(embedding, structure):
    assignments = pd.read_csv(structure / "cluster_color_mapping.csv",dtype=DTYPES)
    bridge = keyed(assignments,["condition","leiden"])
    observed = keyed(embedding.loc[embedding.condition.eq("Observed")],ROW_KEYS)
    matches = []
    for condition,gap in CONDITIONS.items():
        chosen = keyed(embedding.loc[embedding.condition.eq(condition)],ROW_KEYS)
        pd.testing.assert_index_equal(chosen.index,observed.index)
        np.testing.assert_array_equal(chosen.cell_id,observed.cell_id)
        labels = sorted(chosen.leiden.unique(),key=int)
        observed_labels = sorted(observed.leiden.unique(),key=int)
        overlap = pd.crosstab(chosen.leiden,observed.leiden).reindex(index=labels,columns=observed_labels,fill_value=0)
        if gap == 0:
            match = {v:v for v in labels}
        else:
            rows,columns = linear_sum_assignment(-overlap.to_numpy())
            match = {labels[i]:observed_labels[j] for i,j in zip(rows,columns) if overlap.iloc[i,j]>0}
            counts = pd.read_csv(structure / "cluster_overlap.csv.gz",
                                 dtype={"generated_leiden":str,"observed_leiden":str})
            replica = counts.loc[counts.condition.eq(condition)].pivot(index="generated_leiden",
                columns="observed_leiden",values="cell_overlap")
            np.testing.assert_array_equal(overlap,replica.loc[labels,observed_labels])
        for label in labels:
            expected = bridge.loc[(condition,label)]
            ref = match.get(label)
            assert expected.n_cells == int(overlap.loc[label].sum())
            tag = f"O{int(ref)+1:02}" if ref is not None else f"G{gap}-U{int(label)+1:02}"
            assert expected.plot_label == tag
            subset = chosen.loc[chosen.leiden.eq(label)]
            assert subset.plot_label.eq(tag).all()
            if ref is None:
                assert pd.isna(expected.reference_leiden) and subset.reference_leiden.isna().all()
                assert expected.matched_cells==0
            else:
                assert expected.reference_leiden==ref and subset.reference_leiden.eq(ref).all()
                assert expected.matched_cells==int(overlap.loc[label,ref])
            matches.append({"condition":condition,"gap":gap,"leiden":label,"reference_leiden":ref,
                            "n_cells":expected.n_cells,"matched_cells":expected.matched_cells,"plot_label":tag})
    return pd.DataFrame(matches)


def verify_sample(zenodo,generation,arms,inventory,common,native,structure):
    reference = pd.read_csv(structure / "sampled_cells.csv.gz",dtype=DTYPES)
    assert len(reference)==91200 and sorted(reference.target_slice.unique())==common
    genes = pd.read_csv(structure / "genes.csv").gene.tolist()
    assert len(genes)==1122
    target_frames = []
    for sid,group in reference.groupby("target_slice",sort=True):
        row = inventory.loc[sid]
        positions = np.sort(np.random.default_rng(2026+sid).choice(row.n_obs,1200,replace=False))
        np.testing.assert_array_equal(group.spot_index,positions)
        name = str(row.filename)
        for condition,gap in CONDITIONS.items():
            path = (zenodo / "inputs" / STUDY / "h5ad" / name if gap==0 else
                    generation / "targets" / arms[gap] / Path(name).stem / "generated.h5ad")
            with h5py.File(path) as h5:
                obs = h5["obs"]
                index = obs[obs.attrs["_index"]].asstr()[positions]
                var = h5["var"]
                assert read_elem(var[var.attrs["_index"]]).tolist()==genes
                xy = h5["obsm/spatial_3d"][positions]
                np.testing.assert_array_equal(index,group.cell_id)
                np.testing.assert_allclose(xy[:,2],row.z,rtol=0,atol=1e-10)
            target_frames.append(pd.DataFrame(xy,columns=["x","y","z"]).assign(
                condition=condition,target_slice=sid,spot_index=positions,cell_id=index))
    verified = keyed(pd.concat(target_frames,ignore_index=True),CLUSTER_KEYS)
    display = native.loc[native.method.isin(["Observed","FEAST"]) & native.target_slice.isin(common)].copy()
    display["condition"] = np.where(display.method.eq("Observed"),"Observed","FEAST_gap"+display.gap.astype(str))
    display = keyed(display.rename(columns={"spot_id":"cell_id"}),CLUSTER_KEYS)
    pd.testing.assert_index_equal(verified.index,display.index)
    pd.testing.assert_frame_equal(verified,display[verified.columns],check_exact=True)
    span = verified[["x","y","z"]].to_numpy().reshape(4,91200,3)
    for i in range(1,4):
        np.testing.assert_array_equal(span[0],span[i])
    return verified


def compare_replay(embedding, replay, archived_analysis):
    if not (replay / "analysis.json").is_file():
        return None
    record = json.loads((replay / "analysis.json").read_text())
    assert record["status"]=="complete"
    for field in ("common_heldout_sections","parameters","environment"):
        assert record[field]==archived_analysis[field],field
    trial = keyed(pd.read_csv(replay / "embedding_plot_data.csv.gz",dtype=DTYPES),CLUSTER_KEYS)
    source = keyed(embedding,CLUSTER_KEYS)
    pd.testing.assert_index_equal(source.index,trial.index)
    np.testing.assert_array_equal(source.cell_id,trial.cell_id)
    verified = []
    for condition in CONDITIONS:
        previous,current = source.loc[condition],trial.loc[condition]
        verified.append({"condition":condition,"archived_clusters":previous.leiden.nunique(),
                         "replayed_clusters":current.leiden.nunique(),"partition_ari":adjusted_rand_score(previous.leiden,current.leiden),
                         "max_umap_absolute_difference":float(np.max(np.abs(
                             previous[["umap_1","umap_2"]].to_numpy()-current[["umap_1","umap_2"]].to_numpy())))})
    return verified


def prepare(zenodo):
    root = zenodo / "results" / STUDY
    output = root / "figure_s5"
    output.mkdir(parents=True,exist_ok=True)
    generation = root / "reconstruction"
    source = root / "expression_structure"
    old_plots = root / "comparison/figure_s5"
    analysis = json.loads((source / "analysis.json").read_text())
    plan = json.loads((generation / "plan.json").read_text())
    inventory = pd.read_csv(generation / "input_manifest.csv").set_index("slice_id",verify_integrity=True)
    common = sorted(set.intersection(*(set(arm["target_pool"]) for arm in plan["densities"].values())))
    assert len(common)==76 and common==analysis["common_heldout_sections"]
    arms = {arm["spec"]["gap"]:name for name,arm in plan["densities"].items()}
    sections, scales, marker_sources, counts = native_markers(zenodo,generation,arms,inventory,common)
    old_sections = pd.read_csv(old_plots / "Fig_S5_section_data.csv.gz")
    marker_key = ["method",*ROW_KEYS]
    active = keyed(sections,marker_key)
    prior = keyed(old_sections,marker_key)
    pd.testing.assert_index_equal(active.index,prior.index)
    np.testing.assert_allclose(active[["x","y","z",*GENES]],prior[["x","y","z",*GENES]],rtol=0,atol=1e-12)

    native = pd.read_parquet(root / "figure05_sources/native_volume.parquet",columns=[
        "method","gap","target_slice","spot_index","spot_id","x","y","z","source_kind"])
    coordinates = verify_sample(zenodo,generation,arms,inventory,common,native,source)
    embedding = pd.read_csv(source / "embedding_plot_data.csv.gz",dtype=DTYPES)
    assert len(embedding)==364800 and np.isfinite(embedding[["umap_1","umap_2"]]).all().all()
    assert embedding.groupby(["condition","target_slice"]).size().eq(1200).all()
    for condition in CONDITIONS:
        labels = pd.read_csv(source / f"{condition}_clusters.csv.gz",dtype=DTYPES)
        sample = embedding.loc[embedding.condition.eq(condition)]
        pd.testing.assert_frame_equal(keyed(sample,ROW_KEYS)[["leiden","cell_id"]],
                                      keyed(labels,ROW_KEYS)[["leiden","cell_id"]])
    matches = check_color_matching(embedding,source)
    combined = embedding.drop(columns="panel_id").merge(coordinates.reset_index(),on=[*CLUSTER_KEYS,"cell_id"],
                                                        validate="one_to_one").assign(panel_id="b")
    assert len(combined)==364800
    old_metadata = json.loads((old_plots / "Fig_S5_provenance.json").read_text())
    combined["plot_color"] = combined.plot_label.map(old_metadata["cluster_colors"])
    combined["display_label"] = combined.plot_label.map(old_metadata["cluster_display_labels"])
    assert combined[["plot_color","display_label"]].notna().all().all()
    old_embed = pd.read_csv(old_plots/"Fig_S5_cluster_data.csv.gz",dtype=DTYPES)
    fields = ["leiden","cell_id","plot_color","display_label","plot_label"]
    previous,current = keyed(old_embed,CLUSTER_KEYS),keyed(combined,CLUSTER_KEYS)
    pd.testing.assert_frame_equal(current[fields],previous[fields])
    coords = ["x","y","z","umap_1","umap_2"]
    np.testing.assert_allclose(current[coords],previous[coords],rtol=0,atol=1e-12)
    cluster_counts = combined.groupby("condition").leiden.nunique().to_dict()
    assert cluster_counts=={str(row["condition"]):row["n_clusters"] for row in analysis["condition_summary"]}
    keyed_builds = pd.read_csv(zenodo / "inputs" / STUDY / "article_settings/target_builds.csv")
    common_builds = keyed_builds.loc[keyed_builds.target_slice.isin(common)]
    builds = {int(gap):table.feast_build.value_counts().to_dict() for gap,table in common_builds.groupby("gap")}
    assert builds[3] == {"original":67,"precision":9}
    replay = compare_replay(embedding,root/"expression_structure_replay",analysis)
    sections.to_csv(output / "section_data.csv.gz",index=False)
    combined.to_csv(output / "cluster_data.csv.gz",index=False)
    matches.to_csv(output / "cluster_mapping.csv",index=False)
    pd.DataFrame([{"gene":gene,"lower":0,"upper":cap} for gene,cap in scales.items()]).to_csv(output/"color_scales.csv",index=False)
    pd.DataFrame(analysis["condition_summary"]).to_csv(output/"condition_summary.csv",index=False)
    report = {"source_run":"results/06_3d_stack/expression_structure/analysis.json",
              "study06_verification":"results/06_3d_stack/figure05_sources/verification.json",
              "figure_promotion_authorized":analysis["figure_promotion_authorized"],
              "native_marker_source_h5ads":marker_sources,"gap5_measured_references":len(plan["densities"][arms[5]]["reference_pool"]),
              "section_spots_per_method":counts,"gap5_section_rows":len(sections),"marker_log_count_caps":scales,
              "common_target_ids":common,"matched_cell_conditions":len(CONDITIONS),
              "verified_raw_coordinate_rows":len(coordinates),"cluster_counts":cluster_counts,
              "setup_and_environment":analysis["parameters"],"archived_cluster_environment":analysis["environment"],
              "common_target_builds":builds,"cluster_replay":replay}
    (output / "provenance.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo",type=Path,default=Path(__file__).resolve().parents[2] / "Zenodo")
    prepare(p.parse_args().zenodo)
