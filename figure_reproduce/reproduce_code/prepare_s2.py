"""Reorganize and verify the three completed study outputs used in S2."""
import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CODE = Path(__file__).resolve().parent
for study in ("01_clustering","02_alignment","03_deconvolution"):
    sys.path.insert(0,str(CODE / study))
from intervention_profile import load_profile_data
from metric_utils import angular_error, ordered_reference_positions, recovered_rotation
from summarize_supplementary_classes import summarize


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def index(data, fields):
    return data.set_index(list(fields),verify_integrity=True).sort_index()


def old_new(current, previous, keys, metrics):
    table = current[list(keys)+list(metrics)].merge(previous[list(keys)+list(metrics)], on=list(keys),
                                                   suffixes=("_current","_article"), validate="one_to_one")
    assert len(table)==len(current)==len(previous)
    for field in metrics:
        table[f"{field}_change"] = table[f"{field}_current"]-table[f"{field}_article"]
    return table


def clustering(root, previous):
    summary = pd.read_csv(root / "expanded/report/combined_mean_variance_metrics.csv", dtype={"slice_id":str})
    base_metrics = pd.read_csv(root / "metrics/fixed_panel_benchmark_metrics.csv", dtype={"slice_id":str})
    reference = base_metrics.loc[base_metrics.simulation_id.eq("real_baseline")]
    assert len(summary) == 153 and len(reference) == 9
    assert summary.feast_version.eq("1.0.6").all() and summary.public_seed.eq(2026).all()
    source_fields = ["slice_id","simulation_id","method"]
    panels = []
    fold_changes = [.2,.5,.67,.8,1,1.25,1.5,2,5]
    for panel, factor in (("a","mean"),("b","variance")):
        part = summary.loc[summary.alteration_type.eq(factor) | summary.simulation_id.eq("baseline")]
        current = pd.concat([part,reference],ignore_index=True).assign(panel_id=panel)
        current["input_role"] = current.simulation_id.map({
            "baseline":"unaltered_simulation","real_baseline":"raw_reference"}).fillna("altered_simulation")
        current["requested_fold_change"] = current.alteration_value.where(current.input_role.eq("altered_simulation"), 1.)
        assert len(current)==90 and current.status.eq("ok").all()
        for _, group in current.loc[current.input_role.ne("raw_reference")].groupby(["slice_id","method"]):
            np.testing.assert_array_equal(np.sort(group.requested_fold_change),fold_changes)
        panels.append(current)
    scores = pd.concat(panels,ignore_index=True)
    assert len(scores.drop_duplicates(source_fields)) == 162
    expected = pd.read_csv(previous / "Fig_S2_clustering_scores.csv",dtype={"slice_id":str})
    expected["panel_id"] = expected.panel_id.str.lower()
    score_comparison = old_new(scores,expected,["panel_id",*source_fields],["ARI"])

    profile = load_profile_data(root)
    base = profile.loc[profile.factor.isin(["mean","variance"])].rename(columns={
        "mean":"mean_ratio","variance":"variance_ratio","zero_prop":"zero_fraction_ratio",
    })
    extra_dir = root / "expanded/simulations"
    realized = pd.read_csv(root / "expanded/report/realized_alteration_diagnostics.csv",dtype={"slice_id":str})
    extra = realized.rename(columns={
        "alteration_type":"factor","nominal_fc":"nominal_level","realized_mean_fc":"mean_ratio",
        "realized_variance_fc":"variance_ratio","realized_zero_prop_fc":"zero_fraction_ratio",
    })
    actual = pd.read_csv(extra_dir / "simulation_manifest.csv",dtype={"slice_id":str})
    assert len(extra) == len(actual) == 12 and actual.public_seed.eq(2026).all() and actual.feast_version.eq("1.0.6").all()
    key = ["slice_id","simulation_id"]
    expected_diagnostics = extra.set_index(key,verify_integrity=True)
    ratios = ["mean_ratio","variance_ratio","zero_fraction_ratio"]
    paths = {}
    for row in actual.itertuples(index=False):
        path = extra_dir / row.file_path
        a = ad.read_h5ad(path, backed="r")
        try:
            observed = a.uns["alteration_diagnostics"]["realized_stage_achieved_change"]
            np.testing.assert_allclose([float(observed[f]) for f in ("mean","variance","zero_prop")],
                                       expected_diagnostics.loc[(row.slice_id,row.simulation_id),ratios],
                                       rtol=0,atol=1e-12)
        finally:
            a.file.close()
        paths[row.slice_id,row.simulation_id] = str(path.relative_to(root))
    manifest = pd.read_csv(root / "run/simulation_manifest.csv",dtype={"slice_id":str})
    paths.update({(r.slice_id,r.simulation_id):f"run/{r.file_path}" for r in manifest.itertuples(index=False)})
    columns = ["slice_id","simulation_id","factor","nominal_level",*ratios]
    combined = pd.concat([base[columns],extra[columns]],ignore_index=True)
    combined["panel_id"] = combined.factor.map({"mean":"a","variance":"b"})
    combined["source_h5ad_from_study_root"] = [paths[k] for k in zip(combined.slice_id,combined.simulation_id)]
    assert len(combined) == 54
    for _, group in combined.groupby(["factor","slice_id"]):
        np.testing.assert_array_equal(np.sort(group.nominal_level),fold_changes)
    old_ratios = pd.read_csv(previous / "Fig_S2_clustering_diagnostics.csv",dtype={"slice_id":str})
    diagnostics_comparison = old_new(combined,old_ratios,["factor","nominal_level", *key],ratios)
    output = root / "s2"
    for table, name in ((scores,"clustering_scores"),(combined,"clustering_diagnostics"),
                        (score_comparison,"clustering_score_comparison"),
                        (diagnostics_comparison,"clustering_diagnostic_comparison")):
        write(table, output / f"{name}.csv")
    return {"score_rows":len(scores), "distinct_scores":len(scores.drop_duplicates(source_fields)),
            "full_count_ratios":len(combined), "seed":2026,
            "current_neutral_151676":score_comparison.loc[
                score_comparison.slice_id.eq("151676") & score_comparison.simulation_id.eq("baseline") &
                score_comparison.panel_id.eq("a"),["method","ARI_current","ARI_article"]].to_dict("records"),
            "maximum_absolute_ARI_change":float(score_comparison.ARI_change.abs().max()),
            "variance_x2_mean_realized_ratio":float(combined.loc[combined.factor.eq("variance") &
                combined.nominal_level.eq(2),"variance_ratio"].mean())}


def relative_fields(frame, columns, root):
    for field in columns:
        frame[field] = frame[field].map(lambda path: str(Path(path).resolve().relative_to(root.resolve())))


def deconvolution(root, previous, zenodo):
    classes, support = summarize(root / "cell_class")
    staging = root / "supplementary_diagnostics_S2"
    key = ["pair_id","method","cell_class"]
    fields = ["pearson","rmse","n_scored_spots","truth_mean_proportion"]
    saved = pd.read_csv(staging / "deconvolution_classes.csv",dtype={"slice_id":str})
    actual = index(classes,key)
    expected = index(saved,key)
    pd.testing.assert_index_equal(actual.index,expected.index)
    np.testing.assert_allclose(actual[fields],expected[fields],rtol=0,atol=1e-12,equal_nan=True)
    assert classes.loc[classes.is_other].pearson.isna().all()
    assert np.isfinite(classes.loc[classes.is_other].rmse).all()
    old = pd.read_csv(previous / "Fig_S2_class_key.csv",dtype={"slice_id":str})
    names = classes[["slice_id","class_order","cell_class"]].drop_duplicates()
    names = names.merge(old,on=["slice_id","class_order","cell_class"],validate="one_to_one")
    assert len(names)==39 and names.groupby("slice_id").size().to_dict()=={"007":8,"050":15,"100":16}
    historical = pd.read_csv(previous / "Fig_S2_deconvolution_classes.csv")
    compared = old_new(classes,historical,key,["pearson","rmse"])
    for frame in (classes,support):
        frame["panel_id"] = "c"
    relative_fields(classes,("source_path","truth_path"),zenodo)
    relative_fields(support,("simulation_path","truth_path"),zenodo)
    for frame,name in ((classes,"deconvolution_classes"),(support,"deconvolution_support"),(names,"class_key"),
                       (compared,"deconvolution_comparison")):
        write(frame, root / "s2" / f"{name}.csv")
    return {"class_method_rows":len(classes),"aggregate_scores_reconciled":12,
            "rmse_scale_maximum":float(classes.rmse.max()),
            "maximum_absolute_pearson_change":float(compared.pearson_change.abs().max()),
            "maximum_absolute_rmse_change":float(compared.rmse_change.abs().max()),
            "support":support[["pair_id","n_full_spots","n_scored_spots","n_retained_cell_classes"]].to_dict("records")}


def xy(path, key):
    data = ad.read_h5ad(path, backed="r")
    try:
        coords = pd.DataFrame(np.asarray(data.obsm[key],dtype=float), index=data.obs_names, columns=["x","y"])
    finally:
        data.file.close()
    assert coords.index.is_unique and np.isfinite(coords).all().all()
    return coords


def alignment(root, inputs, previous, zenodo):
    rotations = root / "rotations"
    fixed = xy(inputs / "151675.h5ad","spatial")
    bootstrap = xy(rotations / "base_simulations/baseline.h5ad","spatial")
    pd.testing.assert_frame_equal(fixed,bootstrap,check_exact=True)
    spacing = float(np.median(cKDTree(fixed.to_numpy()).query(fixed.to_numpy(),k=2)[0][:,1]))
    manifest = pd.read_csv(rotations / "rotation_manifest.csv").set_index(["alteration","angle_degrees"],verify_integrity=True)
    scores = pd.concat([pd.read_csv(root / "spateo_scores/alignment_metrics.csv"),
                        pd.read_csv(root / "paste2/scores/alignment_metrics.csv")],ignore_index=True)
    key = ["method","alteration","angle_degrees"]
    assert len(scores)==56 and not scores.duplicated(key).any() and scores.status.eq("ok").all()
    overlay, computed = [], []
    for method, directory_name in (("spateo","methods"),("paste2","paste2")):
        paths = root / directory_name
        runs = pd.read_csv(paths / "method_manifest.csv").set_index(["alteration","angle_degrees"],verify_integrity=True)
        selected = scores.loc[scores.method.eq(method)]
        for row in selected.itertuples(index=False):
            run = runs.loc[(row.alteration,row.angle_degrees)]
            rotation = rotations / manifest.loc[(row.alteration,row.angle_degrees),"output_path"]
            path = paths / run.output_dir / "aligned.h5ad"
            moving = xy(rotation,"spatial")
            aligned = xy(path,"spatial_aligned")
            assert (paths / run.moving).resolve() == rotation.resolve()
            assert (paths / run.reference).resolve() == (inputs / "151675.h5ad").resolve()
            assert moving.index.equals(aligned.index)
            match = ordered_reference_positions(fixed.index, aligned.index)
            assert len(match) == row.n_aligned
            expected = fixed.to_numpy()[match]
            displacement = float(np.linalg.norm(aligned.to_numpy()-expected,axis=1).mean()/spacing)
            angle = angular_error(recovered_rotation(moving.to_numpy(), aligned.to_numpy()), -row.angle_degrees)
            np.testing.assert_allclose([displacement,angle,spacing,len(match)/len(fixed)],
                                       [row.mean_spatial_error_spot_units,row.rotation_recovery_error,
                                        row.reference_median_spot_spacing,row.retained_fraction],
                                       rtol=0,atol=1e-10,err_msg=row.job_id)
            computed.append({"job_id":row.job_id,"aligned_h5ad":str(path.relative_to(zenodo)),
                             "moving_h5ad":str(rotation.relative_to(zenodo)),"recomputed_displacement":displacement,
                             "recomputed_rotation_error":angle})
            if row.alteration=="baseline" and row.angle_degrees==45:
                assert len(aligned)==2691
                values = aligned.rename(columns={"x":"aligned_x","y":"aligned_y"})
                values[["reference_x","reference_y"]] = expected
                overlay.append(values.rename_axis("spot_barcode").reset_index().assign(method=method,
                               source_path=str(path.relative_to(zenodo))))
    views = pd.concat(overlay,ignore_index=True)
    assert len(views)==5382 and len(fixed)==3565
    assert views.loc[views.method.eq("spateo"),"spot_barcode"].tolist()==views.loc[views.method.eq("paste2"),"spot_barcode"].tolist()
    scores = scores.merge(pd.DataFrame(computed),on="job_id",validate="one_to_one")
    old = pd.read_csv(previous / "Fig_S2_alignment_scores.csv")
    fields = ["mean_spatial_error_spot_units","rotation_recovery_error","retained_fraction"]
    comparison = old_new(scores,old,key,fields)
    np.testing.assert_allclose(comparison.retained_fraction_change,0,rtol=0,atol=1e-15)
    for frame, name in ((scores,"alignment_scores"),(views,"alignment_overlay"),(comparison,"alignment_comparison"),
                        (fixed.rename_axis("spot_barcode").reset_index(),"alignment_reference")):
        write(frame,root / "s2" / f"{name}.csv")
    columns = ["method","mean_spatial_error_spot_units_current","mean_spatial_error_spot_units_article",
               "rotation_recovery_error_current","rotation_recovery_error_article"]
    end = comparison.loc[comparison.alteration.eq("sparsity_0.5") & comparison.angle_degrees.eq(75),columns]
    return {"aligned_jobs_from_h5ad":len(scores),"reference_spots":len(fixed),"overlaid_rows":len(views),
            "reference_median_spacing":spacing,"sparsity_75_degree_comparison":end.to_dict("records")}


def prepare(zenodo):
    result, input_data = zenodo / "results", zenodo / "inputs"
    one, two, three = (result / study for study in ("01_clustering","02_alignment","03_deconvolution"))
    previous = two / "comparison/figure_s2_article"
    for root, role, report in ((one,"mean_and_variance_control",clustering(one,previous)),
                               (three,"deconvolution",deconvolution(three,previous,zenodo)),
                               (two,"alignment",alignment(two,input_data/"02_alignment",previous,zenodo))):
        (root / "s2/provenance.json").write_text(json.dumps({"analysis":role,**report},indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo",type=Path,default=Path(__file__).resolve().parent.parent / "Zenodo")
    prepare(p.parse_args().zenodo)
