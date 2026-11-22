"""Verify Study 04A support, GraphST sensitivity and endpoint confusion for S3."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from metric_support import ALL_SPOTS, PAIRED_SUPPORT, REPRESENTATIONS
from prepare_figure_data import (K, load_inputs, predict_and_verify, read_slice, simulation_path,
                                 trained_embedding, uncorrected_pair)

STUDY = "04_batch_effect_removal"
METRICS = ("batch_mixing_entropy_k30", "domain_purity_k30", "paired_retrieval_top1")
MODES = ("shift_only", "diagonal_affine")
LAYERS = tuple(f"Layer_{i}" for i in range(1,7)) + ("WM",)
METHODS = ("Uncorrected","GraphST","STAMP","scVI")
STRENGTHS = (.25,.5,.75,1,1.25,1.5)


def sensitivity(root):
    analysis = root / "same_slice"
    files = analysis / "metrics"
    atomic = pd.read_csv(files / "atomic_metrics.csv")
    graphst = atomic.loc[atomic.method.eq("GraphST")].set_index(
        ["mode", "alpha", "support_variant", "representation"], verify_integrity=True)
    saved = pd.read_csv(files / "graphst_pca20_vs_pca10_sensitivity.csv")
    audit = pd.read_csv(files / "support_audit.csv").set_index(["mode","alpha"], verify_integrity=True)
    genes = pd.read_csv(analysis / "panel/panel.csv").gene.tolist()
    assert len(genes) == 2000 and len(set(genes)) == 2000
    rows = []
    for mode in MODES:
        base = ad.read_h5ad(analysis / "simulations" / mode / "alpha_0.00.h5ad")
        assert np.all(np.asarray(base[:, genes].X.sum(axis=1)).ravel() > 0)
        for alpha in STRENGTHS:
            query = ad.read_h5ad(analysis / "simulations" / mode / f"alpha_{alpha:.2f}.h5ad")
            assert base.obs_names.equals(query.obs_names)
            zeros = np.asarray(query[:, genes].X.sum(axis=1)).ravel() == 0
            excluded = query.obs_names[zeros].tolist()
            support = audit.loc[(mode,alpha)]
            assert support.n_spots_per_batch==len(base)==3611
            assert excluded == json.loads(support.fixed_panel_zero_query_spot_ids)
            assert len(excluded) == support.paired_support_removed_paired_spots
            for variant, removed in [(ALL_SPOTS,[]), *([(PAIRED_SUPPORT,excluded)] if excluded else [])]:
                n = len(query) - len(removed)
                left, right = [graphst.loc[(mode,alpha,variant,rep)] for rep in REPRESENTATIONS["GraphST"]]
                source = saved.loc[saved["mode"].eq(mode) & saved.alpha.eq(alpha) & saved.support_variant.eq(variant)]
                assert len(source) == 1 and source.removed_paired_spots.item() == len(removed)
                assert left.n_reference == left.n_query == right.n_query == right.n_reference == n
                for metric in METRICS:
                    delta = right[metric]-left[metric]
                    np.testing.assert_allclose(
                        [left[metric],right[metric],delta],
                        source[[f"{metric}_pca20",f"{metric}_pca10",f"delta_{metric}_pca10_minus_pca20"]].iloc[0],
                        rtol=0,atol=1e-12)
                    rows.append({"panel_id":"a","mode":mode,"alpha":alpha,"support_variant":variant,
                                 "excluded_pair_count":len(removed),"excluded_pair_ids":json.dumps(removed),
                                 "n_pairs":n,"metric":metric,"pca20":left[metric],"pca10":right[metric],"difference":delta})
    assert len(rows) == 3*len(saved)
    return pd.DataFrame(rows)


def confusion(root, inputs):
    plot_root = root / "plot_data/same_slice"
    predictions = pd.read_csv(plot_root / "same_slice_layer_transfer_spatial_predictions.csv", dtype={"source_spot_id":str})
    stored = pd.read_csv(plot_root / "same_slice_layer_transfer_spatial_scores.csv")
    stored_layers = pd.read_csv(plot_root / "same_slice_layer_transfer_spatial_per_layer_scores.csv")
    key = ["mode","alpha","method","source_spot_id"]
    assert len(predictions) == 14444 and set(predictions["mode"]) == {"diagonal_affine"}
    assert not predictions.duplicated(key).any()
    run = root / "same_slice"
    manifest, genes = load_inputs(run)
    metrics = pd.read_csv(run / "metrics/atomic_metrics.csv")
    primary = metrics.loc[metrics.analysis_role.eq("primary")]
    endpoint_predictions, endpoint_scores, endpoint_layers = [], [], []
    for mode in MODES:
        ref = read_slice(simulation_path(run,manifest,0.,mode),genes)
        query = read_slice(simulation_path(run,manifest,1.,mode),genes)
        uncorrected, _ = uncorrected_pair(ref,query)
        _, ids, labels, coordinates = ref
        query_labels = query[2]
        for method in METHODS:
            values = uncorrected if method=="Uncorrected" else trained_embedding(run,primary,method,ids,labels,mode)
            calls, score, layer_rows = predict_and_verify(values,labels,query_labels)
            info = {"mode":mode,"alpha":1.0,"method":method}
            endpoint_predictions.append(pd.DataFrame({**info,"source_spot_id":ids,
                "x":coordinates[:,0],"y":coordinates[:,1],"ground_truth":query_labels,"predicted_layer":calls,
                "correct":calls==query_labels}))
            endpoint_scores.append({**info,"n_reference":len(ids),"n_query":len(ids),"k":K,"macro_f1":score})
            endpoint_layers.extend({**info,**row} for row in layer_rows)
    all_predictions = pd.concat(endpoint_predictions,ignore_index=True)
    all_scores = pd.DataFrame(endpoint_scores)
    all_layers = pd.DataFrame(endpoint_layers)
    for current, parked, identifiers in ((all_predictions,predictions,key),(all_scores,stored,key[:3]),
                                         (all_layers,stored_layers,[*key[:3],"layer"])):
        diag = current.loc[current["mode"].eq("diagonal_affine")].set_index(identifiers).sort_index()
        pd.testing.assert_frame_equal(diag,parked.set_index(identifiers).sort_index(),
                                      rtol=0,atol=1e-12,check_dtype=False)
    predictions, stored, stored_layers = all_predictions, all_scores, all_layers
    assert len(predictions) == 28888 and not predictions.duplicated(key).any()
    assert set(predictions["mode"]) == set(MODES) and set(predictions.method) == set(METHODS)
    assert predictions.alpha.eq(1).all() and stored.k.eq(30).all()
    assert set(predictions.ground_truth) == set(LAYERS) and set(predictions.predicted_layer) <= set(LAYERS)
    assert (predictions.correct == predictions.ground_truth.eq(predictions.predicted_layer)).all()
    adata = ad.read_h5ad(inputs / "151673.h5ad", backed="r")
    try:
        measured = pd.DataFrame(adata.obsm["spatial"][:,:2], index=adata.obs_names, columns=["x","y"])
        label_key = "dlpfc_layer" if "dlpfc_layer" in adata.obs.columns else "ground_truth"
        measured["ground_truth"] = adata.obs[label_key].astype(str).str.replace(" ", "_", regex=False)
    finally:
        adata.file.close()

    result, macros, per_layer = [], [], []
    for mode, letter in zip(MODES,"bc"):
        for method in METHODS:
            table = predictions.loc[predictions["mode"].eq(mode) & predictions.method.eq(method)].set_index(
                "source_spot_id",verify_integrity=True)
            assert len(table)==3611 and set(table.index)==set(measured.index)
            np.testing.assert_array_equal(table.ground_truth, measured.loc[table.index,"ground_truth"])
            np.testing.assert_array_equal(table[["x","y"]], measured.loc[table.index,["x","y"]])
            counts = pd.crosstab(table.ground_truth, table.predicted_layer).reindex(index=LAYERS,columns=LAYERS,fill_value=0)
            actual = counts.sum(axis=1).to_numpy()
            predicted = counts.sum(axis=0).to_numpy()
            np.testing.assert_array_equal(actual,[273,253,989,218,673,692,513])
            tp = np.diag(counts.to_numpy())
            f1 = 2*tp/(actual+predicted)
            for layer,n,tp_value,fp,fn,score in zip(LAYERS,actual,tp,predicted-tp,actual-tp,f1):
                per_layer.append({"panel_id":letter,"mode":mode,"method":method,"layer":layer,
                                  "n_true":int(n),"tp":int(tp_value),"fp":int(fp),"fn":int(fn),"f1":float(score)})
                for call in LAYERS:
                    result.append({"panel_id":letter,"mode":mode,"method":method,"true_layer":layer,
                                   "predicted_layer":call,"count":int(counts.loc[layer,call]),
                                   "row_fraction":float(counts.loc[layer,call]/n),"n_true":int(n)})
            stored_score = stored.loc[stored["mode"].eq(mode) & stored.method.eq(method),"macro_f1"].item()
            np.testing.assert_allclose(f1.mean(),stored_score,rtol=0,atol=1e-12)
            macros.append({"panel_id":letter,"mode":mode,"method":method,"n_pairs":len(table),"macro_f1":float(f1.mean())})
    layers = pd.DataFrame(per_layer)
    identifiers = ["mode","method","layer"]
    np.testing.assert_allclose(layers.set_index(identifiers)[["tp","fp","fn","f1"]].sort_index(),
                               stored_layers.set_index(identifiers)[["tp","fp","fn","f1"]].sort_index(),
                               rtol=0,atol=1e-12)
    return pd.DataFrame(result), layers, pd.DataFrame(macros), predictions


def prepare(zenodo):
    root, inputs = (zenodo / folder / STUDY for folder in ("results","inputs"))
    result = root / "figure_s3"
    result.mkdir(parents=True, exist_ok=True)
    deltas = sensitivity(root)
    matrices, layer_scores, summary, predictions = confusion(root, inputs)
    assert (len(matrices),len(layer_scores),len(summary)) == (392,56,8)
    history = root / "comparison/figure_s3_article"
    old = pd.read_csv(history / "Fig_S3_sensitivity.csv")
    key = ["mode","alpha","support_variant","metric"]
    comparison = deltas.merge(old[key+["pca10_minus_pca20"]], on=key, how="outer", validate="one_to_one")
    comparison["delta_vs_article"] = comparison.difference-comparison.pca10_minus_pca20
    old_f1 = pd.read_csv(history / "Fig_S3_macro_f1.csv")[["mode","method","macro_f1"]]
    f1_compare = summary.merge(old_f1, on=["mode","method"],suffixes=("_current","_article"),validate="one_to_one")
    for name,frame in (("sensitivity", deltas), ("confusion", matrices), ("layer_scores", layer_scores),
                       ("macro_f1", summary), ("sensitivity_comparison",comparison), ("macro_f1_comparison",f1_compare)):
        frame.to_csv(result / f"{name}.csv",index=False)
    predictions.to_csv(result / "endpoint_predictions.csv.gz",index=False)
    restricted = deltas.loc[deltas.support_variant.ne(ALL_SPOTS),["mode","alpha","excluded_pair_count"]].drop_duplicates()
    report = {"prediction_source":"30-neighbour vote on the staged Study 04A endpoint representations",
              "diagonal_affine_predictions_match_figure_4e":True,
              "prediction_rows":28888, "confusion_cells":392, "f1_values":8, "per_layer_scores":56,
              "full_support_comparisons":12,"plotted_differences":len(deltas),
              "restricted_support":restricted.to_dict("records"),"maximum_absolute_delta_vs_article":float(comparison.delta_vs_article.abs().max()),
              "dip_signs":{key:dict(negative=int(g.difference.lt(0).sum()),positive=int(g.difference.gt(0).sum()))
                           for key,g in deltas.groupby("metric")},
              "maximum_absolute_macro_f1_delta":float((f1_compare.macro_f1_current-f1_compare.macro_f1_article).abs().max())}
    (result / "provenance.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo",type=Path,default=Path(__file__).resolve().parents[2] / "Zenodo")
    prepare(p.parse_args().zenodo)
