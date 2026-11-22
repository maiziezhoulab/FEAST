#!/usr/bin/env python3
"""Extract measured coordinates and counts from the two observed male sections."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd

from ccc_current_io import DATA, STUDY

GENES = ("Wnt7a", "Fzd2", "Fzd8", "Lrp5", "Lrp6")
SAMPLES = (("GSM8323035_E14M_1", "E14", "E15.5"),
           ("GSM8323030_E18M_1", "E18", "E18.5"))


def prepare(reference_root, output):
    output.mkdir(parents=True, exist_ok=True)
    sources, summaries, article_rna, article_joint = [], [], [], []
    for sample, stage, target in SAMPLES:
        path = reference_root / f"{sample}.h5ad"
        with h5py.File(path, "r") as h:
            region = read_elem(h["obs/region"])
            names = read_elem(h["var"]).index.tolist()
            xy = h["obsm/spatial"][:]
            n = len(region)
            assert xy.shape == (n, 2) and np.isfinite(xy).all()
            assert set(np.asarray(read_elem(h["obs/sex"])).astype(str)) == {"M"}
            assert set(np.asarray(read_elem(h["obs/developmental_stage"])).astype(str)) == {stage}
            assert set(np.asarray(read_elem(h["obs/section_id"])).astype(str)) == {sample}
            counts = read_elem(h["layers/counts"])
            assert counts.shape == (n, len(names))
            expression = counts[:, [names.index(gene) for gene in GENES]].toarray()
            totals = np.asarray(counts.sum(axis=1)).ravel()
            normalized = np.log1p(np.divide(100*expression, totals[:,None],
                                            out=np.zeros_like(expression,dtype=float),where=totals[:,None]>0))
            assert np.isfinite(expression).all() and (expression >= 0).all()
            index_key = h["obs"].attrs["_index"]
            ids = h[f"obs/{index_key}"].asstr()[:]
            assert len(set(ids)) == n
            label_source = sorted(set(np.asarray(read_elem(h["obs/label_source"])).astype(str)))
        for sender,receiver,receptor in [("IZ","DorsalVZ","Fzd2_Lrp5"),("GE","VentralVZ","Fzd8_Lrp6")]:
            fzd,lrp=receptor.split("_")
            lr=f"Wnt7a--{receptor}"
            for role,label,gene in [("sender",sender,"Wnt7a"),("receiver",receiver,fzd),("receiver",receiver,lrp)]:
                take=np.asarray(region)==label
                article_rna.append({"LR":lr,"sample":sample,"source_stage":stage,"role":role,"region":label,
                                    "gene":gene,"mean_log1p_expression":float(normalized[take,GENES.index(gene)].mean()),
                                    "n_measured_cells":int(take.sum())})
            take=np.asarray(region)==receiver
            joint=(expression[take,GENES.index(fzd)]>0)&(expression[take,GENES.index(lrp)]>0)
            article_joint.append({"LR":lr,"sample":sample,"source_stage":stage,"receiver_region":receiver,
                                  "receptor":receptor,"n_measured_cells":int(take.sum()),
                                  "n_joint_positive":int(joint.sum()),"joint_positive_pct":100*float(joint.mean())})
        frame = pd.DataFrame({"cell_id": ids, "x_um": xy[:, 0], "y_um": xy[:, 1], "region": region})
        for index, gene in enumerate(GENES):
            frame[gene] = expression[:, index]
            summaries.append({"sample": sample, "source_stage": stage, "gene": gene,
                              "n_cells": n, "n_positive_cells": int((expression[:, index] > 0).sum()),
                              "total_observed_counts": float(expression[:, index].sum()),
                              "maximum_observed_count": float(expression[:, index].max())})
        frame.to_parquet(output / f"{sample}_observed.parquet", index=False, compression="zstd")
        sources.append({"sample_id": sample, "source_stage": stage, "atlas_target_stage": target,
                        "path": str(path.resolve()), "sex": "M", "n_cells": n,
                        "n_other_cells": int(frame.region.eq("Other").sum()),
                        "n_non_other_cells": int(frame.region.ne("Other").sum()),
                        "label_source": label_source,
                        "region_counts": {str(k): int(v) for k, v in frame.region.value_counts().items()}})
        print(f"Extracted observed {sample}: {n:,} cells; original coordinates and measured counts retained", flush=True)
    pd.DataFrame(summaries).to_csv(output / "gene_support.csv", index=False)
    pd.DataFrame(article_rna).to_csv(output / "measured_RNA_source_data.csv",index=False)
    pd.DataFrame(article_joint).to_csv(output / "measured_receptor_codetection.csv",index=False)
    (output / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "entry_point": str(Path(__file__).resolve()),
        "sources": sources, "genes": GENES, "male_only": True,
        "selection": "The two existing observed-2D CCC comparison samples, both members of the male reconstruction reference pools; one illustrative section per stage, not selected by expression.",
        "coordinates": "Original obsm/spatial cell centroids, in micrometers.",
        "counts": "Unmodified measured layers/counts for the five genes in the two main Wnt7a cases.",
        "regions": "Existing recovered marker-based region labels; no new annotation or cell-type assignment.",
        "filtering": "None: all available observed cells, including Other, are retained. No QC, expression or regional filtering is added.",
        "context": "FEAST used five eligible male references per age; these two are illustrative measured inputs and the observed-2D control samples.",
        "limitations": ["These are measured cell-centroid/count maps, not microscopy or histology images.",
                        "The two sections contributed to the reconstruction and are not held-out or independent validation.",
                        "E14/E18 source stages differ from E15.5/E18.5 atlas targets; cross-age section homology is not established.",
                        "Raw-count expression maps are descriptive input illustrations, without library-size normalization, differential testing or signaling-activity inference."],
        "new_inference": False}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=STUDY.parents[1] / "Datasets/Processed/GSE269617/h5ad_region_annotated")
    parser.add_argument("--output-dir", type=Path, default=DATA / "observed_st")
    args = parser.parse_args()
    prepare(args.reference_root, args.output_dir)
