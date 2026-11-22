#!/usr/bin/env python3
"""Current-run Figure 4A–C: regional redistribution and sparse-edge support."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import read_elem
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from ccc_current_io import AGES, DATA, REGIONS, LIMITATIONS, ccc_path, input_root, run_info

def regional_sum(group, codes, chunk_rows):
    """Read only CSR row blocks; accumulate 8x8 totals and support diagnostics."""
    n = len(codes)
    assert tuple(group.attrs["shape"]) == (n, n)
    w = np.zeros((8, 8)); counts = np.zeros((8, 8), dtype=np.int64)
    self_by_region = np.zeros(8)
    sender = np.zeros(n, dtype=bool); receiver = np.zeros(n, dtype=bool)
    sum_squares = 0.; max_weight = 0.; total_direct = 0.; stored_zeros = 0
    for lo in range(0, n if len(group["data"]) else 0, chunk_rows):
        hi = min(n, lo+chunk_rows)
        ptr = group["indptr"][lo:hi+1]
        first, last = int(ptr[0]), int(ptr[-1])
        columns = group["indices"][first:last]
        values = group["data"][first:last]
        assert np.isfinite(values).all() and (values >= 0).all()
        stored_zeros += int(np.count_nonzero(values == 0))
        rows = np.repeat(np.arange(lo, hi, dtype=np.int32), np.diff(ptr))
        diagonal = rows == columns
        self_by_region += np.bincount(codes[rows[diagonal]], weights=values[diagonal], minlength=8)
        positive = (values > 0) & ~diagonal
        rows, columns, values = rows[positive], columns[positive], values[positive]
        keys = 8*codes[rows]+codes[columns]
        w += np.bincount(keys, weights=values, minlength=64).reshape(8, 8)
        counts += np.bincount(keys, minlength=64).reshape(8, 8)
        sender[rows] = True; receiver[columns] = True
        total_direct += float(values.sum())
        sum_squares += float(np.dot(values, values))
        if len(values): max_weight = max(max_weight, float(values.max()))
    total = float(w.sum())
    np.testing.assert_allclose(total, total_direct, rtol=1e-12, atol=1e-10)
    support = {"total_nonself_mass": total, "self_mass_excluded": float(self_by_region.sum()),
               "n_nonself_edges": int(counts.sum()), "n_sender_positions": int(sender.sum()),
               "n_receiver_positions": int(receiver.sum()), "n_participating_positions": int((sender|receiver).sum()),
               "effective_weighted_edges": total*total/sum_squares if total else 0.,
               "max_edge_fraction": max_weight/total if total else 0., "stored_zero_entries": stored_zeros}
    return w, counts, self_by_region, support


def prepare(raw_root, output, chunk_rows, verified_root=None):
    statuses = {age: pd.read_csv(Path(raw_root)/f'{age}_3D/all_LR_status.csv').set_index('Key') for age in AGES}
    keys = sorted(set(statuses[AGES[0]].index) & set(statuses[AGES[1]].index))
    shared = pd.DataFrame([{'ligand': key.split('--')[0], 'receptor': key.split('--')[1],
                           'E15.5_nonzero': int(statuses[AGES[0]].loc[key,'Status']),
                           'E18.5_nonzero': int(statuses[AGES[1]].loc[key,'Status'])} for key in keys])
    assert len(shared) == 85
    selected = shared[(shared['E15.5_nonzero']==1) & (shared['E18.5_nonzero']==1)]
    assert len(selected) == 45 and shared['E15.5_nonzero'].eq(shared['E18.5_nonzero']).all()
    pairs = list(selected[['ligand','receptor']].itertuples(index=False, name=None))
    profiles, support, sources = {}, {}, {}
    edges, volumes, validations = [], [], []
    for age in AGES:
        stored = pd.read_csv(Path(raw_root)/f'{age}_3D/LR_region_edges.csv')
        if verified_root is not None:
            raw_edges = pd.read_csv(verified_root / f'{age}_3D/regional_edges.csv')
            raw_status = pd.read_csv(verified_root / f'{age}_3D/sparse_support.csv').set_index('Key')
            verified = {}
            for (ligand, receptor), data in raw_edges.groupby(['ligand', 'receptor']):
                matrices = [data.pivot(index='sender_region', columns='receiver_region', values=field)
                            .loc[list(REGIONS), list(REGIONS)].to_numpy() for field in ('W', 'n_edges')]
                verified[(ligand, receptor)] = (*matrices, raw_status.loc[f'{ligand}--{receptor}']
                                               .drop(['Status', 'maximum_region_mass_error']).to_dict())
        with h5py.File(ccc_path(raw_root,age),'r') as h:
            info = run_info(h)
            categories = read_elem(h['obs/region'])
            codes = pd.Categorical(categories,categories=REGIONS).codes.astype(np.int16)
            assert (codes>=0).all()
            sizes = np.bincount(codes,minlength=8)
            for i,region in enumerate(REGIONS):
                volumes.append({'stage':age,'region':region,'n_positions':int(sizes[i]),'n_total':len(codes),'volume_fraction':sizes[i]/len(codes)})
            sources[age]={'path':str(ccc_path(raw_root,age)),'configuration_id':info['preflight']['source_configuration_id'],
                          'feast_version':'1.0.6','n_lr_candidates':85,'distance_threshold_mm':.035,'normalization_target_sum':100,
                          'reference_artifact_ids':list(h['uns/source_generation_provenance/reference_artifacts'])}
            for number,(ligand,receptor) in enumerate(pairs,1):
                key=f'{ligand}--{receptor}'
                if verified_root is None:
                    w,counts,_,record=regional_sum(h[f'obsp/commot-User_db-{ligand}-{receptor}'],codes,chunk_rows)
                else:
                    w, counts, record = verified[(ligand, receptor)]
                expected=stored[(stored.ligand==ligand)&(stored.receptor==receptor)]
                ew=expected.pivot(index='sender_region',columns='receiver_region',values='W').loc[list(REGIONS),list(REGIONS)].to_numpy()
                en=expected.pivot(index='sender_region',columns='receiver_region',values='n_edges').loc[list(REGIONS),list(REGIONS)].to_numpy()
                np.testing.assert_allclose(w,ew,rtol=1e-11,atol=1e-9);np.testing.assert_array_equal(counts,en)
                total=record['total_nonself_mass']
                assert total > 0
                m=(w.sum(axis=1)+w.sum(axis=0))/(2*total)
                np.testing.assert_allclose(m.sum(),1,atol=1e-12)
                f_inter=float(w[~np.eye(8,dtype=bool)].sum()/total)
                profiles[(key,age)],support[(key,age)]=m,{**record,'F_inter':f_inter}
                validations.append({'stage':age,'LR':key,'M_sum':float(m.sum()),'max_W_error_vs_completed_run':float(np.max(np.abs(w-ew))),'edge_counts_match':True,'passed':True})
                for i,a in enumerate(REGIONS):
                    for j,b in enumerate(REGIONS):
                        edges.append({'stage':age,'ligand':ligand,'receptor':receptor,'sender_region':a,'receiver_region':b,'W':w[i,j],'n_edges':int(counts[i,j])})
                print(f'{age} {number}/45 {key}: regional sums and edge support verified',flush=True)
    metrics,region_profiles=[],[]
    for ligand,receptor in pairs:
        key=f'{ligand}--{receptor}';a,b=[profiles[(key,age)] for age in AGES]
        js=float(jensenshannon(a,b,base=2)**2)
        midpoint=(a+b)/2
        direct=float((np.sum(rel_entr(a,midpoint))+np.sum(rel_entr(b,midpoint)))/(2*np.log(2)))
        np.testing.assert_allclose(js,direct,rtol=1e-11,atol=1e-14)
        record={'ligand':ligand,'receptor':receptor,'LR':key,'JSD_base2':js,
                'F_inter_early':support[(key,AGES[0])]['F_inter'],'F_inter_late':support[(key,AGES[1])]['F_inter'],
                'delta_F_inter_pp':100*(support[(key,AGES[1])]['F_inter']-support[(key,AGES[0])]['F_inter'])}
        for label,age in zip(['early','late'],AGES):record.update({f'{label}_{k}':v for k,v in support[(key,age)].items()})
        metrics.append(record)
        np.testing.assert_allclose((b-a).sum(),0,atol=1e-12)
        for i,r in enumerate(REGIONS):region_profiles.append({'ligand':ligand,'receptor':receptor,'LR':key,'region':r,'M_early':a[i],'M_late':b[i],'delta_M_pp':100*(b[i]-a[i])})
    metrics=pd.DataFrame(metrics).sort_values(['JSD_base2','LR'],ascending=[False,True]).reset_index(drop=True)
    metrics.insert(0,'JSD_rank',np.arange(1,46))
    v=pd.DataFrame(volumes).pivot(index='region',columns='stage',values='volume_fraction').loc[list(REGIONS)]
    delta_v=pd.DataFrame({'region':REGIONS,'V_early':v[AGES[0]].to_numpy(),'V_late':v[AGES[1]].to_numpy(),'delta_volume_fraction_pp':100*(v[AGES[1]]-v[AGES[0]]).to_numpy()})
    output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(edges).to_parquet(output/'LR_region_edges.parquet',index=False,compression='zstd')
    pd.DataFrame(region_profiles).to_csv(output/'LR_region_profiles.csv',index=False)
    metrics.to_csv(output/'LR_redistribution_metrics.csv',index=False)
    shared.to_csv(output/'shared_LR_status.csv',index=False)
    pd.DataFrame(volumes).to_csv(output/'region_volume_fractions.csv',index=False)
    delta_v.to_csv(output/'region_volume_fraction_change.csv',index=False)
    pd.DataFrame(validations).to_csv(output/'aggregation_validation.csv',index=False)
    info={'created_utc':datetime.now(timezone.utc).isoformat(),'entry_point':str(Path(__file__).resolve()),'sources':sources,'region_order':REGIONS,
          'W':'Regional sum over i!=j; source CSR blocks, verified against completed-run regional CSV.',
          'M':'(row mass + column mass)/(2*total nonself mass)',
          'JSD':'Base-2 Jensen-Shannon divergence: scipy jensenshannon squared; independently checked with relative entropy.',
          'row_order':'Descending JSD, alphabetical LR tie break; no clustering or support filtering.',
          'support':'Actual positive nonself edges, unique senders/receivers/union, weighted effective edges, maximum edge fraction.',
          'limitations':LIMITATIONS+['M is a relative composition, not absolute activity. Voxel-volume fractions do not adjust for eligible spatial pairs or contact area.',
                                     'LR pairs sharing components are not independent biological replicates. No regression, significance test or biological-program claim.',
                                     'Panel D and representative-case selection remain deferred.']}
    (output/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Figure 4 data complete:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=input_root())
    p.add_argument('--output-dir',type=Path,default=DATA/'redistribution')
    p.add_argument('--chunk-rows',type=int,default=250000)
    p.add_argument('--verified-root', type=Path)
    a=p.parse_args();prepare(a.input_root,a.output_dir,a.chunk_rows,a.verified_root)
