"""Independent checks of round-two predictions, tables, metrics and matrices."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
import diagnose_round1 as d


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/round2')
    a=p.parse_args(); out=a.output
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    source=Path(cfg['source_results'])
    assert d.sha(source/'assignments.csv')==cfg['source_assignment_sha256']
    originals=defaultdict(dict)
    for r in d.read_csv(source/'assignments.csv'):
        originals[(r['seed'],r['variant'])][r['sample_id']]=(r['activity'],float(r['rate_hz']))
    grouped=defaultdict(list)
    for r in d.read_csv(out/'assignments.csv'): grouped[(r['seed'],r['variant'],r['method'])].append(r)
    metrics=d.read_csv(out/'metrics.csv'); assert len(metrics)==len(grouped)==165
    for r in metrics:
        key=(r['seed'],r['variant'],r['method']); rows=grouped[key]
        assert len(rows)==300 and len({x['sample_id'] for x in rows})==300
        for x in rows:
            assert (x['activity'],float(x['rate_hz']))==originals[key[:2]][x['sample_id']]
        y=[int(x['activity']) for x in rows]; pred=[int(x['cluster']) for x in rows]
        assert abs(ari(y,pred)-float(r['ari']))<1e-12
        assert abs(nmi(y,pred)-float(r['nmi']))<1e-12
        sizes=np.unique(pred,return_counts=True)[1]
        assert len(sizes)==6 and abs(sizes.max()/300-float(r['largest_cluster_fraction']))<1e-12
        folder=out/f'seed_{key[0]}'/key[1]/key[2]
        for row in d.read_csv(folder/'class_cluster_counts.csv'):
            for name,value in row.items():
                if name.startswith('cluster_'):
                    assert int(value)==sum(int(x['activity'])==int(row['activity']) and int(x['cluster'])==int(name[8:]) for x in rows)
            assert int(row['total'])==50
        base=grouped[(key[0],'50hz_lowpass',key[2])]
        base=sorted(base,key=lambda x:int(x['sample_id']))
        ordered=sorted(rows,key=lambda x:int(x['sample_id']))
        assert abs(ari([int(x['cluster']) for x in base],[int(x['cluster']) for x in ordered])-float(r['stability_vs_filtered50']))<1e-12
    for row in d.read_csv(out/'summary.csv'):
        group=[r for r in metrics if r['variant']==row['variant'] and r['method']==row['method']]
        assert len(group)==3
        for key in ['ari','nmi','largest_cluster_fraction','stability_vs_filtered50']:
            values=np.array([float(r[key]) for r in group])
            assert abs(values.mean()-float(row[key+'_mean']))<1e-12
            assert abs(values.std()-float(row[key+'_std']))<1e-12
    files=list(out.glob('distance_*.npy')); assert len(files)==60
    for f in files:
        x=np.load(f); assert x.shape==(300,300) and np.isfinite(x).all() and (x>=0).all()
        np.testing.assert_array_equal(x,x.T); np.testing.assert_array_equal(np.diag(x),0)
    baseline=d.read_csv(out/'baseline_reproduction.csv')
    assert len(baseline)==33 and all(float(r['ari_vs_saved'])==1 for r in baseline)
    assert len(list(out.rglob('class_cluster_counts.png')))==165
    assert len(list(out.rglob('cluster_sizes.png')))==165
    result=dict(passed=True,metrics_recomputed=165,contingency_tables_checked=165,
                distance_matrices_checked=60,baseline_reproductions_checked=33,
                source_assignments_unchanged=True,sample_objects_and_rates_unchanged=True,
                summaries_and_stability_recomputed=True)
    (out/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
