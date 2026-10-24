"""Recompute saved metrics and check sampling/provenance consistency."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def validate(root):
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with (root/'metrics.csv').open(encoding='utf-8-sig') as f: metrics=list(csv.DictReader(f))
    with (root/'assignments.csv').open(encoding='utf-8-sig') as f: assignments=list(csv.DictReader(f))
    assert len(metrics)==len(config['seeds'])*5*4
    assert len(assignments)==len(metrics)*6*config['per_class']
    for record in metrics:
        group=[r for r in assignments if all(r[k]==record[k] for k in ['seed','variant','method'])]
        y=[int(r['activity']) for r in group]; predicted=[int(r['cluster']) for r in group]
        assert len(np.unique(predicted))==config['clusters']
        assert abs(adjusted_rand_score(y,predicted)-float(record['ari']))<1e-12
        assert abs(normalized_mutual_info_score(y,predicted)-float(record['nmi']))<1e-12
    for seed in config['seeds']:
        path=root/('seed_'+str(seed)); reference=np.load(path/'original.npz')
        for variant in ['50hz','50hz_lowpass','25hz','12p5hz','mixed']:
            d=np.load(path/(variant+'.npz'))
            np.testing.assert_array_equal(d['y'],reference['y'])
            assert d['common50'].shape==(len(reference['y']),128)
            assert set(d['rates']).issubset({50.,25.,12.5})
            for i,length in enumerate(d['lengths']):
                np.testing.assert_allclose(np.diff(d['times'][i,:length]),1/d['rates'][i])
                assert length==int(2.56*d['rates'][i])
                assert np.isfinite(d['values'][i,:length]).all()
        for p in path.glob('*_distances.npy'):
            d=np.load(p); assert np.isfinite(d).all() and (d>=0).all()
            np.testing.assert_allclose(d,d.T); np.testing.assert_array_equal(np.diag(d),0)
        for variant in ['50hz','50hz_lowpass']:
            np.testing.assert_array_equal(np.load(path/(variant+'_dtw_native_hac_distances.npy')),
                                          np.load(path/(variant+'_dtw_common50_hac_distances.npy')))
    checks=dict(passed=True,metric_rows=len(metrics),assignment_rows=len(assignments),
                metric_recomputed=True,sampling_grids_checked=True,distance_matrices_checked=True,
                same_grid_dtw_equality_checked=True)
    (root/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))


if __name__=='__main__':
    import sys
    validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/pilot_300')
