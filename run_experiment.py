"""Controlled UCI HAR sampling-rate clustering pilot; see README.md."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from distances import pairwise
from ctds import embedding

VARIANTS = ['50hz', '50hz_lowpass', '25hz', '12p5hz', 'mixed']
METHODS = ['dtw_native_hac', 'dtw_common50_hac', 'twed_native_hac', 'ctds_common50_kmeans']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_root(path):
    candidates = ([path] if (path / 'activity_labels.txt').exists() else [])
    candidates += [p.parent for p in path.rglob('activity_labels.txt')]
    candidates = sorted(set(p for p in candidates if (p / 'train/y_train.txt').exists()))
    if len(candidates) != 1:
        raise ValueError('Expected one extracted UCI HAR root, found: ' + str(candidates))
    return candidates[0]


def load_data(root, channel):
    xx, yy, subjects, meta, input_hashes = [], [], [], [], {}
    for split in ['train', 'test']:
        signal = root / split / 'Inertial Signals' / (channel + '_' + split + '.txt')
        label = root / split / ('y_' + split + '.txt')
        subject = root / split / ('subject_' + split + '.txt')
        x = np.loadtxt(signal)
        y = np.loadtxt(label, dtype=int)
        s = np.loadtxt(subject, dtype=int)
        if x.shape != (len(y), 128) or len(y) != len(s) or not np.isfinite(x).all():
            raise ValueError('Invalid signal array.')
        xx.append(x); yy.append(y); subjects.append(s)
        meta += [{'split': split, 'source_row_1based': i + 1} for i in range(len(y))]
        for p in [signal, label, subject]:
            input_hashes[str(p)] = sha(p)
    return np.concatenate(xx), np.concatenate(yy), np.concatenate(subjects), meta, input_hashes


def select_samples(x, y, subjects, meta, count, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for label in sorted(np.unique(y)):
        pool = np.flatnonzero(y == label)
        if len(pool) < count:
            raise ValueError('Not enough observations per class.')
        indices.extend(rng.choice(pool, count, replace=False).tolist())
    indices = np.array(indices)[rng.permutation(len(indices))]
    rows = [dict(sample_id=i, global_index=int(j), activity=int(y[j]),
                 subject=int(subjects[j]), **meta[j]) for i, j in enumerate(indices)]
    return x[indices], y[indices], rows


def make_variant(original, variant, seed):
    # Identical physical bandwidth for 25/12.5 Hz observations and the filter-only control.
    filtered = sosfiltfilt(butter(6, 5.0, fs=50.0, output='sos'), original, axis=1)
    rng = np.random.default_rng(seed)
    factors = {'50hz': np.ones(len(original), dtype=int),
               '50hz_lowpass': np.ones(len(original), dtype=int),
               '25hz': np.full(len(original), 2),
               '12p5hz': np.full(len(original), 4),
               'mixed': rng.choice([1, 2, 4], len(original))}[variant]
    source = original if variant == '50hz' else filtered
    values, timestamps, common = [], [], []
    for i, factor in enumerate(factors):
        low = source[i].copy() if factor == 1 else resample_poly(source[i], 1, int(factor), padtype='line')
        values.append(low)
        timestamps.append(np.arange(len(low)) * factor / 50.0)
        common.append(low.copy() if factor == 1 else resample_poly(low, int(factor), 1, padtype='line')[:128])
    lengths = np.array([len(v) for v in values], dtype=np.int64)
    padded = np.zeros((len(values), 128)); times = np.zeros_like(padded)
    for i, (v, t) in enumerate(zip(values, timestamps)):
        padded[i, :len(v)] = v; times[i, :len(t)] = t
    return padded, times, lengths, np.array(common), 50.0 / factors


def hac(distances, k):
    if not np.isfinite(distances).all() or not np.allclose(distances, distances.T):
        raise ValueError('Invalid distance matrix')
    return cut_tree(linkage(squareform(distances, checks=True), method='average'), n_clusters=[k]).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results/pilot_300')
    p.add_argument('--per-class', type=int, default=50)
    p.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    p.add_argument('--channel', default='total_acc_x')
    p.add_argument('--clusters', type=int, default=6)
    p.add_argument('--psi', type=int, default=128)
    p.add_argument('--repetitions', type=int, default=50)
    p.add_argument('--twed-nu', type=float, default=0.1)
    p.add_argument('--twed-penalty', type=float, default=0.01)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory is not empty. Use a new --output to preserve previous runs.')
    if a.per_class < 2 or a.clusters < 2 or a.twed_nu <= 0 or a.twed_penalty < 0:
        p.error('Invalid count, cluster count, or TWED parameters.')
    a.output.mkdir(parents=True, exist_ok=True)
    root = find_root(a.data_dir)
    all_x, all_y, subjects, meta, input_hashes = load_data(root, a.channel)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(dataset_root=str(root), python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, sklearn=sklearn.__version__,
                  raw_samples=len(all_y), input_sha256=input_hashes,
                  nominal_window_seconds=2.56, original_rate_hz=50,
                  lowpass_hz=5, lowpass_order=6, resampling='scipy.signal.resample_poly padtype=line',
                  normalization='none before distances; per-window z-score within CTDS',
                  source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(a.output / 'config.json', config)
    # Compilation warm-up outside timing, so runtimes measure algorithm work rather than JIT.
    warm = np.array([[0., 1., 2.], [1., 2., 3.]])
    for use_twed in [False, True]:
        pairwise(warm, np.tile(np.arange(3.) / 50, (2, 1)), np.array([3, 3], dtype=np.int64), use_twed)
    rows, assignments = [], []
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            original, labels, selected = select_samples(all_x, all_y, subjects, meta, a.per_class, seed)
            out = a.output / ('seed_' + str(seed)); out.mkdir()
            write_csv(out / 'samples.csv', selected)
            np.savez_compressed(out / 'original.npz', x=original, y=labels)
            baseline_predictions = {}; filtered_predictions = {}
            for variant in VARIANTS:
                prep_started = time.perf_counter()
                values, times, lengths, common, rates = make_variant(original, variant, seed + 10000)
                prep_seconds = time.perf_counter() - prep_started
                np.savez_compressed(out / (variant + '.npz'), values=values, times=times,
                                    lengths=lengths, common50=common, rates=rates, y=labels)
                for method in METHODS:
                    started = time.perf_counter()
                    if method == 'ctds_common50_kmeans':
                        features = embedding(common, seed, a.psi, a.repetitions)
                        predictions = KMeans(n_clusters=a.clusters, random_state=seed, n_init=10).fit_predict(features)
                    else:
                        if method == 'dtw_common50_hac':
                            data = common
                            ts = np.tile(np.arange(128.) / 50.0, (len(common), 1))
                            lens = np.full(len(common), 128, dtype=np.int64)
                        else:
                            data, ts, lens = values, times, lengths
                        matrix = pairwise(data, ts, lens, method == 'twed_native_hac', a.twed_nu, a.twed_penalty)
                        predictions = hac(matrix, a.clusters)
                        np.save(out / (variant + '_' + method + '_distances.npy'), matrix)
                    seconds = time.perf_counter() - started
                    if len(np.unique(predictions)) != a.clusters:
                        raise AssertionError('Unexpected cluster count')
                    if variant == '50hz': baseline_predictions[method] = predictions.copy()
                    if variant == '50hz_lowpass': filtered_predictions[method] = predictions.copy()
                    row = dict(seed=seed, variant=variant, method=method, n=len(labels),
                               ari=adjusted_rand_score(labels, predictions),
                               nmi=normalized_mutual_info_score(labels, predictions),
                               stability_vs_50hz=adjusted_rand_score(baseline_predictions[method], predictions),
                               stability_vs_filtered50=adjusted_rand_score(filtered_predictions[method], predictions)
                               if method in filtered_predictions else None,
                               algorithm_seconds=seconds, shared_preprocessing_seconds=prep_seconds,
                               min_points=int(lengths.min()), max_points=int(lengths.max()))
                    rows.append(row)
                    assignments += [dict(seed=seed, variant=variant, method=method, sample_id=i,
                                         activity=int(labels[i]), cluster=int(predictions[i]), rate_hz=float(rates[i]))
                                    for i in range(len(labels))]
                    print(json.dumps(row), flush=True)
                    write_csv(a.output / 'metrics.csv', rows)
                    write_csv(a.output / 'assignments.csv', assignments)
    summary = []
    for variant in VARIANTS:
        for method in METHODS:
            group = [r for r in rows if r['variant'] == variant and r['method'] == method]
            record = dict(variant=variant, method=method, runs=len(group))
            for key in ['ari', 'nmi', 'stability_vs_50hz', 'stability_vs_filtered50', 'algorithm_seconds']:
                numbers = [r[key] for r in group if r[key] is not None]
                record[key + '_mean'] = float(np.mean(numbers)) if numbers else None
                record[key + '_std'] = float(np.std(numbers)) if numbers else None
            summary.append(record)
    write_csv(a.output / 'summary.csv', summary)
    write_json(a.output / 'completion.json', dict(complete=True, metric_rows=len(rows), seeds=a.seeds))
    print('COMPLETE ' + str(a.output), flush=True)


if __name__ == '__main__':
    main()
