"""Controlled round-two ablations on the exact saved pilot objects."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import argparse, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diagnose_round1 as d
from distances import pairwise
from ctds import embedding


def transform(x, mode):
    if mode=='raw': return x.copy(),0
    centered=x-x.mean(axis=1,keepdims=True)
    std=x.std(axis=1,keepdims=True)
    if mode=='center': return centered,0
    # A numerical safeguard, not a tuned denoising threshold. Count near-flat inputs.
    return centered/np.maximum(std,1e-8),int((std<1e-8).sum())


def statistics(x):
    return np.column_stack([x.mean(1),x.std(1),np.sqrt(np.mean(x*x,axis=1)),np.ptp(x,axis=1)])


def hac(matrix):
    assert np.isfinite(matrix).all() and (matrix>=0).all()
    np.testing.assert_allclose(matrix,matrix.T)
    np.testing.assert_array_equal(np.diag(matrix),0)
    return cut_tree(linkage(squareform(matrix),method='average'),n_clusters=[6]).ravel()


def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=base/'results/pilot_300')
    p.add_argument('--data-dir',type=Path,default=base.parent/'data')
    p.add_argument('--output',type=Path,default=base/'results/round2')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a new empty output directory')
    cfg=json.loads((a.results/'config.json').read_text(encoding='utf-8'))
    assignments=d.read_csv(a.results/'assignments.csv')
    original_hash=d.sha(a.results/'assignments.csv')
    arrays,hashes=d.read_originals(d.locate_data(a.data_dir),cfg)
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(seeds=cfg['seeds'],source_results=str(a.results),input_sha256=hashes,
                source_assignment_sha256=original_hash,psi=cfg['psi'],repetitions=cfg['repetitions'],
                preprocessing_modes=['raw','center','zscore'],std_floor=1e-8,
                fusion_weights=[.25,.5,1.0],fusion='L2-normalized CTDS + weight * L2-normalized standardized stats',
                statistics=['mean','std_population','rms','peak_to_peak'],n_init=10,clusters=6,
                selection='All predeclared configurations evaluated; labels used for development diagnostics only',
                twed_grid=[[.1,.01],[.01,.01],[1.,.01],[.1,0.],[.1,.1]],
                source_hashes={name:d.sha(base/name) for name in ['run_round2.py','ctds.py','distances.py','diagnose_round1.py']})
    (a.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    metrics=[]; predictions=[]; numerics=[]; matrix_stats=[]; agreement=[]
    baseline_predictions={}; block_features={}
    old_lookup=defaultdict(list)
    for r in assignments: old_lookup[(int(r['seed']),r['variant'],r['method'])].append(r)
    def record(seed,variant,method,y,pred,rates,seconds,family,mode='',weight=''):
        sizes=np.bincount(pred); sizes=sizes[sizes>0]
        assert len(sizes)==6
        key=(seed,method)
        if variant=='50hz_lowpass': baseline_predictions[key]=pred.copy()
        stability=ari(baseline_predictions[key],pred) if key in baseline_predictions else ''
        metrics.append(dict(seed=seed,variant=variant,method=method,family=family,mode=mode,weight=weight,
                            ari=float(ari(y,pred)),nmi=float(nmi(y,pred)),
                            stability_vs_filtered50=stability,largest_cluster_fraction=float(sizes.max()/len(y)),
                            smallest_cluster_count=int(sizes.min()),singleton_clusters=int((sizes==1).sum()),seconds=seconds))
        predictions.extend(dict(seed=seed,variant=variant,method=method,sample_id=i,activity=int(y[i]),cluster=int(pred[i]),rate_hz=float(rates[i])) for i in range(len(y)))
        folder=a.output/f'seed_{seed}'/variant/method
        d.diagnose_group([dict(sample_id=i,activity=int(y[i]),cluster=int(pred[i])) for i in range(len(y))],folder,seed,variant,method,100)
        d.write_csv(a.output/'metrics.csv',metrics)
        d.write_csv(a.output/'assignments.csv',predictions)
    with threadpool_limits(limits=1):
        # Filter-only is the reference; raw 50Hz is evaluated last, avoiding causal conflation.
        for seed in cfg['seeds']:
            rows=d.read_csv(a.results/f'seed_{seed}/samples.csv')
            original,y=d.select_saved_samples(arrays,rows)
            for variant in ['50hz_lowpass','25hz','12p5hz','mixed','50hz']:
                ref=sorted(old_lookup[(seed,variant,'dtw_native_hac')],key=lambda r:int(r['sample_id']))
                np.testing.assert_array_equal([int(r['activity']) for r in ref],y)
                rates=np.array([float(r['rate_hz']) for r in ref])
                native,ts,common=d.rebuild_observations(original,rates,variant,cfg)
                time_grid=np.tile(np.arange(128)/50,(len(y),1)); lengths=np.full(len(y),128,dtype=np.int64)
                stats=statistics(common)
                # Scaling is fitted without labels, within this transductive clustering dataset.
                scaled=StandardScaler().fit_transform(stats)
                scaled_unit=normalize(scaled)
                features={}
                for mode in ['raw','center','zscore']:
                    x,nearflat=transform(common,mode)
                    numerics.append(dict(seed=seed,variant=variant,mode=mode,min_std=float(common.std(1).min()),
                                         below_1e_8=nearflat,below_1e_3=int((common.std(1)<1e-3).sum())))
                    start=time.perf_counter(); matrix=pairwise(x,time_grid,lengths)
                    pred=hac(matrix)
                    name='dtw_'+mode
                    np.save(a.output/f'distance_{seed}_{variant}_{name}.npy',matrix)
                    record(seed,variant,name,y,pred,rates,time.perf_counter()-start,'dtw',mode)
                    start=time.perf_counter(); features[mode]=embedding(x,seed,cfg['psi'],cfg['repetitions'])
                    pred=KMeans(n_clusters=6,random_state=seed,n_init=10).fit_predict(features[mode])
                    record(seed,variant,'ctds_'+mode,y,pred,rates,time.perf_counter()-start,'ctds',mode)
                    if mode=='raw':
                        for new,old in [('dtw_raw','dtw_common50_hac'),('ctds_raw','ctds_common50_kmeans')]:
                            previous=sorted(old_lookup[(seed,variant,old)],key=lambda r:int(r['sample_id']))
                            current=[r['cluster'] for r in predictions if r['seed']==seed and r['variant']==variant and r['method']==new]
                            score=ari([int(r['cluster']) for r in previous],current)
                            agreement.append(dict(seed=seed,variant=variant,method=new,ari_vs_saved=float(score)))
                            assert score==1.,'Baseline reproduction failed'
                for mode in ['center','zscore']:
                    delta=features['raw']-features[mode]
                    block_features[(seed,variant,mode)]=dict(seed=seed,variant=variant,comparison='raw_vs_'+mode,
                        max_abs_feature_difference=float(np.abs(delta).max()),relative_frobenius=float(np.linalg.norm(delta)/np.linalg.norm(features['raw'])))
                start=time.perf_counter()
                pred=KMeans(n_clusters=6,random_state=seed,n_init=10).fit_predict(scaled)
                record(seed,variant,'stats',y,pred,rates,time.perf_counter()-start,'stats')
                for weight in [.25,.5,1.]:
                    start=time.perf_counter()
                    joined=np.column_stack([normalize(features['raw']),weight*scaled_unit])
                    pred=KMeans(n_clusters=6,random_state=seed,n_init=10).fit_predict(joined)
                    record(seed,variant,'fusion_'+str(weight).replace('.','p'),y,pred,rates,time.perf_counter()-start,'fusion',weight=weight)
                # Separate one-factor-at-a-time TWED audit, only filter-only 50Hz.
                if variant=='50hz_lowpass':
                    for nu,penalty in config['twed_grid']:
                        start=time.perf_counter(); matrix=pairwise(common,time_grid,lengths,True,nu,penalty)
                        values=matrix[np.triu_indices(len(y),1)]
                        matrix_stats.append(dict(seed=seed,nu=nu,penalty=penalty,
                            **{'q'+str(q):float(np.percentile(values,q)) for q in [0,25,50,75,95,99,100]}))
                        name=f'twed_nu{nu}_p{penalty}'.replace('.','p')
                        np.save(a.output/f'distance_{seed}_{name}.npy',matrix)
                        record(seed,variant,name,y,hac(matrix),rates,time.perf_counter()-start,'twed')
                        if nu==.1 and penalty==.01:
                            previous=sorted(old_lookup[(seed,variant,'twed_native_hac')],key=lambda r:int(r['sample_id']))
                            score=ari([int(r['cluster']) for r in previous],hac(matrix)); assert score==1.
                            agreement.append(dict(seed=seed,variant=variant,method=name,ari_vs_saved=float(score)))
                print(f'Completed seed={seed}, variant={variant}, total runs={len(metrics)}',flush=True)
    summaries=[]
    for variant,method in dict.fromkeys((r['variant'],r['method']) for r in metrics):
        group=[r for r in metrics if r['variant']==variant and r['method']==method]
        row=dict(variant=variant,method=method,runs=len(group))
        for key in ['ari','nmi','largest_cluster_fraction','stability_vs_filtered50']:
            nums=[r[key] for r in group if r[key]!='']
            row[key+'_mean']=float(np.mean(nums)) if nums else ''
            row[key+'_std']=float(np.std(nums)) if nums else ''
        summaries.append(row)
    for name,rows in [('summary',summaries),('normalization_checks',numerics),('ctds_feature_invariance',block_features.values()),('twed_distance_quantiles',matrix_stats),('baseline_reproduction',agreement)]:
        d.write_csv(a.output/(name+'.csv'),rows)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    filtered=[r for r in summaries if r['variant']=='50hz_lowpass']
    for ax,key in zip(axes,['ari','nmi']):
        ax.barh([r['method'] for r in filtered],[r[key+'_mean'] for r in filtered],xerr=[r[key+'_std'] for r in filtered])
        ax.set_xlabel(key.upper()+' mean +/- population std (3 seeds)'); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(a.output/'filtered50_comparison.png',dpi=140); plt.close(fig)
    report=['# 第二轮实验结果','', '所有结果均为开发集诊断，未做独立受试者验证。全部配置事先列出并评估；不将标签选出的最优结果作为无偏测试性能。',
        '使用原样本、标签、频率分配。raw表示保留滤波后信号的绝对水平，不表示未滤波。统计特征在当前聚类样本集无监督缩放。融合表示两块先分别L2归一化，再给统计块权重。',
        '每次实验含6簇；三个种子不是独立测试集，样本可能重叠。标准差为三个种子的总体标准差。',
        '## 50Hz低通对照','| 方法 | ARI均值±标准差 | NMI均值 | 最大簇占比 |','|---|---:|---:|---:|']
    for r in filtered: report.append(f"|{r['method']}|{r['ari_mean']:.4f} ± {r['ari_std']:.4f}|{r['nmi_mean']:.4f}|{r['largest_cluster_fraction_mean']:.1%}|")
    report+=['','## 各采样条件','|条件|方法|ARI|NMI|相对低通50Hz稳定性ARI|','|---|---|---:|---:|---:|']
    for r in summaries:
        if r['method'] in ['dtw_raw','ctds_raw','stats','fusion_0p25','fusion_0p5','fusion_1p0']:
            report.append(f"|{r['variant']}|{r['method']}|{r['ari_mean']:.4f}|{r['nmi_mean']:.4f}|{r['stability_vs_filtered50_mean']:.4f}|")
    report+=['','## 文件说明','summary.csv汇总；metrics.csv逐次指标；assignments.csv逐样本预测；seed_*/条件/方法/含计数表及簇大小图。',
             'ctds_feature_invariance.csv检查输入标准化前后的CTDS特征差异；normalization_checks.csv记录小标准差；twed_distance_quantiles.csv记录TWED非对角距离分位数。',
             'baseline_reproduction.csv核对重跑基线与原预测（允许编号置换）；distance_*.npy只对应文件名指定输入与参数，不跨配置复用。',
             'TWED网格仅改变nu或penalty其中一项，固定时间单位为秒、平均连接；不能从有限网格推断该方法的普遍上限。']
    (a.output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    import html
    (a.output/'index.html').write_text('<meta charset="utf-8"><title>第二轮实验</title><h1>第二轮实验</h1><img style="max-width:100%" src="filtered50_comparison.png"><pre style="white-space:pre-wrap">'+html.escape('\n'.join(report))+'</pre>',encoding='utf-8')
    assert d.sha(a.results/'assignments.csv')==original_hash
    assert len(metrics)==165 and len(predictions)==49500 and len(agreement)==33
    validation=dict(complete=True,runs=len(metrics),assignment_rows=len(predictions),baseline_checks=len(agreement),
                    all_baselines_reproduced=True,source_assignments_unchanged=True,independent_holdout=False)
    (a.output/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation),flush=True)


if __name__=='__main__': main()
