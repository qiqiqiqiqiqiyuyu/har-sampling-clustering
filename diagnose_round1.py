"""Round-one diagnostics of SAVED clustering assignments, without model fitting.

No distance, kernel, clustering, or experiment-runner modules are imported.
Native/50Hz-common observations are reconstructed from saved sample row IDs
and saved sampling rates when caches are absent. See ROUND1_DIAGNOSTICS.md.
"""
import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTIVITIES = {1:'Walking',2:'Walking upstairs',3:'Walking downstairs',4:'Sitting',5:'Standing',6:'Laying'}
ZH = {1:'平地走路',2:'上楼',3:'下楼',4:'坐着',5:'站着',6:'躺着'}
METHODS = {'dtw_native_hac':'原频率 DTW＋平均连接',
           'dtw_common50_hac':'统一50Hz DTW＋平均连接',
           'twed_native_hac':'原频率 TWED＋平均连接',
           'ctds_common50_kmeans':'统一50Hz CTDS＋KMeans'}
VARIANTS = ['50hz','50hz_lowpass','25hz','12p5hz','mixed']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows=list(rows)
    if not rows: raise ValueError('Refusing to save an empty table: '+str(path))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in str(value)):
        raise ValueError('Unsafe group name: '+str(value))
    return str(value)


def class_cluster_counts(rows):
    """Rows are true activity, columns are ORIGINAL cluster IDs, no relabeling."""
    classes=sorted({int(r['activity']) for r in rows})
    clusters=sorted({int(r['cluster']) for r in rows})
    counts=np.zeros((len(classes),len(clusters)),dtype=int)
    class_index={v:i for i,v in enumerate(classes)}; cluster_index={v:j for j,v in enumerate(clusters)}
    ids=[int(r['sample_id']) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate sample IDs in assignment group')
    for r in rows: counts[class_index[int(r['activity'])],cluster_index[int(r['cluster'])]]+=1
    return classes,clusters,counts


def diagnose_group(rows, folder, seed, variant, method, dpi):
    folder.mkdir(parents=True)
    classes,clusters,counts=class_cluster_counts(rows)
    sizes=counts.sum(axis=0); totals=counts.sum(axis=1); n=int(counts.sum())
    write_csv(folder/'class_cluster_counts.csv',[
        dict(activity=c,activity_name=ZH.get(c,str(c)),**{'cluster_'+str(k):int(counts[i,j]) for j,k in enumerate(clusters)},total=int(totals[i]))
        for i,c in enumerate(classes)])
    write_csv(folder/'class_cluster_row_percent.csv',[
        dict(activity=c,activity_name=ZH.get(c,str(c)),**{'cluster_'+str(k):float(100*counts[i,j]/totals[i]) for j,k in enumerate(clusters)})
        for i,c in enumerate(classes)])
    write_csv(folder/'cluster_sizes.csv',[
        dict(cluster=k,count=int(sizes[j]),fraction=float(sizes[j]/n),
             dominant_activity=classes[int(counts[:,j].argmax())],
             dominant_activity_fraction=float(counts[:,j].max()/sizes[j]))
        for j,k in enumerate(clusters)])
    fig,ax=plt.subplots(figsize=(8.6,5.1))
    image=ax.imshow(counts,cmap='Blues',vmin=0,vmax=max(1,int(counts.max())),aspect='auto')
    for i in range(len(classes)):
        for j in range(len(clusters)):
            ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=10,
                    color='white' if counts[i,j]>counts.max()*.55 else 'black')
    ax.set_xticks(range(len(clusters)),['Cluster '+str(k) for k in clusters])
    ax.set_yticks(range(len(classes)),[str(c)+' '+ACTIVITIES.get(c,str(c)) for c in classes])
    ax.set_xlabel('Original cluster ID (not a class label)'); ax.set_ylabel('True activity')
    ax.set_title(f'{seed} | {variant} | {method}\nClass-by-cluster counts')
    fig.colorbar(image,ax=ax,label='Number of samples'); fig.tight_layout()
    fig.savefig(folder/'class_cluster_counts.png',dpi=dpi); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.1))
    bars=ax.bar(['Cluster '+str(k) for k in clusters],sizes,color='#4472a8')
    for bar,size in zip(bars,sizes):
        ax.text(bar.get_x()+bar.get_width()/2,size+n*.015,f'{size}\n({size/n:.1%})',ha='center',va='bottom',fontsize=9)
    ax.axhline(n/len(clusters),color='gray',linestyle='--',label='Equal-size reference only')
    ax.set_ylim(0,max(sizes)*1.22+1); ax.set_ylabel('Number of samples')
    ax.set_title(f'{seed} | {variant} | {method}\nCluster sizes; n={n}')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(folder/'cluster_sizes.png',dpi=dpi); plt.close(fig)
    prob=sizes/n
    summary=dict(seed=seed,variant=variant,method=method,n=n,clusters_observed=len(clusters),
                 largest_cluster_id=clusters[int(sizes.argmax())],largest_cluster_count=int(sizes.max()),
                 largest_cluster_fraction=float(sizes.max()/n),smallest_cluster_count=int(sizes.min()),
                 singleton_clusters=int((sizes==1).sum()),effective_clusters=float(np.exp(-np.sum(prob*np.log(prob)))),
                 dominant_cluster_alert=bool(sizes.max()/n>=.8))
    composition=[]
    for i,c in enumerate(classes):
        order=np.argsort(-counts[i],kind='stable')
        composition.append(dict(seed=seed,variant=variant,method=method,activity=c,activity_name=ZH[c],
                                largest_destination_cluster=clusters[int(order[0])],
                                largest_destination_count=int(counts[i,order[0]]),activity_total=int(totals[i])))
    return summary,composition


def locate_data(path):
    roots=sorted(set(p.parent for p in Path(path).rglob('activity_labels.txt') if (p.parent/'train/y_train.txt').is_file()))
    if (Path(path)/'activity_labels.txt').is_file() and (Path(path)/'train/y_train.txt').is_file(): roots=sorted(set(roots+[Path(path)]))
    if len(roots)!=1: raise ValueError('Specify a directory containing exactly one UCI HAR dataset: '+str(roots))
    return roots[0]


def read_originals(data_root, config):
    arrays={}; hashes={}
    for split in ['train','test']:
        signal=data_root/split/'Inertial Signals'/(config['channel']+'_'+split+'.txt')
        label=data_root/split/('y_'+split+'.txt')
        subject=data_root/split/('subject_'+split+'.txt')
        arrays[split]=(np.loadtxt(signal),np.loadtxt(label,dtype=int),np.loadtxt(subject,dtype=int))
        for path in [signal,label,subject]:
            relative=path.relative_to(data_root).as_posix(); actual=sha(path)
            expected=[h for p,h in config.get('input_sha256',{}).items()
                      if p.replace('\\','/')==relative or p.replace('\\','/').endswith('/'+relative)]
            if expected and any(h!=actual for h in expected): raise ValueError('Original input hash mismatch: '+relative)
            hashes[relative]=actual
    return arrays,hashes


def select_saved_samples(arrays, sample_rows):
    sample_rows=sorted(sample_rows,key=lambda r:int(r['sample_id']))
    if [int(r['sample_id']) for r in sample_rows]!=list(range(len(sample_rows))):
        raise ValueError('Expected contiguous saved sample IDs')
    signals=[]; labels=[]
    for r in sample_rows:
        x,y,s=arrays[r['split']]; index=int(r['source_row_1based'])-1
        if int(y[index])!=int(r['activity']) or int(s[index])!=int(r['subject']):
            raise ValueError('Saved sample provenance does not match raw data')
        signals.append(x[index]); labels.append(int(y[index]))
    return np.asarray(signals),np.asarray(labels)


def rebuild_observations(original, rates, variant, config):
    """Replay preprocessing only; saved per-sample rates are authoritative."""
    fs=float(config['original_rate_hz'])
    source=original if variant=='50hz' else sosfiltfilt(
        butter(int(config['lowpass_order']),float(config['lowpass_hz']),fs=fs,output='sos'),original,axis=1)
    native=[]; times=[]; common=[]
    for row,rate in zip(source,rates):
        factor=int(round(fs/rate))
        if factor not in [1,2,4] or not np.isclose(fs/factor,rate): raise ValueError('Unexpected saved sampling rate')
        v=row.copy() if factor==1 else resample_poly(row,1,factor,padtype='line')
        native.append(v); times.append(np.arange(len(v))/rate)
        common.append(v.copy() if factor==1 else resample_poly(v,factor,1,padtype='line')[:len(row)])
    return native,times,np.asarray(common)


def sample_statistics(signals, labels, rates, seed, variant, view):
    return [dict(seed=seed,variant=variant,view=view,sample_id=i,activity=int(labels[i]),
                 source_rate_hz=float(rates[i]),points=len(v),mean=float(np.mean(v)),
                 std=float(np.std(v)),rms=float(np.sqrt(np.mean(np.asarray(v)**2))),
                 peak_to_peak=float(np.ptp(v)),minimum=float(np.min(v)),maximum=float(np.max(v)))
            for i,v in enumerate(signals)]


def plot_examples(signals,times,labels,rates,folder,seed,variant,view,count,dpi):
    """Same lowest sample IDs per class in all conditions; no best-case selection."""
    fig,axes=plt.subplots(6,count,figsize=(4.3*count,12.5),squeeze=False,sharex=True)
    examples=[]
    for row,c in enumerate(ACTIVITIES):
        ids=np.flatnonzero(labels==c)[:count]
        for col,idx in enumerate(ids):
            v=signals[idx]; ax=axes[row,col]
            ax.plot(times[idx],v,color='#2b6c9d',linewidth=1)
            ax.axhline(np.mean(v),color='#d06431',linestyle='--',linewidth=.8)
            ax.set_title(f'{ACTIVITIES[c]} | ID {idx} | {rates[idx]:g} Hz source\nmean={np.mean(v):.4f}, std={np.std(v):.4f}',fontsize=9)
            ax.set_xlim(0,2.56); ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.grid(alpha=.2); ax.set_ylabel('Acceleration (g)',fontsize=8)
            if row==5: ax.set_xlabel('Time (s)')
            examples.append(dict(activity=c,sample_id=int(idx),source_rate_hz=float(rates[idx])))
    fig.suptitle(f'{seed} | {variant} | {view}\nIndependent y-scales; dashed line = sample mean',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(folder/(view+'_examples.png'),dpi=dpi); plt.close(fig)
    return examples


def html_table(rows, columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(r[key]))+'</td>' for key,_ in columns)+'</tr>' for r in rows)+'</tbody></table>'


def make_overview(summary, output, dpi):
    seeds=sorted({r['seed'] for r in summary}); methods=sorted({r['method'] for r in summary})
    fig,axes=plt.subplots(1,len(seeds),figsize=(5.1*len(seeds),4.5),squeeze=False,sharey=True)
    for j,seed in enumerate(seeds):
        matrix=np.array([[next(r['largest_cluster_fraction'] for r in summary if r['seed']==seed and r['method']==m and r['variant']==v)
                          for v in VARIANTS] for m in methods])
        ax=axes[0,j]; im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Oranges',aspect='auto')
        for row in range(len(methods)):
            for col in range(len(VARIANTS)): ax.text(col,row,f'{matrix[row,col]:.0%}',ha='center',va='center',fontsize=9)
        ax.set_xticks(range(5),['50 raw','50 LPF','25','12.5','Mixed']); ax.set_yticks(range(len(methods)),methods,fontsize=8)
        ax.set_title(f'Seed {seed}'); ax.set_xlabel('Sampling condition')
    fig.suptitle('Largest-cluster fraction (diagnostic, not accuracy)')
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(output/'largest_cluster_overview.png',dpi=dpi); plt.close(fig)


def write_report(output,summary,composition,signal_stats,examples,provenance):
    columns=[('seed','种子'),('variant','采样条件'),('method','方法'),('largest','最大簇占比'),
             ('min_size','最小簇样本数'),('singletons','单样本簇数'),('links','详细图表')]
    table=[]
    for r in summary:
        folder=f"seed_{r['seed']}/{r['variant']}/{r['method']}"
        table.append(dict(seed=r['seed'],variant=r['variant'],method=r['method'],largest=f"{r['largest_cluster_fraction']:.1%}",
                          min_size=r['smallest_cluster_count'],singletons=r['singleton_clusters'],links=folder))
    lines=['# 第一轮聚类结果诊断','',
           f"读取已有预测，共诊断 {len(summary)} 组。未调用距离计算、特征生成或聚类算法，也未重新抽样。",'',
           '## 如何阅读','',
           '类别×簇计数表：行是六种真实活动，列是原始预测簇编号；簇0不代表活动1。编号也不能跨种子或方法直接对应。行百分比表按真实类别归一化。',
           '簇大小图：检查是否存在一个大簇和多个小簇；虚线仅为等大参考，不要求真实簇等大。最大簇超过80%的标记是诊断提示，不是算法失败的统计判定。',
           '信号示例：每类按sample_id排序取前三条，在所有采样条件中使用同一批示例，不挑选最容易区分或最差的个例。各子图纵轴独立，请看刻度及均值/标准差，不能只凭曲线高度比较幅值。','',
           'native为各自采样频率下的观测，供原频率DTW/TWED使用；common50为重采样后的输入，供统一频率DTW/CTDS使用。后者的图显示进入方法前的信号，不是CTDS逐窗口标准化后的窗口。相同输入的信号图在不同方法间共享。','',
           '## 混合频率下的簇大小','',
           '| 种子 | 方法 | 最大簇占比 | 最小簇大小 | 单样本簇 |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['variant']=='mixed': lines.append(f"| {r['seed']} | {r['method']} | {r['largest_cluster_fraction']:.1%} | {r['smallest_cluster_count']} | {r['singleton_clusters']} |")
    lines+=['','## 当前观察','']
    for method in sorted({r['method'] for r in summary}):
        group=[r for r in summary if r['variant']=='mixed' and r['method']==method]
        fraction=np.mean([r['largest_cluster_fraction'] for r in group])
        lines.append(f'- {METHODS.get(method,method)}：混合频率最大簇占比平均为 {fraction:.2%}。')
    lines+=['','## 解释边界','',
            '计数表中的活动标签只用于诊断，不改变原预测。类别在一个簇内高度集中不一定说明聚类好：所有类别都进入同一簇时，也会各自集中。必须同时检查该簇包含哪些其他类别。',
            '信号均值和标准差可能反映活动、姿态、受试者或设备佩戴差异。本轮只能指出可能原因，不证明因果关系，也不据此自动选择参数。',
            'total_acc_x为单轴总加速度，单位g；六类标签分别为走路、上楼、下楼、坐着、站着、躺着。完整信号包含重力影响。',
            '## 文件','',
            '- `index.html`：总入口，按种子、采样条件、方法展开查看全部计数表、柱状图及关联的信号示例。',
            '- `group_summary.csv`：60组簇大小及失衡诊断；`class_destinations.csv`：每类进入最多的簇。',
            '- `signal_sample_statistics.csv`：每条信号的均值、标准差、RMS及极差；`signal_class_statistics.csv`：按类别汇总。',
            '- `seed_*/采样条件/方法/`：原始计数表、行百分比表、簇大小CSV及两张图片。',
            '- `seed_*/采样条件/signals/`：native与common50两种输入的各类示例。',
            '- `provenance.json`：预测、样本行号及原始输入哈希；`validation.json`：诊断核验。','',
            '## 重建信号的说明','',
            '从每个种子的samples.csv读取已选中的split和原文件行号，不重新抽样。核对活动标签、受试者编号和原始数据哈希。按照assignments.csv记录的实际频率，重放原有低通与多相重采样；不重新生成频率分配，也不调用run_experiment.py。','']
    (output/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    body=['<h1>第一轮聚类结果诊断</h1>','<p>仅分析已保存的预测；不重新计算距离、特征或聚类。</p>',
          '<p><b>原始簇编号没有类别含义，且不可跨运行直接对应。</b>行是真实类别，列为预测簇。计数表使用真实标签作诊断，不修改预测。</p>',
          '<p>各类信号图采用独立纵轴，请结合刻度、均值与标准差阅读；同一输入在不同方法间共享图表。native对应原频率DTW/TWED，common50对应统一频率DTW/CTDS（标准化前）。</p>',
          '<p><a href="DIAGNOSTIC_REPORT.md">文字说明</a> · <a href="group_summary.csv">全部簇大小诊断CSV</a> · <a href="signal_class_statistics.csv">信号类别统计CSV</a></p>',
          '<img src="largest_cluster_overview.png" alt="Largest cluster overview">']
    body.append(html_table(table,columns))
    for seed in sorted({r['seed'] for r in summary}):
        body.append(f'<h2>种子 {seed}</h2>')
        for variant in VARIANTS:
            body.append(f'<h3>{variant}</h3>')
            for r in [r for r in summary if r['seed']==seed and r['variant']==variant]:
                folder=f"seed_{seed}/{variant}/{r['method']}"; signal_folder=f'seed_{seed}/{variant}/signals'
                view='native' if r['method'] in ['dtw_native_hac','twed_native_hac'] else 'common50'
                body.append(f'<details><summary>{html.escape(METHODS.get(r["method"],r["method"]))} — 最大簇 {r["largest_cluster_fraction"]:.1%}</summary>')
                for filename in ['class_cluster_counts.png','cluster_sizes.png']:
                    body.append(f'<img loading="lazy" src="{folder}/{filename}" alt="{filename}">')
                body.append(f'<p><a href="{folder}/class_cluster_counts.csv">计数表CSV</a> · <a href="{folder}/class_cluster_row_percent.csv">行百分比CSV</a> · <a href="{folder}/cluster_sizes.csv">簇大小CSV</a></p>')
                body.append(f'<p>此方法输入对应的各类信号示例：{view}</p><img loading="lazy" src="{signal_folder}/{view}_examples.png" alt="Signal examples"></details>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第一轮聚类诊断</title><style>body{font:16px/1.65 Arial,"Microsoft YaHei",sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#17202a}img{max-width:100%;height:auto;display:block;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9d9d9;padding:6px;text-align:left}th{background:#edf2f6}summary{cursor:pointer;font-weight:bold;padding:10px}details{border:1px solid #ddd;margin:10px 0;padding:6px}a{color:#145a99}</style>'+''.join(body)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent
    parser.add_argument('--results',type=Path,default=base/'results/pilot_300')
    parser.add_argument('--data-dir',type=Path,default=base.parent/'data')
    parser.add_argument('--output',type=Path,default=base/'diagnostics/round1')
    parser.add_argument('--examples-per-class',type=int,default=3)
    parser.add_argument('--dpi',type=int,default=120)
    args=parser.parse_args()
    if args.examples_per_class<1 or args.dpi<60: parser.error('Invalid example count or DPI')
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output is not empty; choose a new --output')
    config=json.loads((args.results/'config.json').read_text(encoding='utf-8'))
    assignments=read_csv(args.results/'assignments.csv'); groups=defaultdict(list)
    provenance=dict(results=str(args.results.resolve()),clustering_rerun=False,distance_computation_rerun=False,
                    source_assignment_sha256=sha(args.results/'assignments.csv'),config_sha256=sha(args.results/'config.json'),
                    samples_sha256={},signal_source='Original UCI HAR rows; replay preprocessing with saved rates',
                    script_sha256=sha(__file__),examples_rule='Lowest sample IDs per activity; same across conditions',
                    original_cluster_ids_preserved=True)
    for r in assignments:
        if r['variant'] not in VARIANTS or r['method'] not in METHODS: raise ValueError('Unknown method or variant')
        groups[(int(r['seed']),r['variant'],r['method'])].append(r)
    expected={(s,v,m) for s in config['seeds'] for v in VARIANTS for m in METHODS}
    if set(groups)!=expected: raise ValueError('Missing or extra assignment groups')
    roots=locate_data(args.data_dir); arrays,input_hashes=read_originals(roots,config)
    provenance['raw_data_root']=str(roots); provenance['raw_input_sha256']=input_hashes
    args.output.mkdir(parents=True,exist_ok=True)
    summary=[]; composition=[]; stats=[]; example_records=[]
    for seed in config['seeds']:
        sample_file=args.results/f'seed_{seed}'/'samples.csv'
        sample_rows=read_csv(sample_file); provenance['samples_sha256'][str(seed)]=sha(sample_file)
        original,labels=select_saved_samples(arrays,sample_rows)
        if args.examples_per_class>min((labels==c).sum() for c in ACTIVITIES): raise ValueError('Too many requested examples')
        for variant in VARIANTS:
            reference=sorted(groups[(seed,variant,'dtw_native_hac')],key=lambda r:int(r['sample_id']))
            if [int(r['sample_id']) for r in reference]!=list(range(len(labels))): raise ValueError('Assignment/sample IDs disagree')
            np.testing.assert_array_equal([int(r['activity']) for r in reference],labels)
            rates=np.array([float(r['rate_hz']) for r in reference])
            for method in METHODS:
                group=sorted(groups[(seed,variant,method)],key=lambda r:int(r['sample_id']))
                if [(r['sample_id'],r['activity'],r['rate_hz']) for r in group]!=[(r['sample_id'],r['activity'],r['rate_hz']) for r in reference]:
                    raise ValueError('Methods are not using identical objects/rates')
                folder=args.output/f'seed_{seed}'/variant/safe_name(method)
                row,details=diagnose_group(group,folder,seed,variant,method,args.dpi)
                summary.append(row); composition+=details
            native,times,common=rebuild_observations(original,rates,variant,config)
            signal_folder=args.output/f'seed_{seed}'/variant/'signals'; signal_folder.mkdir()
            for view,signals,ts in [('native',native,times),('common50',common,[np.arange(128)/50]*len(common))]:
                current_stats=sample_statistics(signals,labels,rates,seed,variant,view); stats+=current_stats
                examples=plot_examples(signals,ts,labels,rates,signal_folder,seed,variant,view,args.examples_per_class,args.dpi)
                example_records += [dict(seed=seed,variant=variant,view=view,**r) for r in examples]
            print(f'Completed seed={seed}, condition={variant}: 4 groups and 2 shared signal figures',flush=True)
    write_csv(args.output/'group_summary.csv',summary)
    write_csv(args.output/'class_destinations.csv',composition)
    write_csv(args.output/'signal_sample_statistics.csv',stats)
    write_csv(args.output/'selected_signal_examples.csv',example_records)
    class_stats=[]
    for seed in config['seeds']:
        for variant in VARIANTS:
            for view in ['native','common50']:
                for c in ACTIVITIES:
                    group=[r for r in stats if r['seed']==seed and r['variant']==variant and r['view']==view and r['activity']==c]
                    class_stats.append(dict(seed=seed,variant=variant,view=view,activity=c,activity_name=ZH[c],n=len(group),
                                            mean_of_sample_means=float(np.mean([r['mean'] for r in group])),
                                            between_sample_std_of_means=float(np.std([r['mean'] for r in group])),
                                            mean_within_sample_std=float(np.mean([r['std'] for r in group])),
                                            mean_peak_to_peak=float(np.mean([r['peak_to_peak'] for r in group]))))
    write_csv(args.output/'signal_class_statistics.csv',class_stats)
    make_overview(summary,args.output,args.dpi)
    write_report(args.output,summary,composition,stats,example_records,provenance)
    if sha(args.results/'assignments.csv')!=provenance['source_assignment_sha256']: raise AssertionError('Source assignments changed')
    if sha(args.results/'config.json')!=provenance['config_sha256']: raise AssertionError('Source config changed')
    for seed,expected_hash in provenance['samples_sha256'].items():
        if sha(args.results/f'seed_{seed}'/'samples.csv')!=expected_hash: raise AssertionError('Source samples changed')
    validation=dict(passed=True,groups=len(summary),assignments=len(assignments),
                    class_count_total=sum(r['n'] for r in summary),signal_stat_rows=len(stats),
                    class_cluster_plots=len(summary),cluster_size_plots=len(summary),
                    signal_example_plots=len(config['seeds'])*len(VARIANTS)*2,
                    source_assignments_unchanged=True,source_config_unchanged=True,source_samples_unchanged=True,
                    activity_and_subject_provenance_checked=True,original_data_hashes_checked=True,
                    rates_and_objects_equal_across_methods=True,clustering_rerun=False)
    for name,obj in [('provenance.json',provenance),('validation.json',validation)]:
        (args.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    print('Open '+str(args.output/'index.html'),flush=True)


if __name__=='__main__': main()
