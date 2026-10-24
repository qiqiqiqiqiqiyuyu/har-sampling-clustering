"""Render a report from completed result files, without modifying experiment outputs."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NAMES = {'dtw_native_hac':'DTW native + HAC', 'dtw_common50_hac':'DTW common 50Hz + HAC',
         'twed_native_hac':'TWED native + HAC', 'ctds_common50_kmeans':'CTDS common 50Hz + KMeans'}
VARIANTS = ['50hz','50hz_lowpass','25hz','12p5hz','mixed']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results/pilot_300')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    completion=json.loads((a.results/'completion.json').read_text(encoding='utf-8'))
    if not completion['complete']: raise ValueError('Incomplete experiment')
    config=json.loads((a.results/'config.json').read_text(encoding='utf-8'))
    with (a.results/'summary.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    x=np.arange(len(VARIANTS))
    for method,label in NAMES.items():
        data=[next(r for r in rows if r['method']==method and r['variant']==v) for v in VARIANTS]
        for ax,key,title in zip(axes,['ari','nmi','stability_vs_filtered50'],
                                ['Clustering ARI','Clustering NMI','Stability vs filtered 50 Hz (ARI)']):
            yy=[float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in data]
            err=[float(r[key+'_std']) if r[key+'_std'] else np.nan for r in data]
            ax.errorbar(x,yy,yerr=err,marker='o',capsize=3,label=label)
            ax.set_title(title); ax.set_xticks(x,['50 raw','50 LPF','25','12.5','Mixed'])
            ax.set_xlabel('Sampling rate (Hz)'); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/'metrics.png',dpi=170); plt.close(fig)
    data=np.load(a.results/('seed_'+str(config['seeds'][0]))/'original.npz')
    low=np.load(a.results/('seed_'+str(config['seeds'][0]))/'12p5hz.npz')
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
    activity=['Walking','Walking upstairs','Walking downstairs','Sitting','Standing','Laying']
    for k,ax in enumerate(axes.flat,1):
        idx=int(np.flatnonzero(data['y']==k)[0]); ax.plot(np.arange(128)/50,data['x'][idx],label='50 Hz original')
        ax.plot(low['times'][idx,:32],low['values'][idx,:32],'.-',label='12.5 Hz filtered',alpha=.8)
        ax.set_title(activity[k-1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Acceleration (g)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/'signal_examples.png',dpi=160); plt.close(fig)
    lines=['# UCI HAR 采样频率聚类初步实验报告','',
           '## 实际完成范围','',
           f"本轮使用 total_acc_x 单通道，每类 {config['per_class']} 条，共 {6*config['per_class']} 条；"
           f"种子为 {config['seeds']}。五种采样条件、四组方法，共完成 {completion['metric_rows']} 次评估。"
           '所有结果来自已保存的实际输出，以下数值为运行间均值 ± 总体标准差。','',
           '采用六个已知活动类别设置 K=6。每个种子重新平衡抽样；同种子内各方法使用完全相同的对象。'
           '距离和特征构造不使用真实标签，未根据 ARI/NMI 选择参数。','',
           '## 聚类质量','',
           '| 采样条件 | 方法 | ARI | NMI | 计算秒数 |','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['variant']+' | '+NAMES[r['method']]+' | '+
                     ' | '.join(f"{float(r[k+'_mean']):.4f} ± {float(r[k+'_std']):.4f}" for k in ['ari','nmi','algorithm_seconds'])+' |')
    lines += ['','## 采样变化下的分组稳定性','',
              '这里用同一方法在同一批对象上的两组预测计算 ARI。高稳定性不意味着与活动标签一致；'
              '错误或过于粗糙的分组也可能稳定。应与上表一起阅读。','',
              '| 采样条件 | 方法 | 对原始50Hz的稳定性 | 对滤波50Hz的稳定性 |','|---|---|---:|---:|']
    for r in rows:
        stability='—' if not r['stability_vs_filtered50_mean'] else f"{float(r['stability_vs_filtered50_mean']):.4f}"
        lines.append(f"| {r['variant']} | {NAMES[r['method']]} | {float(r['stability_vs_50hz_mean']):.4f} | {stability} |")
    lines += ['','## 本轮观察','']
    for method,label in NAMES.items():
        lookup={r['variant']:r for r in rows if r['method']==method}
        raw=float(lookup['50hz']['ari_mean']); filtered=float(lookup['50hz_lowpass']['ari_mean']); mixed=float(lookup['mixed']['ari_mean'])
        lines += [f'- {label}：原始 50Hz ARI={raw:.4f}，仅低通后为 {filtered:.4f}，混合频率为 {mixed:.4f}；'
                  f'混合频率相对滤波对照的 ARI 变化为 {mixed-filtered:+.4f}。']
    best=max((r for r in rows if r['variant']=='mixed'),key=lambda r:float(r['ari_mean']))
    lines += ['',f"本轮混合频率条件下平均 ARI 最高的是 {NAMES[best['method']]}（{float(best['ari_mean']):.4f}）。"
              '这是本通道、参数及三个样本集上的描述性结果，不能推断为一般优劣或统计显著差异。','',
              '## 簇大小诊断','',
              '平均连接层次聚类可能保留少量离群样本为小簇，而把大部分样本归入一个簇。下面检查混合频率条件下的最大簇占比；占比过高时，低ARI可能主要体现簇大小失衡。','',
              '| 方法 | 最大簇占比均值 | 与采样频率分组的ARI均值 |','|---|---:|---:|']
    from sklearn.metrics import adjusted_rand_score
    with (a.results/'assignments.csv').open(encoding='utf-8-sig') as f: assignments=list(csv.DictReader(f))
    for method,label in NAMES.items():
        fractions=[]; rate_aris=[]
        for seed in config['seeds']:
            group=[r for r in assignments if r['variant']=='mixed' and r['method']==method and int(r['seed'])==seed]
            predicted=np.array([int(r['cluster']) for r in group])
            rates=np.array([float(r['rate_hz']) for r in group])
            fractions.append(np.unique(predicted,return_counts=True)[1].max()/len(predicted))
            rate_categories=np.unique(rates,return_inverse=True)[1]
            rate_aris.append(adjusted_rand_score(rate_categories,predicted))
        lines += [f'| {label} | {np.mean(fractions):.2%} | {np.mean(rate_aris):.4f} |']
    lines += ['',
              '## 解释边界','',
              '- 原始50Hz与低通50Hz的区别包含丢弃高频及滤波边界效应；低通50Hz与其他低频条件的比较更接近采样密度实验。',
              '- 数据原本是128点短窗口；本轮没有检验自然观测时长变化，也没有证明对任意非平稳过程有效。',
              '- total_acc_x 的幅值与偏置可包含姿态信息。CTDS 逐窗口 z-score 与距离基线保留原值的差异可能影响静态活动，不能简单将差距都归因于采样处理。',
              '- 原始HAR窗口重叠50%，本轮不是受试者隔离的独立泛化评估。标签只用于平衡抽样与评价；不能宣称完全无标签抽样。',
              '- TWED 时间戳以秒计，nu=0.1、lambda=0.01，未调优；带时间戳不等于自动具有采样频率不变性。',
              '- DTW/TWED均采用完整动态规划，无路径长度归一化；HAC为平均连接。CTDS采用两区64点、窗口32点、psi=128、50次随机分区、KMeans n_init=10。',
              '- 运行时间不含Numba首次编译、数据加载和共享预处理；包含距离矩阵保存，不作为端到端性能排名。未测量峰值内存。','',
              '## 文件与复现','',
              f'- 完成标记：`{a.results.name}/completion.json`。',
              '- 原始逐次指标、标准差及预测：结果目录的 metrics.csv、summary.csv、assignments.csv。',
              '- 每个样本的原文件行号、受试者、时间戳、采样率及距离矩阵均已保存，可逐条追溯。',
              '- 输入与源代码SHA256、软件版本、参数见 config.json；完整依赖见 environment_freeze.txt。',
              '- 运行命令与方法细节见 README.md；新运行应指定新的 --output。','',
              '## 下一步实验','',
              '增加其他通道及多变量；固定抽样进一步改变混合频率种子；按受试者隔离评价；'
              '扩展Character Trajectories自然变长数据；增加VISTA-SSM。优先增加有区分力的实验，不根据本轮真实标签反复筛选最优设置。','',
              '## 原始论文与数据','',
              '- UCI HAR：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones',
              '- DTW：https://doi.org/10.1109/TASSP.1978.1163055',
              '- TWED：https://doi.org/10.1109/TPAMI.2008.76',
              '- CTDS：https://doi.org/10.1007/978-981-97-2242-6_21',
              '- CTDS作者代码：https://github.com/LeisureGong/CTDS','']
    (root/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Standalone HTML: figures embedded, so the report can be moved independently.
    import base64
    rendered=[]; in_table=False
    for line in lines:
        if line.startswith('|'):
            if set(line.replace('|','').replace(':','').replace('-','').strip())==set(): continue
            if not in_table: rendered.append('<table>'); in_table=True
            rendered.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
        else:
            if in_table: rendered.append('</table>'); in_table=False
            if line.startswith('# '): rendered.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '): rendered.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line: rendered.append('<p>'+html.escape(line)+'</p>')
    images=''.join('<img style="width:100%" src="data:image/png;base64,'+base64.b64encode((figures/f).read_bytes()).decode()+'">' for f in ['metrics.png','signal_examples.png'])
    page='<html lang="zh-CN"><meta charset="utf-8"><title>UCI HAR 初步实验报告</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;line-height:1.7;color:#17202a}table{border-collapse:collapse;width:100%;font-size:13px}td{border:1px solid #d9d9d9;padding:7px}tr:first-child{background:#eef2f5;font-weight:bold}h1,h2{color:black}</style>'+images+''.join(rendered)+'</html>'
    (root/'EXPERIMENT_REPORT.html').write_text(page,encoding='utf-8')
    print('Reports and figures saved.')


if __name__=='__main__': main()
