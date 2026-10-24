# UCI HAR 采样频率聚类初步实验报告

## 实际完成范围

本轮使用 total_acc_x 单通道，每类 50 条，共 300 条；种子为 [2026, 2027, 2028]。五种采样条件、四组方法，共完成 60 次评估。所有结果来自已保存的实际输出，以下数值为运行间均值 ± 总体标准差。

采用六个已知活动类别设置 K=6。每个种子重新平衡抽样；同种子内各方法使用完全相同的对象。距离和特征构造不使用真实标签，未根据 ARI/NMI 选择参数。

## 聚类质量

| 采样条件 | 方法 | ARI | NMI | 计算秒数 |
|---|---|---:|---:|---:|
| 50hz | DTW native + HAC | 0.4527 ± 0.0116 | 0.6906 ± 0.0143 | 1.8601 ± 0.0115 |
| 50hz | DTW common 50Hz + HAC | 0.4527 ± 0.0116 | 0.6906 ± 0.0143 | 1.8804 ± 0.0286 |
| 50hz | TWED native + HAC | 0.0005 ± 0.0002 | 0.0341 ± 0.0030 | 2.8696 ± 0.0184 |
| 50hz | CTDS common 50Hz + KMeans | 0.3236 ± 0.0147 | 0.5114 ± 0.0292 | 1.8832 ± 0.0258 |
| 50hz_lowpass | DTW native + HAC | 0.4521 ± 0.0205 | 0.6929 ± 0.0103 | 1.8524 ± 0.0072 |
| 50hz_lowpass | DTW common 50Hz + HAC | 0.4521 ± 0.0205 | 0.6929 ± 0.0103 | 1.8679 ± 0.0223 |
| 50hz_lowpass | TWED native + HAC | 0.0146 ± 0.0060 | 0.1300 ± 0.0265 | 2.8682 ± 0.0196 |
| 50hz_lowpass | CTDS common 50Hz + KMeans | 0.2668 ± 0.0110 | 0.4248 ± 0.0128 | 1.8919 ± 0.0280 |
| 25hz | DTW native + HAC | 0.4526 ± 0.0215 | 0.6939 ± 0.0111 | 0.5021 ± 0.0015 |
| 25hz | DTW common 50Hz + HAC | 0.4539 ± 0.0207 | 0.6926 ± 0.0090 | 1.8523 ± 0.0042 |
| 25hz | TWED native + HAC | 0.0122 ± 0.0084 | 0.1163 ± 0.0425 | 0.7946 ± 0.0067 |
| 25hz | CTDS common 50Hz + KMeans | 0.2670 ± 0.0212 | 0.4316 ± 0.0236 | 1.9222 ± 0.0223 |
| 12p5hz | DTW native + HAC | 0.2580 ± 0.1447 | 0.5218 ± 0.1237 | 0.1556 ± 0.0010 |
| 12p5hz | DTW common 50Hz + HAC | 0.4522 ± 0.0190 | 0.6938 ± 0.0097 | 1.8658 ± 0.0062 |
| 12p5hz | TWED native + HAC | 0.0279 ± 0.0259 | 0.1524 ± 0.0629 | 0.2360 ± 0.0023 |
| 12p5hz | CTDS common 50Hz + KMeans | 0.2611 ± 0.0136 | 0.4103 ± 0.0230 | 1.8757 ± 0.0335 |
| mixed | DTW native + HAC | 0.3518 ± 0.1307 | 0.6159 ± 0.1113 | 0.6630 ± 0.0170 |
| mixed | DTW common 50Hz + HAC | 0.4526 ± 0.0215 | 0.6939 ± 0.0111 | 1.8535 ± 0.0038 |
| mixed | TWED native + HAC | 0.0055 ± 0.0065 | 0.0717 ± 0.0427 | 1.0315 ± 0.0291 |
| mixed | CTDS common 50Hz + KMeans | 0.2731 ± 0.0115 | 0.4380 ± 0.0083 | 1.8693 ± 0.0329 |

## 采样变化下的分组稳定性

这里用同一方法在同一批对象上的两组预测计算 ARI。高稳定性不意味着与活动标签一致；错误或过于粗糙的分组也可能稳定。应与上表一起阅读。

| 采样条件 | 方法 | 对原始50Hz的稳定性 | 对滤波50Hz的稳定性 |
|---|---|---:|---:|
| 50hz | DTW native + HAC | 1.0000 | — |
| 50hz | DTW common 50Hz + HAC | 1.0000 | — |
| 50hz | TWED native + HAC | 1.0000 | — |
| 50hz | CTDS common 50Hz + KMeans | 1.0000 | — |
| 50hz_lowpass | DTW native + HAC | 0.9422 | 1.0000 |
| 50hz_lowpass | DTW common 50Hz + HAC | 0.9422 | 1.0000 |
| 50hz_lowpass | TWED native + HAC | 0.1986 | 1.0000 |
| 50hz_lowpass | CTDS common 50Hz + KMeans | 0.5593 | 1.0000 |
| 25hz | DTW native + HAC | 0.9316 | 0.9489 |
| 25hz | DTW common 50Hz + HAC | 0.9373 | 0.9673 |
| 25hz | TWED native + HAC | 0.1533 | 0.8109 |
| 25hz | CTDS common 50Hz + KMeans | 0.5213 | 0.8329 |
| 12p5hz | DTW native + HAC | 0.5311 | 0.5745 |
| 12p5hz | DTW common 50Hz + HAC | 0.9291 | 0.9714 |
| 12p5hz | TWED native + HAC | 0.2108 | 0.8085 |
| 12p5hz | CTDS common 50Hz + KMeans | 0.5412 | 0.8685 |
| mixed | DTW native + HAC | 0.7170 | 0.7573 |
| mixed | DTW common 50Hz + HAC | 0.9316 | 0.9835 |
| mixed | TWED native + HAC | 0.1675 | 0.4698 |
| mixed | CTDS common 50Hz + KMeans | 0.5136 | 0.7650 |

## 本轮观察

- DTW native + HAC：原始 50Hz ARI=0.4527，仅低通后为 0.4521，混合频率为 0.3518；混合频率相对滤波对照的 ARI 变化为 -0.1003。
- DTW common 50Hz + HAC：原始 50Hz ARI=0.4527，仅低通后为 0.4521，混合频率为 0.4526；混合频率相对滤波对照的 ARI 变化为 +0.0005。
- TWED native + HAC：原始 50Hz ARI=0.0005，仅低通后为 0.0146，混合频率为 0.0055；混合频率相对滤波对照的 ARI 变化为 -0.0091。
- CTDS common 50Hz + KMeans：原始 50Hz ARI=0.3236，仅低通后为 0.2668，混合频率为 0.2731；混合频率相对滤波对照的 ARI 变化为 +0.0063。

本轮混合频率条件下平均 ARI 最高的是 DTW common 50Hz + HAC（0.4526）。这是本通道、参数及三个样本集上的描述性结果，不能推断为一般优劣或统计显著差异。

## 簇大小诊断

平均连接层次聚类可能保留少量离群样本为小簇，而把大部分样本归入一个簇。下面检查混合频率条件下的最大簇占比；占比过高时，低ARI可能主要体现簇大小失衡。

| 方法 | 最大簇占比均值 | 与采样频率分组的ARI均值 |
|---|---:|---:|
| DTW native + HAC | 56.78% | -0.0021 |
| DTW common 50Hz + HAC | 47.00% | -0.0016 |
| TWED native + HAC | 96.22% | 0.0004 |
| CTDS common 50Hz + KMeans | 26.89% | -0.0029 |

## 解释边界

- 原始50Hz与低通50Hz的区别包含丢弃高频及滤波边界效应；低通50Hz与其他低频条件的比较更接近采样密度实验。
- 数据原本是128点短窗口；本轮没有检验自然观测时长变化，也没有证明对任意非平稳过程有效。
- total_acc_x 的幅值与偏置可包含姿态信息。CTDS 逐窗口 z-score 与距离基线保留原值的差异可能影响静态活动，不能简单将差距都归因于采样处理。
- 原始HAR窗口重叠50%，本轮不是受试者隔离的独立泛化评估。标签只用于平衡抽样与评价；不能宣称完全无标签抽样。
- TWED 时间戳以秒计，nu=0.1、lambda=0.01，未调优；带时间戳不等于自动具有采样频率不变性。
- DTW/TWED均采用完整动态规划，无路径长度归一化；HAC为平均连接。CTDS采用两区64点、窗口32点、psi=128、50次随机分区、KMeans n_init=10。
- 运行时间不含Numba首次编译、数据加载和共享预处理；包含距离矩阵保存，不作为端到端性能排名。未测量峰值内存。

## 文件与复现

- 完成标记：`pilot_300/completion.json`。
- 原始逐次指标、标准差及预测：结果目录的 metrics.csv、summary.csv、assignments.csv。
- 每个样本的原文件行号、受试者、时间戳、采样率及距离矩阵均已保存，可逐条追溯。
- 输入与源代码SHA256、软件版本、参数见 config.json；完整依赖见 environment_freeze.txt。
- 运行命令与方法细节见 README.md；新运行应指定新的 --output。

## 下一步实验

增加其他通道及多变量；固定抽样进一步改变混合频率种子；按受试者隔离评价；扩展Character Trajectories自然变长数据；增加VISTA-SSM。优先增加有区分力的实验，不根据本轮真实标签反复筛选最优设置。

## 原始论文与数据

- UCI HAR：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- DTW：https://doi.org/10.1109/TASSP.1978.1163055
- TWED：https://doi.org/10.1109/TPAMI.2008.76
- CTDS：https://doi.org/10.1007/978-981-97-2242-6_21
- CTDS作者代码：https://github.com/LeisureGong/CTDS
