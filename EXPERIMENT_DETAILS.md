# UCI HAR 采样频率聚类初步实验

本项目检查：同一批活动信号的采样频率变化后，聚类质量及分组一致性是否变化。聚类对象是每条 2.56 秒信号窗口。第一轮只用 `total_acc_x`（总加速度 x 轴，单位 g），不是 561 维人工特征，也不是完整受试者记录。

## 文件与运行

- `run_experiment.py`：自动定位数据、抽样、滤波与重采样、运行四组方法、保存评价和逐样本预测。
- `distances.py`：Numba 编译的 DTW 与带真实秒级时间戳的 TWED。
- `ctds.py`：最近中心 Voronoi 映射、窗口分布统计、区域归一化和拼接。
- `check_algorithms.py`：与独立动态规划参考实现核对距离，并检查时间戳、重采样与分区。
- `results/pilot_300`：正式初步实验。`results/smoke`：仅用于流程核验，不作主要结论。
- `EXPERIMENT_REPORT.md`：由实际结果生成的实验报告。

在 PowerShell 中运行（路径中的反斜线与正斜线均可）：

```powershell
Set-Location <REPO>
& ..\.venv\Scripts\python.exe check_algorithms.py
& ..\.venv\Scripts\python.exe run_experiment.py --per-class 50 --seeds 2026 2027 2028 --output results/new_run
```

也可以运行 `./run_pilot.ps1`。默认结果目录不为空时程序主动停止，避免覆盖旧实验。创建新输出目录即可重跑；不要删除已有结果来隐式覆盖。

依赖是项目外层的独立环境 `<VENV>`，固定核心版本见 `requirements.txt`。迁移机器时，使用 Python 3.12 执行 `python -m venv .venv`，随后以该环境的 Python 执行 `-m pip install -r har_pilot/requirements.txt`。完整安装版本记录于 `environment_freeze.txt`。

## 数据与抽样

输入默认是 `<DATA_DIR>` 下解压的 UCI HAR Dataset，可用 `--data-dir` 修改。程序读取 TRAIN、TEST 的窗口信号、标签与受试者编号，合并作为传导式聚类池。每个种子从每类无放回抽取 50 条，共 300 条，六类分别为 walking、walking upstairs、walking downstairs、sitting、standing、laying。K 预先设为 6。

三个种子同时改变抽样、混合频率分配以及 CTDS/K-means 随机性；因此报告的是三个初步运行的描述性均值与标准差，不是仅 K-means 初始化方差，不作统计显著性结论。同一种子下所有方法、采样条件使用相同样本且保持相同顺序。

标签仅用于平衡抽样及最终评价，受试者编号用于溯源；两者不进入距离或表示，不进行按标签择优调参。原始窗口有 50% 重叠，随机抽到的样本可能相关；这不是受试者隔离的泛化实验，不能解释为独立样本测试。

## 采样条件

1. `50hz`：原始 128 点，无额外滤波。
2. `50hz_lowpass`：仍为 128 点，只做低通，是滤波对照。
3. `25hz`：低通后降为 64 点。
4. `12p5hz`：低通后降为 32 点。
5. `mixed`：每条样本独立、等概率选取 50/25/12.5 Hz；三种频率全部来自相同低通后的信号，避免频率与滤波状态混杂。每条源样本只保留一个版本，频率分配不查看类别。

低通为 6 阶 Butterworth，截止 5 Hz，使用双向 `sosfiltfilt`；这是探索性离线实验，不是因果在线滤波。降采样及上采样使用 `scipy.signal.resample_poly(..., padtype='line')`，包含 FIR 抗混叠/插值滤波。所有设置共同保留约 5 Hz 内信息；即使如此，不能保证该频带足以保留所有类别信息。128 点短窗口的滤波边界效应不可忽略。

时间戳定义为 `arange(n)/rate`，均为相对秒；名义支持区间为 `[0, 2.56)`。最后一个采样时刻分别为 2.54、2.52、2.48 秒，不把最后时刻强行拉伸到相同值。`common50` 分支用多相重采样恢复为 128 点的 50 Hz 网格；插值不能恢复丢失的高频，末端取值来自插值/边界延拓。此处统一的是实际采样间隔，不是把任意时长的序列缩放到同一长度。

## 四组方法

| 方法名 | 输入与参数 | 检验目的 |
| --- | --- | --- |
| dtw_native_hac | 各自采样频率；完整无约束 DTW；局部平方差，累计后开平方；平均连接层次聚类 | 不显式使用时间戳的弹性匹配基线 |
| dtw_common50_hac | 统一回 50 Hz 后的 DTW；其余同上 | 简单重采样是否足够 |
| twed_native_hac | 原频率信号与秒级时间戳；p=1、nu=0.1、lambda=0.01；平均连接层次聚类 | 利用时间间隔是否有帮助 |
| ctds_common50_kmeans | 统一回 50 Hz；两区，每区 64 点；滑窗 32 点、步长1；psi=128、分区50次；KMeans n_init=10 | 局部分布方法对重采样的表现 |

DTW 和 TWED 使用平均连接而非 Ward；距离不作额外路径长度归一化。TWED 参数单位依赖于幅值（g）与秒，未按真实标签搜索，不能将此设置视为 TWED 最优成绩。DTW 与 TWED 本身是距离，并非原论文中的完整层次聚类算法。

CTDS 保留公开代码的最近中心分配、逐窗口 z-score、频率均值嵌入、每区 L2 归一化与拼接；两区宽度对应 1.28 秒，形状窗口对应 0.64 秒。这里是以物理尺度指定区域的 CTDS 适配版本，不是声称原论文对 HAR 的默认最优参数复现。随机映射来自所有待聚类窗口，为传导式评估。DTW/TWED 不逐序列标准化，保留原始偏置和幅值；CTDS 逐窗口标准化可能丢失静态姿态的判别信息，属于方法差异，应明确报告。

## 指标与结果文件

- ARI、NMI：与活动标签一致性，越大越好。
- stability_vs_50hz：与同种子、同方法原始 50 Hz 分组的 ARI。衡量分组稳定性，不等价于类别准确性。
- stability_vs_filtered50：与同方法滤波对照分组的 ARI，更适合区分采样密度与低通影响。
- algorithm_seconds：特征/距离、聚类及距离矩阵写盘的实测秒数；不含数据加载、共享预处理及 Numba 首次编译。共享预处理时间另列，非独占端到端性能排名。
- `metrics.csv`：每次运行；`summary.csv`：三次均值与总体标准差 ddof=0。
- `assignments.csv`：全部预测、真实类别及实际频率。
- `seed_*/samples.csv`：抽中的原文件、行号、受试者编号及全局索引。
- `seed_*/*.npz`：原始及各采样版本，包含时间戳、有效长度、实际频率与统一网格。
- `seed_*/*_distances.npy`：两两距离矩阵，便于更换聚类后端。
- `config.json`：配置、输入和源代码 SHA256、软件版本。`completion.json` 确认整轮完成。

不测峰值内存；不包含 VISTA-SSM、深度模型和自然观测时长变化。第一轮目的为发现问题及检查可行性，不作为完整基准或“解决非平稳/变长问题”的证据。

## 后续建议

先比较 `50hz` 与 `50hz_lowpass` 判断额外低通影响，再比较滤波对照与低频/混合条件判断采样差异。后续增加其他通道、多变量、更多种子与受试者隔离方案；自然时长差异及非平稳阶段变化须另设实验。可增加“跨方法统一预处理”消融，但不能用额外归一化修改后的结果冒充各原算法。

## 来源

- UCI HAR：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Anguita et al., A Public Domain Dataset for Human Activity Recognition using Smartphones, ESANN 2013；使用数据时引用原数据论文。
- DTW：https://doi.org/10.1109/TASSP.1978.1163055
- TWED：https://doi.org/10.1109/TPAMI.2008.76
- CTDS：https://doi.org/10.1007/978-981-97-2242-6_21
- CTDS 作者代码：https://github.com/LeisureGong/CTDS
