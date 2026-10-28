# 第一轮：诊断已保存的聚类结果

入口：`diagnostics/round1/index.html`，用浏览器打开。程序只读取原来的结果，不重新计算距离、不训练或重新运行聚类。

## 运行

在 PowerShell 中执行（使用原实验的 Python 3.12 环境）：

```powershell
Set-Location D:\cluster\har-sampling-clustering
& D:\cluster\.venv\Scripts\python.exe diagnose_round1.py
```

默认读取 `results/pilot_300/assignments.csv`、`config.json` 和各个种子的 `samples.csv`，在相邻 `data` 目录查找原始 UCI HAR 数据。可用 `--results`、`--data-dir`、`--output` 指定路径。输出目录非空时拒绝覆盖；重复运行请指定新的输出目录。

## 怎么阅读

1. 先看 `largest_cluster_overview.png` 和 `group_summary.csv`，比较各组最大簇占比。大簇提示需要检查合并现象，不能仅凭占比判定算法错误。80% 标记只是诊断提示，不是统计阈值或优化目标。
2. 进入 `seed_种子/采样条件/方法/`，看 `class_cluster_counts.csv` 和对应热力图。行是真实活动，列是原始簇编号；每行合计应为50。观察哪些活动混进同一个簇、某个活动是否分散到多个簇。
3. 看 `cluster_sizes.png` 和 `cluster_sizes.csv`。柱高是样本数，标签注明占比。均匀分配的参考线只帮助阅读，不表示必须让簇大小相等。
4. 看同一采样条件下 `signals/` 中的信号图，比较均值、幅值和波动。每类固定选样本编号最小的3个；同一种子的不同条件使用相同样本。均值虚线及图中 mean/std 方便比较，各小图纵轴独立，需同时阅读刻度和数值。

## 信号图的含义

- `native`：原采样条件下的输入，供原频率 DTW、TWED 参考。
- `common50`：重采样至50Hz的输入，供统一50Hz DTW、CTDS参考；CTDS还会进行后续局部标准化与表示变换，图中不是最终特征。
- 同一条件的方法共享输入信号图，避免重复生成完全相同的图片。横轴为秒，信号为 total_acc_x（总加速度x轴，包含重力），单位为g。
- 分享版没有保存信号数组，因此依据保存的原始行号、活动、受试者和采样率，从原数据重建滤波及重采样步骤。检查原始数据哈希，不重新抽样，不改变预测。
- 本实验的低采样率由同一原始窗口合成，窗口名义时长2.56秒；不等于已完成自然变长、自然非平稳数据的验证。

## 汇总文件

- `class_destinations.csv`：每类活动的主要去向，不能把该占比直接称为分类准确率。
- `signal_sample_statistics.csv`：所有输入的均值、标准差、RMS、极差等。
- `signal_class_statistics.csv`：按种子、采样条件、输入视图和活动汇总的信号统计量。
- `selected_signal_examples.csv`：图中样本编号，便于追溯。
- `provenance.json`、`validation.json`：输入路径、哈希和一致性校验。

**簇0不代表走路，其他编号同样没有类别语义。不同种子、条件、方法的相同簇编号也不表示同一类。** 本轮保留原始簇编号，不做类别对齐。真实活动标签只用于事后解释结果。

## 主要函数

`class_cluster_counts` 生成计数；`diagnose_group` 生成单组表和图；`rebuild_observations` 重建信号输入；`plot_examples` 绘制活动示例；`make_overview` 与 `write_report` 汇总诊断。主函数组织全部60组。
