# 行业 ETF 周度排序实验

## 2024 年开发结果

相同四个特征下，直接排序的平均周度 Rank IC 为 **0.0696**，比 `return_5` 简单基准高 **0.1255**，比同样按 Rank IC 选参的 XGBoost 回归高 **0.0518**。排序器的 Top 3 重合率为 **26.80%**，比简单基准高 **1.31 个百分点**，比重新选参的回归低 **5.23 个百分点**。这段开发窗口中，排序器对 11 只 ETF 的整体排序更接近实际顺序，而回归器选中的前三与实际前三重合更多。

| 方法 | 最大深度 / 树数 | 平均周度 Rank IC | 平均 Top 3 重合率 | 有效 IC 周 / 总周数 |
|---|---:|---:|---:|---:|
| `return_5` 降序 | — | -0.0560 | 25.49% | 51 / 51 |
| XGBoost 回归，按 Rank IC 选参 | 2 / 100 | 0.0178 | 32.03% | 51 / 51 |
| XGBRanker，按 Rank IC 选参 | 2 / 50 | 0.0696 | 26.80% | 51 / 51 |
| 原 XGBoost 回归，按 RMSE 选参（历史参照） | 2 / 50 | 0.0125 | 29.41% | 51 / 51 |

四个方法均无常数分数周。两个学习模型各比较四组参数。排序器候选的平均 Rank IC 为 0.0038 至 0.0696，回归器为 -0.0057 至 0.0178，完整结果保存在 [grid.csv](01_objective_comparison/run_01/grid.csv)。2024 年同时用于选参和评估，以上差值是开发窗口内的描述性结果。下一步按时间推进拟合和评价窗口，每次训练只纳入在该次预测前已退出的标签。2025 年已查看，本次只沿用其原分区做输入边界核验，未新增该年的模型评分。

## 复现

在项目根目录用现有 Python 环境运行，给每次复现一个新名称。

```bash
.venv/bin/python experiments/sector_ranking/00_input_audit/run.py --run-name run_02
.venv/bin/python experiments/sector_ranking/01_objective_comparison/run.py --run-name run_02
```

- [DESIGN.md](01_objective_comparison/DESIGN.md) 记录运行前确定的比较规则。
- [experiment.json](01_objective_comparison/run_01/experiment.json) 记录数据及代码指纹、时间窗口、特征、参数网格、选中参数和种子。本地 XGBoost 为 3.4.1，排序器使用 `rank:pairwise`，以 `group` 传入按信号日期连续排列的 153 个训练组。
- [scores.csv](01_objective_comparison/run_01/scores.csv) 保存 2024 年 561 个 ETF–周样本及四种分数。`RankingScore` 是排序器分数，`RegressionScore` 和 `HistoricalRMSEScore` 是回归收益预测。`Relevance` 是事后实际收益等级。
- [weekly_metrics.csv](01_objective_comparison/run_01/weekly_metrics.csv) 保存逐周 IC、不可计算原因及 Top 3 名单与重合率。[summary.csv](01_objective_comparison/run_01/summary.csv) 和 [deltas.csv](01_objective_comparison/run_01/deltas.csv) 保存汇总及排序器相对各方法的差值。
- [audit.json](00_input_audit/run_01/audit.json) 和 [excluded_weeks.csv](00_input_audit/run_01/excluded_weeks.csv) 保存输入核验和排除周记录。

## 核验

从日行情复算了保留样本的四个特征及开盘到开盘收益，并核对美国交易日历的信号、入场和退出时间。每周均有 11 只 ETF，2021–2023 年拟合 153 周，2024 年开发比较 51 周。训练标签最迟在 2023 年 12 月 26 日开盘实现，早于 2024 年 1 月 5 日收盘的首个开发信号。原始划分共排除三组跨阶段标签，以及一周特征不足和一周持有期不完整的样本。

从导出的分数独立复算每周指标，检查所有方法使用相同样本，并用构造样本核验同收益同等级、并列时平均名次及代码升序、常数分数周的空 IC。原项目 10 个测试通过，比较运行前后原数据、样本、模型代码、结果、图表、报告及消融目录内的 261 个文件指纹一致。
