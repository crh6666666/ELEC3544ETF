# ELEC3544ETF：独立模型调试项目

本目录为原行业 ETF 项目的独立精简副本（整理日期：2026-11-04）。使用当前已下载的 2021—2025 年数据，分别调试 Ridge、Random Forest 和 XGBoost；无需引用原项目的代码、数据或虚拟环境。

## 从这里开始

Python 3.12。在本目录执行：

```bash
# 当前电脑若已有本目录自己的 .venv，可直接运行，不必再次安装。
.venv/bin/python run.py                    # 三个模型
.venv/bin/python run.py --model ridge      # 仅 Ridge
.venv/bin/python run.py --model random_forest
.venv/bin/python run.py --model xgboost
```

也可直接选择模型模块，便于设置断点：

```bash
.venv/bin/python -m models.ridge
.venv/bin/python -m models.random_forest
.venv/bin/python -m models.xgboost
```

IDE 解释器选择本文件夹 `.venv/bin/python`，工作目录设为本文件夹，运行模块选择 `models.ridge`、`models.random_forest` 或 `models.xgboost`。使用模块方式运行，避免 models/xgboost.py 遮蔽第三方同名包。

在另一台电脑上，先安装 Python 3.12，再执行 `sh setup_env.sh` 安装 requirements.txt 的依赖。依赖安装需要网络；训练和数据检查全程使用本地文件，不会下载行情。

## 文件职责

| 位置 | 用途 |
|---|---|
| models/ridge.py | Ridge、训练集标准化与 alpha 选参 |
| models/random_forest.py | 随机森林及深度/叶子选参 |
| models/xgboost.py | XGBoost 及深度/树数选参 |
| models/common.py | 共用评估、预测元数据和单模型入口 |
| compare_predictions.py | 零收益/训练均值基准、统一比较表和可视化 |
| prepare_data.py | 从本地日行情计算四特征、标签和时间划分 |
| calendar_utils.py | 美国交易日历与每周信号、入场和退出日期 |
| study_config.py | 2021—2023 训练、2024 验证、2025 测试 |
| price_basis.py | 只读核对现有行情、事件和价格单位 |
| plot_portfolio.py | 从已有预测生成每日累计价格收益与回撤模拟图 |
| data/ | 五年行情、来源与核验所需原始响应 |
| prepared_data/ | 本目录生成的周度样本、中间表及指纹 |
| results/ridge/ 等 | 对应模型的预测、指标、参数和选参网格 |
| requirements.txt | 仅模型和数据处理所需的依赖版本 |

本副本没有复制下载/追加数据脚本、原项目的组合优化回测、交易账户、探索诊断、历史备份、旧预测结果或原项目的 Git/IDE 配置。新增的 plot_portfolio.py 仅实现下述固定规则的示意模拟。SPY 不参与训练，用于价格核验和模拟基准。download_report_2021_2025.json 是已下载数据的来源记录，不是下载脚本。

## 数据与实验规则

- 11 只行业 ETF，13,805 条日行情；SPY 1,255 条，覆盖 2021-01-04 至 2025-12-31。
- 四特征仍是 return_5、volatility_5、price_ma_5、activity_5；模型设置沿用原项目，没有重新挑选搜索网格。
- 分区依据信号年份，并清除跨阶段、未实现的边界标签。训练 153 周、验证 51 周、测试 51 周。
- 仅训练集拟合，验证集选参；测试仅评估。反复查看测试结果后的改动属于探索。
- 初次运行自动生成 prepared_data；后续核对数据、代码与样本指纹，过期后重新生成。只准备样本可执行 `.venv/bin/python prepare_data.py --force`。
- 结果为不含现金分红的价格收益。特征用 Close，标签用 Open；Adj Close 只用于核验，禁止重复应用拆分。
- 价格检查是同一供应商快照一致性检查，不等于独立来源可靠性认证、真实分红支付账户或历史逐时点报价认证。

输出结果可指定到新目录，以保留多次调试：

```bash
.venv/bin/python run.py --model ridge --output experiments/trial01
```

每次运行覆盖所选输出目录中的相应文件。run.py 默认在训练后生成累计收益与回撤图；自定义结果目录的图表保存在该目录下的 figures/。单模型运行只展示该模型和两个基准，避免混入其他模型的旧预测。

## 自动生成累计收益与回撤图

```bash
.venv/bin/python run.py                  # 训练三个模型，自动画五条曲线
.venv/bin/python run.py --model ridge    # 只训练 Ridge，自动画 Ridge 与两个基准
.venv/bin/python run.py --cost-bps 5     # 单边买卖成本改为 5 个基点
.venv/bin/python run.py --no-plot        # 训练并导出比较表，不画图
.venv/bin/python plot_portfolio.py       # 复用已有预测，只更新图表
```

直接运行 `python -m models.ridge` 等调试入口仍只训练；随后可运行 `plot_portfolio.py --models ridge` 画对应曲线。

默认输出 `figures/portfolio_simulation_2025.png`、SVG、每日曲线 CSV、交易记录 CSV、汇总 CSV 和假设 JSON。单模型输出文件名附带模型名。

- 上图为累计价格收益，下图为从初始资金/历史高点计算的回撤。
- 模型规则：11 只 ETF 各预留 1/11，预测收益严格大于 0 才持有，否则留现金；每周按信号调仓。没有使用测试收益挑选阈值或持仓数量。
- 基准：SPY 买入持有；11 行业 ETF 每周等权再平衡。所有策略均计算买入及最终卖出成本。
- 默认单边成本为成交金额的 10 个基点（0.1%），只是可修改的假设，不是券商实测成本。换仓金额根据实际漂移持仓计算；目标仓位以扣费后的净值确定。
- 使用每天开盘价估值，覆盖共同完整标签区间：当前是 2025-01-06 至 2025-12-29，最后一天开盘清仓。不是完整自然年收盘收益，也不包含日内波动。
- 初始净值为 1，允许碎股，无杠杆，现金利息为零；收益不含现金分红、税款和真实执行摩擦。源行情已统一拆分单位，不重复应用拆分。
- 2025 测试结果之前已经查看，图标题标为探索性模拟，不能当作全新独立留出验证。曲线不保证复现参考图片。
- 绘图前核对原始数据指纹、模型可用时间、样本对齐及开盘标签收益；不一致时要求重新训练。

会计逻辑检查：`.venv/bin/python -m unittest test_portfolio.py`，覆盖现金、买入持有、费用、调仓漂移和非法参数。


## 画模型对比图

训练完成后，在本目录运行 `.venv/bin/python plot_results.py`。该脚本只读取三个模型已有的预测及指标，不重新训练或下载数据。

一张综合图展示测试集预测值与实际值、每周横截面平均走势、MAE/RMSE 对比。输出位于 figures/，包含 PNG 图片、可缩放 SVG 和图中指标 CSV；当前文件为 figures/model_comparison_2025.png。误差单位为百分点，周平均值不代表组合策略收益。

## 自动生成含简单基准的统一比较表和图片

在本目录运行以下命令，直接使用已有模型预测，不重新训练或下载数据：

```bash
.venv/bin/python compare_predictions.py
```

比较零收益、训练均值、Ridge、Random Forest、XGBoost，并加入“始终上涨”的方向准确率参考。训练均值是 2021—2023 年全部 1,683 条 ETF–周标签的共同平均值，固定用于所有 ETF 和所有分区，不使用验证或测试标签更新。训练指标仅为样本内参考。

自动输出：

| 文件 | 内容 |
|---|---|
| results/prediction_comparison_metrics.csv | 训练、验证、测试的全部指标，误差单位为收益小数；含相对零收益、训练均值、Ridge 的 RMSE 降幅（%） |
| results/prediction_comparison_test.csv | 适合报告使用的测试表，MAE/RMSE 单位为百分点，准确率单位为 % |
| results/prediction_comparison.md | 三个分区的可复制表格，以及基准、单位和评价规则说明 |
| results/baselines/zero_return/、train_mean/ | 基准的逐条预测、分区指标、周度 Rank IC 和参数记录 |
| results/prediction_comparison_audit.json | 输入文件指纹、脚本指纹、基准值与运行记录 |
| figures/prediction_comparison.png、.svg | 测试 MAE/RMSE、方向准确率、周度 Rank IC 三部分对比图 |

原始 CSV 中不适用的数值留空；报告表写 N/A。常数预测没有横截面排序，Rank IC 不适用、有效 IC 周数为 0。“始终上涨”只报告方向指标，不虚构 MAE/RMSE。

方向准确率沿用 `sign(预测) == sign(实际)`，其中 `sign(0)=0`。因此零收益基准仅在实际收益恰好为零时方向正确。训练均值当前为正，所以其方向准确率等于“始终上涨”，但两者的评价用途不同。RMSE 降幅定义为 `100 × (1 − 模型 RMSE / 基准 RMSE)`，正值为改善。

脚本会核对每个模型的样本指纹、分区、ETF、信号/入场/退出时间、特征、实际收益和源数据快照，并检查测试信号前模型是否已可用；输入过期或未对齐时停止，要求重新运行相应模型。

```bash
# 仅输出表格
.venv/bin/python compare_predictions.py --no-plot
# 单个模型与基准对比（文件名带 _ridge，不覆盖完整比较表）
.venv/bin/python compare_predictions.py --models ridge
# 从独立实验目录读取预测并输出对应表格及图片
.venv/bin/python compare_predictions.py --results experiments/trial01
# 检查训练均值无前视、方向/排名处理、样本对齐与模型可用时间
.venv/bin/python -m unittest test_prediction_comparison.py
```

`run.py` 已自动衔接该步骤；完整训练后自动生成三模型与基准的比较表和图片。单模型训练仅比较本次模型与基准，使用带模型名的文件名；`--no-plot` 同时跳过比较图和组合图，但仍生成比较表。已有的 `model_comparison_metrics.csv` 保持仅记录三个模型，新比较表使用独立文件名。
