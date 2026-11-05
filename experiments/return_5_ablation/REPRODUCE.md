# 复现方式

在项目根目录运行，训练输出必须选择新的 experiments 子目录，以免覆盖结果：

```sh
.venv/bin/python -B run_feature_ablation.py --output experiments/return_5_ablation_new
.venv/bin/python -B plot_portfolio.py --results experiments/return_5_ablation_new/full --output experiments/return_5_ablation_new/full/figures --cost-bps 10
.venv/bin/python -B plot_portfolio.py --results experiments/return_5_ablation_new/without_return_5 --output experiments/return_5_ablation_new/without_return_5/figures --cost-bps 10
```

如需生成合并收益图和表，将本目录的 build_portfolio_comparison.py 复制到新实验目录再运行。

固定参数来自项目 results 下的原完整模型参数记录：Ridge alpha=100；随机森林 300 棵树、深度 3、叶节点最少 20 条样本；XGBoost 50 棵树、深度 2、学习率 0.03。其他参数完整保存在各模型参数文件和 experiment.json。

训练集 1,683 条、153 周；验证集 561 条、51 周；测试集 561 条、51 周。每条为一个 ETF–周样本。

predictions_paired.csv 每一行包含四列原始特征、真实收益和六组预测；删除特征组的训练和预测均不使用 return_5。预测收益以小数表示，例如 0.01 表示 1%。comparison_readable.csv 的误差以百分点表示。

既有完整模型预测已核对复现，六组导出预测均重新计算指标核验。XGBoost 原 CSV 使用 float32 短精度写出，恢复 float32 后预测逐项一致；本次统一以 float64 导出，避免短精度文本丢失。

2024 用于开发和原参数选择；2025 此前已经查看。本实验是回顾性探索，未做显著性检验。收益模拟沿用已有项目规则，并非完整自然年总收益；不含分红、税费、现金利息和真实成交摩擦。
