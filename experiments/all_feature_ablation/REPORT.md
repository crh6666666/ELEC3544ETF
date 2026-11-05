# 四特征逐一删除消融：完整结果

三种回归模型 × 五组特征，共15组实际重新训练；每次仅删除一个特征，没有搜索全部特征组合。
训练2021—2023（1683条/153周）、验证2024（561条/51周）、测试2025（561条/51周）。
固定原完整模型在2024验证选出的参数；仅训练集拟合，Ridge标准化也仅在训练集拟合。不重新选参，不合并训练/验证。
2024为开发分析，2025为已查看数据的探索性比较。未做显著性检验，不能解释为因果贡献。
MAE/RMSE（pp）为百分点；原始预测和metrics_all.csv为收益小数；ΔRMSE=删除后−完整，负值表示删后误差下降。
预测表保留全部四个原始特征用于核对；实际模型输入见UsedFeatures、Variant及各模型参数文件。

## 主要发现

- ridge：2024验证集删除后RMSE下降的实验：删除近期收益、删除均线比值。
- random_forest：2024验证集删除后RMSE下降的实验：无。
- xgboost：2024验证集删除后RMSE下降的实验：删除波动率、删除均线比值。
- 训练集近期收益与均线比值相关系数约0.840；单项消融衡量的是其他特征仍在时的额外贡献。
- 同周ETF存在共同市场暴露；按周结果与季度汇总另存，避免把561条记录当作561个独立时间样本。

## validation

| ModelId | FeatureGroup | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | 完整四特征 | validation | 561 | 51 | 1.7526 | 2.2960 | 54.5455 | 0.0597 | -0.2659 |
| random_forest | 完整四特征 | validation | 561 | 51 | 1.7508 | 2.2971 | 57.3975 | 0.0756 | -0.3117 |
| xgboost | 完整四特征 | validation | 561 | 51 | 1.7441 | 2.2923 | 56.8627 | 0.0125 | -0.1049 |
| ridge | 删除近期收益 | validation | 561 | 51 | 1.7440 | 2.2877 | 59.5365 | -0.0242 | 0.0974 |
| random_forest | 删除近期收益 | validation | 561 | 51 | 1.7460 | 2.2975 | 58.1105 | 0.0052 | -0.3309 |
| xgboost | 删除近期收益 | validation | 561 | 51 | 1.7429 | 2.2968 | 58.2888 | 0.0041 | -0.3006 |
| ridge | 删除波动率 | validation | 561 | 51 | 1.7501 | 2.2966 | 57.5758 | 0.0651 | -0.2909 |
| random_forest | 删除波动率 | validation | 561 | 51 | 1.7508 | 2.2990 | 56.6845 | 0.0557 | -0.3969 |
| xgboost | 删除波动率 | validation | 561 | 51 | 1.7423 | 2.2896 | 56.5062 | 0.0299 | 0.0153 |
| ridge | 删除均线比值 | validation | 561 | 51 | 1.7477 | 2.2901 | 58.4670 | 0.0312 | -0.0078 |
| random_forest | 删除均线比值 | validation | 561 | 51 | 1.7532 | 2.2986 | 56.5062 | -0.0199 | -0.3779 |
| xgboost | 删除均线比值 | validation | 561 | 51 | 1.7405 | 2.2844 | 59.5365 | -0.0215 | 0.2398 |
| ridge | 删除成交活跃度 | validation | 561 | 51 | 1.7511 | 2.2968 | 56.8627 | 0.0474 | -0.3001 |
| random_forest | 删除成交活跃度 | validation | 561 | 51 | 1.7547 | 2.3014 | 57.3975 | 0.0971 | -0.5006 |
| xgboost | 删除成交活跃度 | validation | 561 | 51 | 1.7479 | 2.2968 | 57.9323 | 0.0260 | -0.2976 |

删除后相对完整模型的变化：

| ModelId | Split | RemovedFeature | full_rmse_pp | removed_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | validation | return_5 | 2.2960 | 2.2877 | -0.0083 | -0.3623 | -0.0840 |
| random_forest | validation | return_5 | 2.2971 | 2.2975 | 0.0004 | 0.0192 | -0.0704 |
| xgboost | validation | return_5 | 2.2923 | 2.2968 | 0.0045 | 0.1955 | -0.0085 |
| ridge | validation | volatility_5 | 2.2960 | 2.2966 | 0.0006 | 0.0249 | 0.0053 |
| random_forest | validation | volatility_5 | 2.2971 | 2.2990 | 0.0020 | 0.0849 | -0.0199 |
| xgboost | validation | volatility_5 | 2.2923 | 2.2896 | -0.0028 | -0.1201 | 0.0173 |
| ridge | validation | price_ma_5 | 2.2960 | 2.2901 | -0.0059 | -0.2574 | -0.0285 |
| random_forest | validation | price_ma_5 | 2.2971 | 2.2986 | 0.0015 | 0.0660 | -0.0954 |
| xgboost | validation | price_ma_5 | 2.2923 | 2.2844 | -0.0079 | -0.3443 | -0.0340 |
| ridge | validation | activity_5 | 2.2960 | 2.2968 | 0.0008 | 0.0341 | -0.0123 |
| random_forest | validation | activity_5 | 2.2971 | 2.3014 | 0.0043 | 0.1883 | 0.0215 |
| xgboost | validation | activity_5 | 2.2923 | 2.2968 | 0.0044 | 0.1925 | 0.0134 |

## test

| ModelId | FeatureGroup | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | 完整四特征 | test | 561 | 51 | 2.0442 | 2.9748 | 54.9020 | -0.0034 | 0.4196 |
| random_forest | 完整四特征 | test | 561 | 51 | 2.0483 | 2.9616 | 53.8324 | 0.0537 | 0.8627 |
| xgboost | 完整四特征 | test | 561 | 51 | 2.0502 | 2.9630 | 53.1194 | 0.0217 | 0.8173 |
| ridge | 删除近期收益 | test | 561 | 51 | 2.0450 | 2.9648 | 55.0802 | -0.0075 | 0.7571 |
| random_forest | 删除近期收益 | test | 561 | 51 | 2.0326 | 2.9466 | 55.6150 | 0.0555 | 1.3644 |
| xgboost | 删除近期收益 | test | 561 | 51 | 2.0303 | 2.9364 | 55.6150 | 0.0611 | 1.7060 |
| ridge | 删除波动率 | test | 561 | 51 | 2.0438 | 2.9804 | 55.6150 | -0.0109 | 0.2345 |
| random_forest | 删除波动率 | test | 561 | 51 | 2.0487 | 2.9594 | 53.2977 | 0.0210 | 0.9376 |
| xgboost | 删除波动率 | test | 561 | 51 | 2.0519 | 2.9713 | 55.6150 | -0.0230 | 0.5377 |
| ridge | 删除均线比值 | test | 561 | 51 | 2.0371 | 2.9525 | 54.9020 | -0.0360 | 1.1663 |
| random_forest | 删除均线比值 | test | 561 | 51 | 2.0283 | 2.9419 | 58.6453 | 0.1224 | 1.5235 |
| xgboost | 删除均线比值 | test | 561 | 51 | 2.0338 | 2.9631 | 57.0410 | 0.0812 | 0.8120 |
| ridge | 删除成交活跃度 | test | 561 | 51 | 2.0511 | 2.9824 | 54.0107 | 0.0337 | 0.1664 |
| random_forest | 删除成交活跃度 | test | 561 | 51 | 2.0670 | 2.9857 | 52.0499 | 0.0246 | 0.0545 |
| xgboost | 删除成交活跃度 | test | 561 | 51 | 2.0602 | 2.9697 | 54.3672 | 0.0254 | 0.5920 |

删除后相对完整模型的变化：

| ModelId | Split | RemovedFeature | full_rmse_pp | removed_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | test | return_5 | 2.9748 | 2.9648 | -0.0101 | -0.3388 | -0.0041 |
| random_forest | test | return_5 | 2.9616 | 2.9466 | -0.0150 | -0.5061 | 0.0017 |
| xgboost | test | return_5 | 2.9630 | 2.9364 | -0.0265 | -0.8960 | 0.0394 |
| ridge | test | volatility_5 | 2.9748 | 2.9804 | 0.0055 | 0.1860 | -0.0075 |
| random_forest | test | volatility_5 | 2.9616 | 2.9594 | -0.0022 | -0.0756 | -0.0328 |
| xgboost | test | volatility_5 | 2.9630 | 2.9713 | 0.0084 | 0.2819 | -0.0447 |
| ridge | test | price_ma_5 | 2.9748 | 2.9525 | -0.0223 | -0.7498 | -0.0326 |
| random_forest | test | price_ma_5 | 2.9616 | 2.9419 | -0.0197 | -0.6666 | 0.0687 |
| xgboost | test | price_ma_5 | 2.9630 | 2.9631 | 0.0002 | 0.0054 | 0.0595 |
| ridge | test | activity_5 | 2.9748 | 2.9824 | 0.0076 | 0.2544 | 0.0371 |
| random_forest | test | activity_5 | 2.9616 | 2.9857 | 0.0241 | 0.8152 | -0.0292 |
| xgboost | test | activity_5 | 2.9630 | 2.9697 | 0.0067 | 0.2272 | 0.0037 |

## train

| ModelId | FeatureGroup | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | 完整四特征 | train | 1683 | 153 | 2.1680 | 2.8905 | 52.8223 | 0.0415 | 0.4104 |
| random_forest | 完整四特征 | train | 1683 | 153 | 2.1237 | 2.8237 | 58.2888 | 0.1345 | 2.7114 |
| xgboost | 完整四特征 | train | 1683 | 153 | 2.1477 | 2.8553 | 55.9121 | 0.1153 | 1.6252 |
| ridge | 删除近期收益 | train | 1683 | 153 | 2.1768 | 2.9003 | 53.5948 | 0.0104 | 0.0744 |
| random_forest | 删除近期收益 | train | 1683 | 153 | 2.1348 | 2.8390 | 56.9222 | 0.1269 | 2.1838 |
| xgboost | 删除近期收益 | train | 1683 | 153 | 2.1580 | 2.8742 | 55.3773 | 0.0472 | 0.9722 |
| ridge | 删除波动率 | train | 1683 | 153 | 2.1687 | 2.8913 | 53.7136 | 0.0708 | 0.3838 |
| random_forest | 删除波动率 | train | 1683 | 153 | 2.1275 | 2.8307 | 57.6352 | 0.1222 | 2.4720 |
| xgboost | 删除波动率 | train | 1683 | 153 | 2.1486 | 2.8577 | 56.0903 | 0.1177 | 1.5398 |
| ridge | 删除均线比值 | train | 1683 | 153 | 2.1756 | 2.8963 | 53.7730 | 0.0173 | 0.2116 |
| random_forest | 删除均线比值 | train | 1683 | 153 | 2.1442 | 2.8444 | 57.5758 | 0.1238 | 1.9977 |
| xgboost | 删除均线比值 | train | 1683 | 153 | 2.1574 | 2.8676 | 55.1396 | 0.0591 | 1.1988 |
| ridge | 删除成交活跃度 | train | 1683 | 153 | 2.1674 | 2.8912 | 52.5253 | 0.0308 | 0.3865 |
| random_forest | 删除成交活跃度 | train | 1683 | 153 | 2.1249 | 2.8277 | 57.3381 | 0.1264 | 2.5728 |
| xgboost | 删除成交活跃度 | train | 1683 | 153 | 2.1484 | 2.8573 | 55.0802 | 0.1132 | 1.5557 |

删除后相对完整模型的变化：

| ModelId | Split | RemovedFeature | full_rmse_pp | removed_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | train | return_5 | 2.8905 | 2.9003 | 0.0098 | 0.3374 | -0.0311 |
| random_forest | train | return_5 | 2.8237 | 2.8390 | 0.0153 | 0.5423 | -0.0076 |
| xgboost | train | return_5 | 2.8553 | 2.8742 | 0.0190 | 0.6637 | -0.0681 |
| ridge | train | volatility_5 | 2.8905 | 2.8913 | 0.0008 | 0.0267 | 0.0292 |
| random_forest | train | volatility_5 | 2.8237 | 2.8307 | 0.0069 | 0.2460 | -0.0123 |
| xgboost | train | volatility_5 | 2.8553 | 2.8577 | 0.0025 | 0.0868 | 0.0024 |
| ridge | train | price_ma_5 | 2.8905 | 2.8963 | 0.0058 | 0.1995 | -0.0242 |
| random_forest | train | price_ma_5 | 2.8237 | 2.8444 | 0.0207 | 0.7336 | -0.0108 |
| xgboost | train | price_ma_5 | 2.8553 | 2.8676 | 0.0124 | 0.4334 | -0.0562 |
| ridge | train | activity_5 | 2.8905 | 2.8912 | 0.0007 | 0.0239 | -0.0107 |
| random_forest | train | activity_5 | 2.8237 | 2.8277 | 0.0040 | 0.1425 | -0.0081 |
| xgboost | train | activity_5 | 2.8553 | 2.8573 | 0.0020 | 0.0706 | -0.0021 |

## 周度稳定性（2024验证）

| ModelId | RemovedFeature | Split | weeks | weeks_removed_better | mean_delta_mse | median_delta_mse | weeks_removed_better_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | activity_5 | validation | 51 | 25 | 0.0000 | 0.0000 | 49.0196 |
| random_forest | price_ma_5 | validation | 51 | 20 | 0.0000 | 0.0000 | 39.2157 |
| random_forest | return_5 | validation | 51 | 31 | 0.0000 | -0.0000 | 60.7843 |
| random_forest | volatility_5 | validation | 51 | 25 | 0.0000 | 0.0000 | 49.0196 |
| ridge | activity_5 | validation | 51 | 28 | 0.0000 | -0.0000 | 54.9020 |
| ridge | price_ma_5 | validation | 51 | 27 | -0.0000 | -0.0000 | 52.9412 |
| ridge | return_5 | validation | 51 | 28 | -0.0000 | -0.0000 | 54.9020 |
| ridge | volatility_5 | validation | 51 | 28 | 0.0000 | -0.0000 | 54.9020 |
| xgboost | activity_5 | validation | 51 | 21 | 0.0000 | 0.0000 | 41.1765 |
| xgboost | price_ma_5 | validation | 51 | 22 | -0.0000 | 0.0000 | 43.1373 |
| xgboost | return_5 | validation | 51 | 28 | 0.0000 | -0.0000 | 54.9020 |
| xgboost | volatility_5 | validation | 51 | 29 | -0.0000 | -0.0000 | 56.8627 |

## 2025探索性收益模拟

沿用既有规则：预测严格大于0时每只ETF占1/11，否则留现金；每周调仓，单边成本10 bps。
2025-01-06至2025-12-29，每日开盘估值、最后开盘清仓；不含分红、税、现金利息和真实执行摩擦。
收益变化也受现金暴露和换手成本影响，不能全部归于预测误差变化。

| Strategy | Variant | TotalPriceReturn_pct | MaxDrawdown_pct |
| --- | --- | --- | --- |
| spy | full | 15.0764 | -20.0069 |
| equal_weight | full | 10.6043 | -15.9876 |
| ridge | full | 7.7097 | -15.1502 |
| random_forest | full | 10.3332 | -8.8045 |
| xgboost | full | 10.7950 | -9.2017 |
| spy | without_return_5 | 15.0764 | -20.0069 |
| equal_weight | without_return_5 | 10.6043 | -15.9876 |
| ridge | without_return_5 | 9.7297 | -15.9876 |
| random_forest | without_return_5 | 13.4690 | -12.0540 |
| xgboost | without_return_5 | 13.2626 | -13.2871 |
| spy | without_volatility_5 | 15.0764 | -20.0069 |
| equal_weight | without_volatility_5 | 10.6043 | -15.9876 |
| ridge | without_volatility_5 | 9.5779 | -15.0984 |
| random_forest | without_volatility_5 | 10.3592 | -9.1893 |
| xgboost | without_volatility_5 | 13.0543 | -10.9301 |
| spy | without_price_ma_5 | 15.0764 | -20.0069 |
| equal_weight | without_price_ma_5 | 10.6043 | -15.9876 |
| ridge | without_price_ma_5 | 7.8112 | -15.7309 |
| random_forest | without_price_ma_5 | 16.0729 | -13.6269 |
| xgboost | without_price_ma_5 | 12.7675 | -15.5669 |
| spy | without_activity_5 | 15.0764 | -20.0069 |
| equal_weight | without_activity_5 | 10.6043 | -15.9876 |
| ridge | without_activity_5 | 8.8493 | -13.8890 |
| random_forest | without_activity_5 | 7.5051 | -8.6696 |
| xgboost | without_activity_5 | 12.2349 | -10.6637 |

## 数据与复现

predictions_paired.csv：2805条样本，每行真实收益与15组预测；2025专表为predictions_paired_2025.csv。
各特征组目录保存三种模型逐条预测、指标、参数、周度Rank IC及收益模拟图/交易流水。
comparison_readable.csv、ablation_deltas.csv为指标汇总与差异；baseline_metrics.csv为简单基准。
weekly_paired_errors.csv、weekly_stability_summary.csv及quarterly_paired_errors.csv为描述性稳定性数据。
ablation_rmse_changes.png展示验证/测试误差变化；portfolio_all_ablations_2025.png展示全部收益/回撤曲线。

在项目根目录执行（需选新的输出目录）：

```sh
.venv/bin/python -B run_all_feature_ablation.py --output experiments/all_feature_ablation_new
```
