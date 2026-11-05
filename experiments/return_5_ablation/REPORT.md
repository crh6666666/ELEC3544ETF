# return_5 特征消融实验

三种模型 × 两组特征，均实际重新训练。训练：2021—2023；验证：2024；测试：2025。
固定原完整模型通过2024验证选出的参数；本次不重新选参，也不合并训练和验证集。
MAE/RMSE（pp）为收益百分点；原始预测和 metrics_all.csv 使用收益小数。
ΔRMSE = 删除后 − 完整；正值表示删除后变差。Rank IC 越大越好。
2024是开发分析，2025已查看、仅为探索性比较；不代表因果贡献或显著性。
预测文件保留四列原特征作核对，实际输入由 Variant、UsedFeatures 和参数文件明确记录。

## validation

| ModelId | Variant | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | full | validation | 561 | 51 | 1.7526 | 2.2960 | 54.5455 | 0.0597 | -0.2659 |
| random_forest | full | validation | 561 | 51 | 1.7508 | 2.2971 | 57.3975 | 0.0756 | -0.3117 |
| xgboost | full | validation | 561 | 51 | 1.7441 | 2.2923 | 56.8627 | 0.0125 | -0.1049 |
| ridge | without_return_5 | validation | 561 | 51 | 1.7440 | 2.2877 | 59.5365 | -0.0242 | 0.0974 |
| random_forest | without_return_5 | validation | 561 | 51 | 1.7460 | 2.2975 | 58.1105 | 0.0052 | -0.3309 |
| xgboost | without_return_5 | validation | 561 | 51 | 1.7429 | 2.2968 | 58.2888 | 0.0041 | -0.3006 |

| ModelId | Split | full_rmse_pp | without_return_5_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| ridge | validation | 2.2960 | 2.2877 | -0.0083 | -0.3623 | -0.0840 |
| random_forest | validation | 2.2971 | 2.2975 | 0.0004 | 0.0192 | -0.0704 |
| xgboost | validation | 2.2923 | 2.2968 | 0.0045 | 0.1955 | -0.0085 |

## test

| ModelId | Variant | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | full | test | 561 | 51 | 2.0442 | 2.9748 | 54.9020 | -0.0034 | 0.4196 |
| random_forest | full | test | 561 | 51 | 2.0483 | 2.9616 | 53.8324 | 0.0537 | 0.8627 |
| xgboost | full | test | 561 | 51 | 2.0502 | 2.9630 | 53.1194 | 0.0217 | 0.8173 |
| ridge | without_return_5 | test | 561 | 51 | 2.0450 | 2.9648 | 55.0802 | -0.0075 | 0.7571 |
| random_forest | without_return_5 | test | 561 | 51 | 2.0326 | 2.9466 | 55.6150 | 0.0555 | 1.3644 |
| xgboost | without_return_5 | test | 561 | 51 | 2.0303 | 2.9364 | 55.6150 | 0.0611 | 1.7060 |

| ModelId | Split | full_rmse_pp | without_return_5_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| ridge | test | 2.9748 | 2.9648 | -0.0101 | -0.3388 | -0.0041 |
| random_forest | test | 2.9616 | 2.9466 | -0.0150 | -0.5061 | 0.0017 |
| xgboost | test | 2.9630 | 2.9364 | -0.0265 | -0.8960 | 0.0394 |

## train

| ModelId | Variant | Split | rows | weeks | mae_pp | rmse_pp | direction_accuracy_pct | mean_weekly_rank_ic | rmse_reduction_vs_train_mean_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | full | train | 1683 | 153 | 2.1680 | 2.8905 | 52.8223 | 0.0415 | 0.4104 |
| random_forest | full | train | 1683 | 153 | 2.1237 | 2.8237 | 58.2888 | 0.1345 | 2.7114 |
| xgboost | full | train | 1683 | 153 | 2.1477 | 2.8553 | 55.9121 | 0.1153 | 1.6252 |
| ridge | without_return_5 | train | 1683 | 153 | 2.1768 | 2.9003 | 53.5948 | 0.0104 | 0.0744 |
| random_forest | without_return_5 | train | 1683 | 153 | 2.1348 | 2.8390 | 56.9222 | 0.1269 | 2.1838 |
| xgboost | without_return_5 | train | 1683 | 153 | 2.1580 | 2.8742 | 55.3773 | 0.0472 | 0.9722 |

| ModelId | Split | full_rmse_pp | without_return_5_rmse_pp | delta_rmse_pp | rmse_change_pct | delta_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| ridge | train | 2.8905 | 2.9003 | 0.0098 | 0.3374 | -0.0311 |
| random_forest | train | 2.8237 | 2.8390 | 0.0153 | 0.5423 | -0.0076 |
| xgboost | train | 2.8553 | 2.8742 | 0.0190 | 0.6637 | -0.0681 |

## 文件

- predictions_paired.csv：所有分区、真实收益、六组预测，同一行直接比较。
- predictions_paired_2025.csv：2025逐条对照。
- full/ 和 without_return_5/：各模型预测、指标、参数和周度Rank IC。
- baseline_metrics.csv：零收益、训练均值基准。
- weekly_paired_errors.csv：每周配对均方误差。
- ablation_deltas.csv：删除特征前后差异。
- ablation_rmse.png / .svg：验证和测试RMSE对比。

## 2025收益模拟

预测严格大于0时每只ETF占1/11，否则留现金；单边成本10 bps，包含首买与末次清仓。2025-01-06至2025-12-29，每日开盘估值，不含分红、税款和现金利息。

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

## 结果解释

- 2024验证：删除return_5仅改善Ridge的RMSE，随机森林与XGBoost略有变差；三者Rank IC均下降。
- 删除后的Ridge验证RMSE略优于训练均值，但Rank IC由正转负；收益误差和横截面排序不是同一目标。
- 2025探索性比较：删除后三者RMSE均下降，降幅相对完整模型分别约0.339%、0.506%、0.896%；不能据此认定稳定提升或普遍无效特征。
- 2025删除后的三个策略累计价格收益提高，最大回撤均加深。收益变化同时受持仓、现金暴露及换手成本影响，不能全部归因于预测改善。
- 训练集return_5与price_ma_5相关系数约0.840，删除后仍保留相关的价格信息。

全部数据见本目录CSV；复现步骤见REPRODUCE.md；原项目results、数据与训练代码保持原样，仅新增独立实验入口和实验目录。
