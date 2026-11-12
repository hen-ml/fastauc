# API reference

## fastauc

::: fastauc
    options:
      show_root_heading: false
      members:
        - binary_metrics
        - roc_auc
        - average_precision
        - pr_auc
        - partial_roc_auc
        - roc_auc_multiclass
        - roc_auc_ap
        - roc_auc_ap_batch
        - roc_auc_ap_grouped
        - GroupedResult
        - ks_statistic
        - max_f1
        - precision_at_recall
        - recall_at_precision
        - tpr_at_fpr
        - lift_at
        - roc_curve
        - precision_recall_curve
        - calibration_curve
        - calibration_metrics
        - log_loss
        - brier_score
        - normalized_entropy
        - calibration_ratio
        - expected_calibration_error
        - ranking_metrics
        - bootstrap_roc_auc_ap
        - BootstrapResult
        - bootstrap_ranking_metrics
        - RankingBootstrapResult
        - delong_roc_auc
        - DeLongResult
        - paired_test
        - PairedTestResult
        - holm
        - BinaryAccumulator
        - set_num_threads
        - num_threads

## fastauc.sklearn

::: fastauc.sklearn
    options:
      show_root_heading: false
      members:
        - roc_auc_score
        - average_precision_score
        - roc_curve
        - precision_recall_curve
        - auc
        - ndcg_score
        - dcg_score
        - log_loss
        - brier_score_loss

## fastauc.lightgbm

::: fastauc.lightgbm
    options:
      show_root_heading: false
      members:
        - ranking
        - binary

## fastauc.xgboost

::: fastauc.xgboost
    options:
      show_root_heading: false
      members:
        - ranking
        - binary
