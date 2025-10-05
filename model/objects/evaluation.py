import pandas as pd


class EvaluationMetrics:
    def __init__(self, event_eval_df: pd.DataFrame, price_eval_df: pd.DataFrame, event_proportion_correct: float, non_event_proportion_correct: float,
    price_r2: float, price_mae: float, price_rmse: float, event_feat_importance_norm: "pd.Series[float]", price_feat_importance_norm: "pd.Series[float]") -> None:
        self.event_eval_df = event_eval_df
        self.price_eval_df = price_eval_df
        self.event_proportion_correct = event_proportion_correct
        self.non_event_proportion_correct = non_event_proportion_correct
        self.price_r2 = price_r2
        self.price_mae = price_mae
        self.price_rmse = price_rmse
        self.event_feat_importance_norm = event_feat_importance_norm
        self.price_feat_importance_norm = price_feat_importance_norm
