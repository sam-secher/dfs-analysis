import pandas as pd


class EvaluationMetrics:
    def __init__(self, event_eval_df: pd.DataFrame, price_eval_df: pd.DataFrame, event_proportion_correct: float, price_r2: float, price_mae: float, price_rmse: float) -> None:
        self.event_eval_df = event_eval_df
        self.price_eval_df = price_eval_df
        self.event_proportion_correct = event_proportion_correct
        self.price_r2 = price_r2
        self.price_mae = price_mae
        self.price_rmse = price_rmse
