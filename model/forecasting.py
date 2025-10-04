import logging
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from data.downloader import DataDownloader
from model.objects.evaluation import EvaluationMetrics

if TYPE_CHECKING:
    from collections.abc import Iterable


class DFSForecastingModel:
    def __init__(self, data_downloader: DataDownloader, issue_time: time) -> None:

        self.logger = logging.getLogger(__name__)

        if not (time(9, 0) <= issue_time <= time(11, 0)):
            msg = "Issue time must be between 9am and 11am"
            self.logger.error(msg)
            raise ValueError(msg)

        self.forecast_time = issue_time # when the forecast is issued (DFS procurement historically between 10am and 11am, so we set to 10am)
        self._evening_block = ("16:00", "21:00")

        self.dfs_data = data_downloader.dfs_data
        self.lolp_drm_data = data_downloader.lolp_drm_data
        self.interconnector_data = data_downloader.interconnector_data
        self.settlement_data = data_downloader.settlement_data

        self.model_data = self._initialise_model_data()
        self.model_data_start = cast("date", self.model_data.index.min().date())
        self.model_data_end = cast("date", self.model_data.index.max().date())

        self.model_dfs_event = cast("Pipeline", None) # logistic regression model for DFS event prediction
        self.model_dfs_max_price = cast("Pipeline", None) # linear regression model for DFS max price prediction

    def predict(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        self.logger.info(f"Predicting DFS events and max prices between {start_dt.strftime('%d/%m/%Y %H:%M')} and {end_dt.strftime('%d/%m/%Y %H:%M')}")

        datetime_idx = pd.date_range(start=start_dt, end=end_dt, freq="30min")

        dfs_accepted = self.dfs_data[self.dfs_data["offer_status"] == "Accepted"]

        dfs_volume_actual = dfs_accepted[dfs_accepted["datetime"].between(start_dt, end_dt)]["offered_volume_mw"].groupby(dfs_accepted["datetime"]).sum()
        dfs_price_actual = dfs_accepted[dfs_accepted["datetime"].between(start_dt, end_dt)]["offered_price"].groupby(dfs_accepted["datetime"]).max()
        dfs_event_actual = (dfs_volume_actual > 0).astype(int)

        actuals_df = pd.DataFrame(index=datetime_idx)
        actuals_df["dfs_volume_actual"] = dfs_volume_actual
        actuals_df["dfs_price_actual"] = dfs_price_actual

        predict_df = pd.DataFrame(index=datetime_idx)

        mask = (self.model_data.index >= start_dt) & (self.model_data.index <= end_dt)

        features_event_pred = self.model_data[mask][[
            "drm_forecast_12h",
            "drm_forecast_8h",
            "lolp_forecast_12h",
            "lolp_forecast_8h",
            "interconnector_volume_1d_lag",
            "interconnector_cp_1d_lag",
            "interconnector_dispatched_1d_lag",
            "system_price_1d_lag",
            "niv_1d_lag",
            "dfs_volume_1d_lag",
        ]].between_time(*self._evening_block)

        dfs_event_prob = self.model_dfs_event.predict_proba(features_event_pred)[:, 1]
        dfs_event = pd.Series((dfs_event_prob > 0.7).astype(int), index=features_event_pred.index) # empirically determined threshold

        features_price_pred = features_event_pred[dfs_event == 1]
        dfs_max_price = (
            pd.Series([np.nan] * len(features_event_pred), index=features_event_pred.index)
            if features_price_pred.empty
            else self.model_dfs_max_price.predict(features_price_pred)
        )

        predict_df["dfs_event_pred"] = dfs_event
        predict_df["dfs_event_pred"] = predict_df["dfs_event_pred"].fillna(0)
        predict_df["dfs_max_price_pred"] = pd.Series(dfs_max_price, index=features_price_pred.index).round(0)

        return pd.concat([actuals_df, predict_df], axis=1)

    def evaluate_all(self) -> EvaluationMetrics:
        self.logger.info("Evaluating models across entire dataset")
        X_event, y_event = self._get_features_and_target(target_type="dfs_event")
        X_price, y_price = self._get_features_and_target(target_type="dfs_max_price")

        y_event_prob = self.model_dfs_event.predict_proba(X_event)[:, 1]
        y_event_pred = (y_event_prob > 0.7).astype(int) # empirically determined threshold

        event_eval_df = pd.concat([y_event, pd.Series(y_event_pred, index=y_event.index, name="dfs_event_pred")], axis=1)
        correct_predictions = ((event_eval_df["dfs_event"] == event_eval_df["dfs_event_pred"])*1).sum()
        proportion_correct = correct_predictions / len(y_event)

        y_price_prob = self.model_dfs_max_price.predict(X_price)
        price_eval_df = pd.concat([y_price, pd.Series(y_price_prob, index=y_price.index, name="dfs_max_price_pred")], axis=1)
        r2 = r2_score(y_price, y_price_prob)
        mae = mean_absolute_error(y_price, y_price_prob)
        rmse = mean_squared_error(y_price, y_price_prob)

        return EvaluationMetrics(event_eval_df, price_eval_df, proportion_correct, r2, mae, rmse)


    def train(self) -> None:
        self._train_logistic_regressor()
        self._train_linear_regressor()

    def _train_linear_regressor(self) -> None:
        self.logger.info("Training linear regression model for DFS max price prediction")
        X, y = self._get_features_and_target(target_type="dfs_max_price")

        n = len(X)
        split = int(0.85 * n) # dataset is too small for cross-validation, only ~550 DFS events

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lin", LinearRegression(fit_intercept=True)),
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.logger.info(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")

        pipe.fit(X, y) # fit on full dataset for deployment

        self.model_dfs_max_price = pipe

    def _train_logistic_regressor(self) -> None:
        self.logger.info("Training logistic regression model for DFS event prediction")

        X, y = self._get_features_and_target(target_type="dfs_event")

        tscv = TimeSeriesSplit(n_splits=5)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            # liblinear is very stable for small datasets; class_weight helps if events are rare
            ("clf", LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")),
        ])

        auc_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            self.logger.info(f"Fold {fold}")
            self.logger.info(classification_report(y_test, y_pred))
            self.logger.info("AUC: %f", roc_auc_score(y_test, y_prob))
            auc_scores.append(roc_auc_score(y_test, y_prob))

        self.logger.info("Mean AUC: %f", sum(auc_scores)/len(auc_scores))

        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": pipe.named_steps["clf"].coef_[0],
        }).sort_values("coef", ascending=False)

        self.logger.info(coef_df)

        pipe.fit(X, y) # fit on full dataset for deployment

        self.model_dfs_event = pipe

    def _get_features_and_target(self, target_type: Literal["dfs_event", "dfs_max_price"]) -> tuple[pd.DataFrame, "pd.Series[float | int]"]:

        model_data = (
            self.model_data[self.model_data["dfs_event"] == 1]
            if target_type == "dfs_max_price"
            else self.model_data
        )

        X = model_data[[ # features
            "drm_forecast_12h",
            "drm_forecast_8h",
            "lolp_forecast_12h",
            "lolp_forecast_8h",
            "interconnector_volume_1d_lag",
            "interconnector_cp_1d_lag",
            "interconnector_dispatched_1d_lag",
            "system_price_1d_lag",
            "niv_1d_lag",
            "dfs_volume_1d_lag",
        ]].astype(np.float64)

        y = model_data[target_type].astype(np.float64)

        return X, y

    def _initialise_model_data(self) -> pd.DataFrame:

        data_start_dt = self.dfs_data["datetime"].min()
        data_end_dt = self.dfs_data["datetime"].max()
        datetime_idx = pd.date_range(start=data_start_dt, end=data_end_dt, freq="30min")
        model_data = pd.DataFrame(index=datetime_idx)

        dfs_accepted = self.dfs_data[self.dfs_data["offer_status"] == "Accepted"]
        dfs_volume_procured = dfs_accepted["offered_volume_mw"].groupby(dfs_accepted["datetime"]).sum()
        dfs_max_price = dfs_accepted["offered_price"].groupby(dfs_accepted["datetime"]).max()

        model_data["dfs_volume_procured"] = dfs_volume_procured
        model_data["dfs_max_price"] = dfs_max_price
        model_data["dfs_event"] = dfs_volume_procured > 0
        model_data["dfs_event"] = model_data["dfs_event"].fillna(0).astype(int)

        model_data["dfs_volume_procured"] = model_data["dfs_volume_procured"].fillna(0.0)
        model_data["dfs_max_price"] = model_data["dfs_max_price"].fillna(0.0)
        model_data["dfs_event"] = model_data["dfs_event"].fillna(0).astype(int)

        model_data["dfs_volume_1d_lag"] = model_data["dfs_volume_procured"].shift(freq="D").fillna(0.0)

        model_data = model_data[model_data.index >= pd.Timestamp(2025, 1, 23, tzinfo=UTC)].copy() # exclude extremely high prices from Dec 24 - Jan 25

        model_data = model_data.dropna()

        drm_forecast_1h = self._get_drm_lolp_forecast(self.lolp_drm_data, "drm", 1)
        drm_forecast_2h = self._get_drm_lolp_forecast(self.lolp_drm_data, "drm", 2)
        drm_forecast_4h = self._get_drm_lolp_forecast(self.lolp_drm_data, "drm", 4)
        drm_forecast_8h = self._get_drm_lolp_forecast(self.lolp_drm_data, "drm", 8)
        drm_forecast_12h = self._get_drm_lolp_forecast(self.lolp_drm_data, "drm", 12)

        lolp_forecast_1h = self._get_drm_lolp_forecast(self.lolp_drm_data, "lolp", 1)
        lolp_forecast_2h = self._get_drm_lolp_forecast(self.lolp_drm_data, "lolp", 2)
        lolp_forecast_4h = self._get_drm_lolp_forecast(self.lolp_drm_data, "lolp", 4)
        lolp_forecast_8h = self._get_drm_lolp_forecast(self.lolp_drm_data, "lolp", 8)
        lolp_forecast_12h = self._get_drm_lolp_forecast(self.lolp_drm_data, "lolp", 12)

        # if forecast unavailable at issue time, need to use neutral replacement or set to NA
        cutoff_8hr = time(self.forecast_time.hour + 8, self.forecast_time.minute)
        cutoff_12hr = time(self.forecast_time.hour + 12, self.forecast_time.minute)

        drm_forecast_12h = drm_forecast_12h.mask(drm_forecast_12h.index.time > cutoff_12hr) # type: ignore[attr-defined]
        lolp_forecast_12h = lolp_forecast_12h.mask(lolp_forecast_12h.index.time > cutoff_12hr) # type: ignore[attr-defined]

        # 8hr forecasts unavailable after 6pm, use 12hr forecast as neutral replacement
        drm_forecast_8h = drm_forecast_8h.mask(drm_forecast_8h.index.time > cutoff_8hr, drm_forecast_12h) # type: ignore[attr-defined]
        lolp_forecast_8h = lolp_forecast_8h.mask(lolp_forecast_8h.index.time > cutoff_8hr, lolp_forecast_12h) # type: ignore[attr-defined]

        model_data["drm_forecast_12h"] = drm_forecast_12h
        model_data["drm_forecast_8h"] = drm_forecast_8h
        model_data["lolp_forecast_12h"] = lolp_forecast_12h
        model_data["lolp_forecast_8h"] = lolp_forecast_8h

        model_data = model_data.dropna() # can't train model where missing DRM/LoLP forecast data

        interconnector_volume = (
            self.interconnector_data["volume_ifa1"] +
            self.interconnector_data["volume_ifa2"] +
            self.interconnector_data["volume_vkl"] +
            self.interconnector_data["volume_el"] +
            self.interconnector_data["volume_nemo"] +
            self.interconnector_data["volume_bn"]
        ).groupby(self.interconnector_data["datetime"]).sum()

        # interconnector data will not be available at issue time, using previous day's data as proxy
        interconnector_volume_1d_lag = interconnector_volume.shift(freq="D")
        interconnector_cp = self.interconnector_data.set_index("datetime")["clearing_price"]
        interconnector_cp_1d_lag = interconnector_cp.shift(freq="D")
        model_data["interconnector_volume_1d_lag"] = interconnector_volume_1d_lag
        model_data["interconnector_cp_1d_lag"] = interconnector_cp_1d_lag
        model_data["interconnector_dispatched_1d_lag"] = interconnector_cp_1d_lag.notna().astype(int)

        model_data["interconnector_volume_1d_lag"] = model_data["interconnector_volume_1d_lag"].fillna(0.0)
        model_data["interconnector_cp_1d_lag"] = model_data["interconnector_cp_1d_lag"].fillna(0.0)
        model_data["interconnector_dispatched_1d_lag"] = model_data["interconnector_dispatched_1d_lag"].fillna(0).astype(int)

        model_data["system_price_1d_lag"] = self.settlement_data.set_index("datetime")["system_price"].shift(freq="D")
        model_data["niv_1d_lag"] = self.settlement_data.set_index("datetime")["niv"].shift(freq="D")

        # constrain model data to period of interest (evening)
        model_data = model_data.between_time(*self._evening_block)

        if model_data.isna().any().any():
            err_msg = "Feature data missing"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        return model_data

    def _dfs_event_dates(self) -> "Iterable[date]":
        return self.model_data[self.model_data["dfs_event"] == 1].index.unique().date # type: ignore[attr-defined]

    def _get_drm_lolp_forecast(self, lolp_drm_data: pd.DataFrame, drm_lolp: Literal["drm", "lolp"], horizon: int) -> "pd.Series[float]":
        forecast = lolp_drm_data[lolp_drm_data["forecast_horizon"] == horizon][["datetime", drm_lolp]]
        return pd.Series(forecast[drm_lolp].values, index=forecast["datetime"])

    def _get_total_interconnector_volume(self, interconnector_data: pd.DataFrame) -> "pd.Series[float]":
        return (
            interconnector_data["volume_ifa1"] +
            interconnector_data["volume_ifa2"] +
            interconnector_data["volume_vkl"] +
            interconnector_data["volume_el"] +
            interconnector_data["volume_nemo"] +
            interconnector_data["volume_bn"]
        ).groupby(self.interconnector_data["datetime"]).sum()
