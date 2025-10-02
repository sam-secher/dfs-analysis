import logging
from datetime import UTC, time
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from data.downloader import DataDownloader


class DFSForecastingModel:
    def __init__(self, data_downloader: DataDownloader, issue_time: time) -> None:

        self.logger = logging.getLogger(__name__)

        if not (time(9, 0) <= issue_time <= time(11, 0)):
            msg = "Issue time must be between 9am and 11am"
            self.logger.error(msg)
            raise ValueError(msg)

        self.forecast_time = issue_time # when the forecast is issued (DFS procurement historically between 10am and 11am, so we set to 10am)

        self.dfs_data = data_downloader.dfs_data
        self.lolp_drm_data = data_downloader.lolp_drm_data
        self.interconnector_data = data_downloader.interconnector_data
        self.settlement_data = data_downloader.settlement_data

        self.model_data = self._initialise_model_data()

        self.model_dfs_event = cast("Pipeline", None) # logistic regression model for DFS event prediction
        self.model_dfs_max_price = cast("Pipeline", None) # linear regression model for DFS max price prediction

    def predict(self) -> None:
        pass

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

        model_data["dfs_volume_1d_lag"] = model_data["dfs_volume_procured"].shift(freq="D")

        model_data = model_data[model_data.index >= pd.Timestamp(2025, 1, 23, tzinfo=UTC)] # exclude extremely high prices from Dec 24 - Jan 25

        model_data = model_data.dropna()

        def _get_drm_lolp_forecast(drm_lolp: Literal["drm", "lolp"], horizon: int) -> "pd.Series[float]":
            forecast = self.lolp_drm_data[self.lolp_drm_data["forecast_horizon"] == horizon][["datetime", drm_lolp]]
            return pd.Series(forecast[drm_lolp].values, index=forecast["datetime"])

        drm_forecast_1h = _get_drm_lolp_forecast("drm", 1)
        drm_forecast_2h = _get_drm_lolp_forecast("drm", 2)
        drm_forecast_4h = _get_drm_lolp_forecast("drm", 4)
        drm_forecast_8h = _get_drm_lolp_forecast("drm", 8)
        drm_forecast_12h = _get_drm_lolp_forecast("drm", 12)

        lolp_forecast_1h = _get_drm_lolp_forecast("lolp", 1)
        lolp_forecast_2h = _get_drm_lolp_forecast("lolp", 2)
        lolp_forecast_4h = _get_drm_lolp_forecast("lolp", 4)
        lolp_forecast_8h = _get_drm_lolp_forecast("lolp", 8)
        lolp_forecast_12h = _get_drm_lolp_forecast("lolp", 12)

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
        model_data = model_data.between_time("16:00", "21:00")

        if model_data.isna().any().any():
            err_msg = "Feature data missing"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        return model_data
