from datetime import UTC, date, datetime, time
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit.elements.plotly_chart import PlotlyState

from model.forecasting import DFSForecastingModel
from model.objects.evaluation import EvaluationMetrics
from utils.datetime_helpers import datetime_to_sp_series, datetime_to_sp


class App:
    def __init__(self, forecasting_model: DFSForecastingModel) -> None:
        self._model = forecasting_model
        self.set_page_config()
        self.PAGES = ("Home", "Dashboard", "Forecasting", "Next steps")
        self.page = "Home"
        self._render_sidebar()
        self.route() # fire on every script reload

    def set_page_config(self) -> None:
        st.set_page_config(
            page_title="DFS Analysis – Take-home",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _render_sidebar(self) -> None:

        with st.sidebar:
            st.title("DFS Analysis")
            self.page = st.radio("Navigate", self.PAGES, index=0)
            st.markdown("---")
            st.caption("Model Information")

            # Global date window (used by Dashboard)
            today = datetime.now(tz=UTC).date()
            # default_start = today - timedelta(days=120)
            start, end = self._model.model_data_start, self._model.model_data_end
            date_range = st.date_input(
                "Date range",
                (start, end),
                help="Date range for data pre-loaded from NESO and Elexon APIs.",
                disabled=True,
            )
            st.session_state["date_range"] = date_range

            # Morning-of forecast time (for your within-day logic)
            forecast_time = st.time_input(
                "Forecast issue time",
                value=self._model.forecast_time,
                help="Model assumes DFS procurement occurs between 10am and 11am.",
                disabled=True,
            )
            st.session_state["forecast_time"] = forecast_time

            st.markdown("---")
            st.caption("Data sources")
            st.markdown("[NESO](https://www.neso.energy/data-portal) — DFS and interconnector data")
            st.markdown("[Elexon](https://bmrs.elexon.co.uk/) — DRM/LoLP and settlement data")
            # use_live_sources = st.checkbox("Use live APIs when available", value=False)
            # st.session_state["use_live_sources"] = use_live_sources


    def _load_historic_data(self, date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load historic data for a given date."""
        def _get_data_for_date(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["datetime"].dt.date == date]

        datetime_idx = pd.date_range(datetime(date.year, date.month, date.day), end=datetime(date.year, date.month, date.day, 23, 30), freq="30min", tz=UTC)
        settlement_periods = datetime_to_sp_series(pd.Series(datetime_idx))
        timeseries_df = pd.DataFrame(index=datetime_idx)
        timeseries_df["settlement_period"] = settlement_periods.values

        dfs_data = _get_data_for_date(self._model.dfs_data)
        lolp_drm_data = _get_data_for_date(self._model.lolp_drm_data)
        interconnector_data = _get_data_for_date(self._model.interconnector_data)
        settlement_data = _get_data_for_date(self._model.settlement_data).set_index("datetime")

        dfs_accepted = dfs_data[dfs_data["offer_status"] == "Accepted"]
        dfs_volume_procured = dfs_accepted.groupby("datetime")["offered_volume_mw"].sum()
        dfs_max_price = dfs_accepted.groupby("datetime")["offered_price"].max()
        dfs_min_price = dfs_accepted.groupby("datetime")["offered_price"].min()
        dfs_vwa_price = (
            (dfs_accepted["offered_price"] * dfs_accepted["offered_volume_mw"]).groupby(dfs_accepted["datetime"]).sum() /
            dfs_accepted["offered_volume_mw"].groupby(dfs_accepted["datetime"]).sum()
        )

        timeseries_df["dfs_volume_procured"] = dfs_volume_procured
        timeseries_df["dfs_max_price"] = dfs_max_price
        timeseries_df["dfs_min_price"] = dfs_min_price
        timeseries_df["dfs_vwa_price"] = dfs_vwa_price

        timeseries_df["drm_forecast_12h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "drm", 12)
        timeseries_df["drm_forecast_8h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "drm", 8)
        timeseries_df["drm_forecast_4h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "drm", 4)
        timeseries_df["drm_forecast_2h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "drm", 2)
        timeseries_df["drm_forecast_1h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "drm", 1)
        timeseries_df["lolp_forecast_12h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "lolp", 12)
        timeseries_df["lolp_forecast_8h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "lolp", 8)
        timeseries_df["lolp_forecast_4h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "lolp", 4)
        timeseries_df["lolp_forecast_2h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "lolp", 2)
        timeseries_df["lolp_forecast_1h"] = self._model._get_drm_lolp_forecast(lolp_drm_data, "lolp", 1)

        timeseries_df["total_interconnector_volume"] = self._model._get_total_interconnector_volume(interconnector_data)
        timeseries_df["system_price"] = settlement_data["system_price"]
        timeseries_df["niv"] = settlement_data["niv"]

        return dfs_data, timeseries_df

    def _render_interconnector_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> None:
        fig = self._create_interconnector_chart(timeseries_df, date_chosen)
        self._render_chart(fig, "Interconnection", help_text="Hover over bars to inspect values, negative is export.")

    def _create_interconnector_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> go.Figure:
        fig = make_subplots()

        x_axis = timeseries_df.index.strftime("%H:%M") # type: ignore[attr-defined]

        hovertemplate = "<b>%{fullData.name}</b><br>%{x}, %{y:,.0f}MW<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=timeseries_df["total_interconnector_volume"],
                name="Net Interconnector Position",
                marker_color="navy",
                hovertemplate=hovertemplate,
            ),
            secondary_y=False,
        )

        fig.update_yaxes(title_text="Position (MW)", showgrid=True, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(showgrid=False, tickfont=dict(color="black"))

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,           # move legend below chart area
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(t=10, r=10, b=10, l=10),
            bargap=0.2,
            dragmode="zoom"
        )

        return fig

    def _render_settlement_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> None:
        fig = self._create_settlement_chart(timeseries_df, date_chosen)
        self._render_chart(fig, "Settlement Data", help_text="Hover over series to inspect values.")

    def _create_settlement_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        x_axis = timeseries_df.index.strftime("%H:%M") # type: ignore[attr-defined]

        niv_hovertemplate = "<b>%{fullData.name}</b><br>%{x}, %{y:,.0f}MW<extra></extra>"
        cashout_price_hovertemplate = "<b>%{fullData.name}</b><br>%{x}, £%{y:.1f}/MWh<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=timeseries_df["niv"],
                name="NIV",
                marker_color="orange",
                hovertemplate=niv_hovertemplate,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeseries_df["system_price"],
                name="System Price",
                mode="lines",
                line=dict(color="black", dash="dash"),
                hovertemplate=cashout_price_hovertemplate,
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Net Imbalance Volume (MW)", secondary_y=False, showgrid=True, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(title_text="System Price (£/MWh)", secondary_y=True, showgrid=False, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(showgrid=False, tickfont=dict(color="black"))

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,           # moves below chart area
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(t=10, r=10, b=10, l=10),
            bargap=0.2,
            dragmode="zoom"
        )

        return fig

    def _render_lolp_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> None:
        fig = self._create_lolp_chart(timeseries_df, date_chosen)
        self._render_chart(fig, "LoLP", help_text="Hover over line to inspect values.")

    def _create_lolp_chart(self, timeseries_df: pd.DataFrame, date_chosen: date) -> go.Figure:
        fig = make_subplots()

        x_axis = timeseries_df.index.strftime("%H:%M") # type: ignore[attr-defined]

        hovertemplate = "<b>%{fullData.name}</b><br>%{x}, %{y:,.6%}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeseries_df["lolp_forecast_1h"],
                name="LoLP Forecast (1hr)",
                mode="lines",
                line=dict(color="orange"),
                hovertemplate=hovertemplate,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeseries_df["lolp_forecast_12h"],
                name="LoLP Forecast (12hr)",
                mode="lines",
                line=dict(color="green"),
                hovertemplate=hovertemplate,
            )
        )

        fig.update_yaxes(tickformat=".6%", title_text="%", showgrid=True, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(showgrid=False, tickfont=dict(color="black"))

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,           # move legend below chart area
                xanchor="center",
                x=0.5,
                title=None,
            ),
            autosize=True,
            margin=dict(t=10, r=10, b=10, l=10),
            # dragmode="zoom"
        )

        return fig

    def _render_dfs_chart(self, timeseries_df: pd.DataFrame, dfs_data: pd.DataFrame, date_chosen: date) -> None:
        event: PlotlyState | dict[str, Any] | None = None

        if "dfs_selected_ts" in st.session_state and st.session_state["dfs_selected_ts"] is not None:
            self._render_dfs_detail(dfs_data)
        else:
            event = self._render_chart(
                self._generate_dfs_chart(timeseries_df, dfs_data),
                "DFS",
                "Click on bar to drill-down into DFS auction details.",
                event_on_select=True,
            )

        self._handle_dfs_chart_click(event, timeseries_df, date_chosen)


    def _generate_dfs_chart(self, timeseries_df: pd.DataFrame, dfs_data: pd.DataFrame) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        x_axis = timeseries_df.index.strftime("%H:%M") # type: ignore[attr-defined]

        hovertemplate = "<b>%{fullData.name}</b><br>%{x}, %{y:,.0f}MW<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=timeseries_df["dfs_volume_procured"],
                name="DFS Procured",
                marker_color="navy",
                hovertemplate=hovertemplate,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeseries_df["drm_forecast_1h"],
                name="DRM Forecast (1hr)",
                mode="lines",
                line=dict(color="orange"),
                hovertemplate=hovertemplate,
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=timeseries_df["drm_forecast_12h"],
                name="DRM Forecast (12hr)",
                mode="lines",
                line=dict(color="green"),
                hovertemplate=hovertemplate,
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="DFS Procured (MW)", secondary_y=False, showgrid=True, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(title_text="Derated Margin (MW)", secondary_y=True, showgrid=False, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(showgrid=False, tickfont=dict(color="black"))

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,           # move legend below chart area
                xanchor="center",
                x=0.5,
                title=None,
            ),
            autosize=True,
            margin=dict(t=10, r=10, b=10, l=10),
            bargap=0.2,
            dragmode="zoom"
        )

        return fig

    def _handle_dfs_chart_click(self, event: PlotlyState | dict | None, timeseries_df: pd.DataFrame, date_chosen: date) -> None:
        ts = self._get_bar_click_timestamp(event, timeseries_df, date_chosen)
        if ts is not None:
            st.session_state["dfs_selected_ts"] = ts
            st.rerun()

        if isinstance(event, dict): # clear selected timestamp if selection is empty
            sel = cast(dict[str, Any], event.get("selection", {}))
            points = sel.get("points", [])
            if not points and st.session_state.get("dfs_selected_ts"):
                st.session_state["dfs_selected_ts"] = None
                st.rerun()

    def _render_dfs_detail(self, dfs_data: pd.DataFrame) -> None:
        ts = st.session_state["dfs_selected_ts"]
        st.subheader(f"DFS results - {ts:%H:%M} UTC (SP {datetime_to_sp(ts)})")
        detail = dfs_data.loc[dfs_data["datetime"] == ts].sort_values("offered_price", ascending=False)
        detail = detail[["dfs_unit_id", "dfs_participant", "offered_volume_mw", "offered_price", "offer_status"]]
        detail.columns = ["Unit ID", "Participant", "Volume (MW)", "Price (GBP/MWh)", "Status"]

        st.dataframe(detail, hide_index=True, width="stretch")

        if st.button("Back to chart"):
            st.session_state["dfs_selected_ts"] = None
            st.rerun()

    def _render_chart(self, fig: go.Figure, title: str = "", help_text: str = "", event_on_select: bool = False) -> PlotlyState:
        if title:
            st.markdown(f"**{title}**")
        if help_text:
            st.caption(help_text)

        kwargs = cast(dict[str, Any], { "use_container_width": True })
        if event_on_select:
            kwargs["on_select"] = "rerun"
            kwargs["selection_mode"] = ("points",)

        return st.plotly_chart(fig, **kwargs)

    def _get_bar_click_timestamp(self, event: PlotlyState | dict | None, timeseries_df: pd.DataFrame, date_chosen: date) -> datetime | None:
        """Extract timestamp only if the bar trace was clicked. Return None otherwise."""
        if not isinstance(event, dict):
            return None
        sel = event.get("selection") or {}
        points = sel.get("points") or []
        if not points:
            return None
        p = points[0]

        # Identify the bar trace (by index). Common keys: 'curveNumber', 'trace_index'.
        if p["curve_number"] != 0: # bar trace index
            return None  # ignore clicks on the line traces

        timestamp = pd.to_datetime(p["x"], utc=True).floor("30min")

        return datetime.combine(date_chosen, timestamp.time(), tzinfo=UTC)

    def _render_home(self) -> None:
        st.title("Demand Flexibility Service (DFS) Analysis")
        st.subheader("Overview")

        st.markdown(
            """
            This is a workspace to explore **historic DFS activity**, track **system tightness** signals
            (e.g., DRM/LoLP), and prototype a **same-day (10:00) event and max price forecast**.

            ### Contents
            - **Dashboard**: quick plots of historic DFS events, derated margin & LoLP, interconnector positions, and system price/NIV context.
            - **Forecasting**: fit model for event and max price prediction and evaluate performance.
            - **Next steps**: further tasks required to evolve this into a commercially useful bidding aid.
            """,
        )

        with st.expander("Assumptions & scope", expanded=True):
            st.markdown(
                f"""
                - Data is pre-downloaded from NESO and Elexon APIs between **{self._model.model_data_start.strftime("%b-%y")} and {self._model.model_data_end.strftime("%b-%y")}**.
                - Model focus is the **evening peak**, assumed to be between {self._model._evening_block[0]} and {self._model._evening_block[1]}.
                - Auction is **pay-as-bid**, maximum accepted price is the forecast target to guide bidding.
                - Only inputs available by ~**{self._model.forecast_time.strftime("%H:%M")}** are considered for forecasting.
                - Features used include DRM and LoLP forecasts **(8hr and 12hr ahead)**, interconnection positions, system price and NIV  and DFS volume **(1-day lag)**.
                """,
            )

        st.markdown("---")
        st.markdown("#### Quick start")
        st.markdown(
            """
            - Open **Dashboard** to inspect historic DFS events and related system tightness indicators.
            - Use **Forecasting** to evaluate logit and regression model performance (used for event and price prediction respectively).
            - Read **Next steps** to understand model limitations and useful future developments.
            """,
        )

        st.markdown("---")
        st.markdown("#### Repository and Documentation")

        st.markdown(
            """
            This project is fully open-source.
            The complete codebase, including model equations, training logic, and evaluation details,
            is available on GitHub:

            [**View the repository here**](https://github.com/sam-secher/dfs-analysis)

            The README contains detailed information on:
            - Model formulation
            - Feature engineering and data timing assumptions
            - Evaluation metrics and validation approach
            - Suggested improvements and roadmap
            """,
        )

    def _render_dashboard(self) -> None:
        st.title("Dashboard")
        st.caption("DFS event viewer and system tightness indicators")

        date_chosen = self._render_date_slider()

        st.subheader(date_chosen.strftime("%d/%m/%Y"))

        dfs_data, timeseries_df = self._load_historic_data(date_chosen)

        with st.container(border=True):
            c1, c2 = st.columns(2, gap="small")

            with c1:
                self._render_dfs_chart(timeseries_df, dfs_data, date_chosen)
            with c2:
                self._render_lolp_chart(timeseries_df, date_chosen)

            d1, d2 = st.columns(2, gap="small")

            with d1:
                self._render_settlement_chart(timeseries_df, date_chosen)
            with d2:
                self._render_interconnector_chart(timeseries_df, date_chosen)

    def _render_date_slider(self) -> date:
        dfs_event_dates = self._model._dfs_event_dates()
        date_default = cast("date", st.session_state.get("date_chosen", dfs_event_dates[-1]))
        date_chosen = st.select_slider(
            "Date picker",
            options=dfs_event_dates,
            value=date_default,
            format_func=lambda d: d.strftime("%d/%m/%Y"),
            help="Only dates with DFS used are shown.",
            key="date_chosen",
        )

        return date_chosen

    def _render_forecasting(self) -> None:
        st.title("Forecasting")

        with st.spinner("Training models..."):
            self._model.train() # trained pipes cached so this only runs when model data is updated

        date_chosen = self._render_date_slider()

        with st.container(border=True):
            st.subheader(f"{date_chosen.strftime("%d/%m/%Y")} - Actual vs. Predicted", help="Red cells indicate event prediction inconsistent with actuals.")

            start = datetime.combine(date_chosen, time(0, 0), tzinfo=UTC)
            end = datetime.combine(date_chosen, time(23, 30), tzinfo=UTC)

            results_df = self._model.predict(start, end)

            format_map = {
                "DFS Volume Actual (MW)": "{:,.1f}",
                "DFS Max Price Actual (GBP/MWh)": "{:.1f}",
                "DFS Predicted (0/1)": "{:.0f}",
                "DFS Max Price Predicted (GBP/MWh)": "{:.1f}",
            }

            results_df.index = results_df.index.strftime("%H:%M") # type: ignore[attr-defined]
            results_df = results_df.reset_index(drop=False)
            results_df.columns = pd.Index(["Time", *format_map.keys()])

            def style_func(df) -> pd.DataFrame:
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for idx, row in df.iterrows():
                    volume_actual, event_pred = row["DFS Volume Actual (MW)"], row["DFS Predicted (0/1)"]
                    if (volume_actual > 0) ^ (event_pred == 1):
                        styles.loc[idx, "DFS Predicted (0/1)"] = "background-color: red"
                return styles

            st.dataframe(results_df.style.apply(style_func, axis=None).format(format_map, na_rep=""), hide_index=True, width="stretch") # type: ignore[call-overload, arg-type]

        # st.markdown("---")

        with st.container(border=True):

            st.subheader("Model Evaluation (entire dataset)")

            c1, c2 = st.columns(2, gap="medium")

            eval_metrics = self._model.evaluate_all()

            with c1:
                self._render_price_evaluation_chart(eval_metrics)
            with c2:
                st.container(height=85, border=False)
                self._render_overall_evaluation_table(eval_metrics)

    def _render_overall_evaluation_table(self, eval_metrics: EvaluationMetrics) -> None:

        column_config = {
            "Metric": st.column_config.Column(width="medium"),
            "Value": st.column_config.Column(width="small"),
            "Score": st.column_config.Column(width="large"),
        }

        # columns = column_config.keys()
        rows = [
            ["Event Proportion Correct", "%", f"{eval_metrics.event_proportion_correct:.2%}"],
            ["Non-Event Proportion Correct", "%", f"{eval_metrics.non_event_proportion_correct:.2%}"],
            ["Price R-squared", "-", f"{eval_metrics.price_r2:.2f}"],
            ["Price MAE", "GBP/MWh", f"{eval_metrics.price_mae:.2f}"],
            ["Price RMSE", "GBP/MWh", f"{eval_metrics.price_rmse:.2f}"],
        ]

        event_important_feats = eval_metrics.event_feat_importance_norm.nlargest(3).index.tolist()
        price_important_feats = eval_metrics.price_feat_importance_norm.nlargest(3).index.tolist()

        event_important_feats_str = ", ".join(event_important_feats)
        price_important_feats_str = ", ".join(price_important_feats)

        rows.extend([
            ["Important Features - Event", "-", event_important_feats_str],
            ["Important Features - Price", "-", price_important_feats_str],
        ])

        eval_df = pd.DataFrame(rows, columns=column_config.keys()) # type: ignore[call-overload]
        st.dataframe(eval_df, hide_index=True, width="stretch", column_config=column_config)

    def _render_price_evaluation_chart(self, eval_metrics: EvaluationMetrics) -> None:
        fig = self._create_price_evaluation_chart(eval_metrics)
        self._render_chart(fig)

    def _create_price_evaluation_chart(self, eval_metrics: EvaluationMetrics) -> go.Figure:
        fig = make_subplots()

        x_axis = list(range(len(eval_metrics.price_eval_df)))

        hovertemplate = "<b>%{fullData.name}</b><br>%{x}, £%{y:.1f}/MWh<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eval_metrics.price_eval_df["dfs_max_price"],
                name="Actual",
                mode="lines",
                line=dict(color="black"),
                hovertemplate=hovertemplate,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eval_metrics.price_eval_df["dfs_max_price_pred"],
                name="Predicted",
                mode="lines",
                line=dict(color="red"),
                hovertemplate=hovertemplate,
            )
        )

        fig.update_yaxes(title_text="DFS Max Price (GBP/MWh)", showgrid=True, tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(title_text="Index", showgrid=False, tickfont=dict(color="black"), title_font=dict(color="black"))

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,           # move legend below chart area
                xanchor="center",
                x=0.5,
                title=None,
            ),
        )

        return fig

    def _render_next_steps(self) -> None:
        st.title("Further Improvements")
        st.markdown(
            """
            Below are suggested improvements make the forecasting model more commercially useful.

            ### 1) Structural Changes
            - The statistical learning approach provides a **first baseline**, but predictive power remains limited by available data and simplifications.
            - Incorporate **fundamental drivers** to complement the existing models
                - Develop a **stack simulation model** to replicate NESO bid clearing against alternative flexible sources
                - Would require significantly more data e.g. **capacity mix, weather/demand/outages** etc
                - Could develop **interconnector** supply curves to infer likely price level as demand rises / DRM drops

            ### 2) Data Pipeline & Feature Engineering
            - Clients have been written to pull data from NESO and Elexon APIS, but these are not yet hooked up to the app
                - Integrate and automate daily retraining
            - Feature set can be expanded with:
                - **Calendar-based effects** (weekday/weekend, holidays, seasonality).
                - **Demand and weather forecasts**, public sources available via NESO, high resolution private sources also available (e.g. Meteomatics).
                - Experiment further with **feature lags and rolling aggregates** to capture persistence / intra-week patterns (e.g., 7-day rolling average DRM or NIV).

            ### 3) Evaluation and additional functionality
            - As historic data is made available, apply cross-validation to price model for more robust performance metrics
            - Implement **confidence or prediction intervals** for price forecasts
            - Build a **simple revenue estimator**:
                `expected_revenue = 0.5 × volume × bid_price × K_factor`
                - `K_factor` is DFS availability deviation from offered
                - `volume` offered in MW
                - `bid_price` at which bid is accepted
            """,
        )

    def route(self) -> None:
        if self.page == "Home":
            self._render_home()
        elif self.page == "Dashboard":
            self._render_dashboard()
        elif self.page == "Forecasting":
            self._render_forecasting()
        else:
            self._render_next_steps()
