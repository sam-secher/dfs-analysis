from datetime import UTC, date, datetime
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit.elements.plotly_chart import PlotlyState

from model.forecasting import DFSForecastingModel
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

    def _render_dfs_detail(self, dfs_data: pd.DataFrame) -> None:
        with st.container(border=True):
            ts = st.session_state["dfs_selected_ts"]
            st.subheader(f"DFS results - {ts:%H:%M} UTC (SP {datetime_to_sp(ts)})")
            detail = dfs_data.loc[dfs_data["datetime"] == ts].sort_values("offered_price", ascending=False)
            detail = detail[["dfs_unit_id", "dfs_participant", "offered_volume_mw", "offered_price", "offer_status"]]
            detail.columns = ["Unit ID", "Participant", "Volume (MW)", "Price (GBP/MWh)", "Status"]

            st.dataframe(detail, hide_index=True, width="stretch")

            # Provide an explicit way back to the chart
            if st.button("Back to chart"):
                st.session_state["dfs_selected_ts"] = None
                st.rerun()

    def _render_chart(self, fig: go.Figure, title: str, help_text: str = "", event_on_select: bool = False) -> PlotlyState:
        with st.container(border=True):
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
        st.title("Demand Flexibility Service (DFS) – Analysis Workspace")
        st.subheader("Take-home overview")

        st.markdown(
            """
            This is a compact workspace to explore **historic DFS activity**, track **system tightness** signals
            (e.g., DRM/LoLP), and prototype a **same-day (10:00) price/volume view**.

            ### What’s in here
            - **Dashboard**: quick plots of historic DFS events, derated margin & LoLP, interconnector stance, and system price/NIV context.
            - **Next steps**: concrete tasks to evolve this into a minimal forecasting + bidding aid.

            ### How I’ll use it in the interview
            1. Show **event frequency & price distributions**, then overlay system tightness indicators to ground the narrative.
            2. Demonstrate a **morning-of** workflow: ingest 10:00 snapshots, produce an **event likelihood** and a **marginal price band**.
            3. Outline a **stack simulation** concept (DFS bids vs. alternatives) and how I’d productionise.
            """,
        )

        with st.expander("Assumptions & scope", expanded=True):
            st.markdown(
                """
                - Focus is the **evening peak**; DFS remains a merit-based margin tool.
                - **Pay-as-bid** dynamics imply we care about the **marginal accepted price** per event.
                - Inputs available by ~10:00 (LoLP/DRM, DA interconnector stance, demand forecast) are the basis for same-day guidance.
                - This project is intentionally scoped for a **short take-home**: clarity over complexity.
                """,
            )

        st.markdown("---")
        st.markdown("#### Quick start")
        st.markdown(
            """
            - Set a **date range** in the sidebar → open **Dashboard**.
            - Replace the chart placeholders with your **Plotly** figures.
            - Wire your **data loaders** and (optionally) a simple **logit + regression** or a **stack simulator**.
                    """,
        )


    def _render_dashboard(self) -> None:
        st.title("Dashboard")
        st.caption("DFS event viewer and system tightness indicators")

        dfs_event_dates = self._model._dfs_event_dates()
        label_to_date = { date.strftime("%d/%m/%Y"): date for date in dfs_event_dates }
        labels = list(label_to_date.keys())
        # end = max(dfs_event_dates)
        date_chosen_str = st.select_slider(
            "Date picker",
            options=labels,
            value=labels[-1],
            help="Only dates with DFS used are shown."
        )
        date_chosen = label_to_date[date_chosen_str]

        st.subheader(date_chosen.strftime("%d/%m/%Y"))

        dfs_data, timeseries_df = self._load_historic_data(date_chosen)

        c1, c2 = st.columns(2, gap="large")

        with c1:

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

        with c2:
            pass

        # with c3:
        #     self.render_placeholder_plot(
        #         "Demand forecast vs. outturn (optional)",
        #         "Helps explain tightness drivers alongside DRM/LoLP.",
        #     )

        st.markdown("---")

        # --- Row 2: Interconnectors + Prices/NIV ---
        st.subheader("Interconnectors & Imbalance Context")
        d1, d2 = st.columns(2, gap="large")

        with d1:
            pass

        with d2:
            pass

        st.markdown("---")

        # --- Optional: Morning-of snapshot panel (inputs you’ll wire) ---
        with st.container(border=True):
            st.subheader("Morning-of view (10:00)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Peak-block LoLP (max)", "—")
            with col_b:
                st.metric("Peak-block DRM (min)", "—")
            with col_c:
                st.metric("DA net imports @ peak", "—")

            st.info(
                "Hook these metrics to your 10:00 snapshot. If you add a model, show "
                "event probability and a price band here, plus a short ‘reason code’.",
                icon="💡",
            )

    def render_forecasting(self) -> None:
        st.title("Forecasting")

    def render_next_steps(self) -> None:
        st.title("Next steps")
        st.markdown(
            """
            Below are concrete, time-boxed steps to turn this into a useful forecasting & bidding aid.

            ### 1) Minimal data pipeline (half-day)
            - Build loaders for: **DFS Utilisation/Summary**, **DRM/LoLP**, **DA interconnector schedules**, and (historical) **system price/NIV**.
            - Normalize to **settlement period granularity**. Create an **event-window table** (evening blocks).

            ### 2) First-pass modelling (half-day)
            - **Event probability**: logistic regression using 10:00-available features (LoLP/DRM peak-block stats, interconnector stance, calendar).
            - **Marginal price** (conditional on event): simple linear or shallow tree regressor.
            - Report calibration (events) and MAE/RMSE (price). Keep it honest about data sparsity.

            ### 3) (Optional) Stack simulation (1–2 half-days)
            - Learn a **conditional DFS offer curve** (MW per price band) from history.
            - Forecast **procured MW** from tightness signals.
            - **Clear** cumulative DFS supply vs. required MW to get a **marginal accepted price**.

            ### 4) Streamlit polish (couple of hours)
            - Replace placeholders with **Plotly** figures.
            - Add a **morning-of panel**: event probability, price band, and a short textual rationale.
            - Add a simple **revenue calculator**: `0.5 × P_bid × Q × K` per SP, plus uncertainty bands.

            ### Notes on scope & realism
            - Keep to **interpretable** models with clear feature provenance.
            - Emphasise **data timing**: only use inputs **available by ~10:00** for same-day guidance.
            - Treat imbalance price/NIV as **historical context**, not day-ahead inputs.
            - Incorporate weather and demand forecasts as additional inputs.
            - Consider developing a stack model to compare DFS bids with alternatives.
                    """,
        )

    def route(self) -> None:
        if self.page == "Home":
            self._render_home()
        elif self.page == "Dashboard":
            self._render_dashboard()
        elif self.page == "Forecasting":
            self.render_forecasting()
        else:
            self.render_next_steps()
