from datetime import UTC, date, datetime, time, timedelta

import streamlit as st

class App:
    def __init__(self) -> None:
        self.set_page_config()
        self.PAGES = ("Home", "Dashboard", "Next steps")
        self.page = "Home"
        self.render_sidebar()
        self.route() # fire on every script reload

    def set_page_config(self) -> None:
        st.set_page_config(
            page_title="DFS Analysis – Take-home",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_sidebar(self) -> None:

        with st.sidebar:
            st.title("DFS Analysis")
            self.page = st.radio("Navigate", self.PAGES, index=0)
            st.markdown("---")
            st.caption("Quick controls (global)")

            # Global date window (used by Dashboard)
            today = datetime.now(tz=UTC).date()
            default_start = today - timedelta(days=120)
            date_range = st.date_input(
                "Date range",
                (default_start, today),
                help="Filters for the historic charts on the Dashboard.",
            )
            st.session_state["date_range"] = date_range

            # Morning-of forecast time (for your within-day logic)
            forecast_time = st.time_input(
                "Morning snapshot time",
                value=time(10, 0),
                help="Use this to pull morning-available signals for same-day forecasting.",
            )
            st.session_state["forecast_time"] = forecast_time

            st.markdown("---")
            st.caption("Data sources (you can wire these up later)")
            use_live_sources = st.checkbox("Use live APIs when available", value=False)
            st.session_state["use_live_sources"] = use_live_sources


    # ---------- Helpers (stubs you can fill in) ----------
    def load_historic_data(self, start: date, end: date, use_live: bool = False) -> dict:
        """TODO: Replace with your actual loaders.
        Return a dict with keys you’ll chart on the Dashboard.
        """
        # e.g. dfs_events, dfs_utilisation, drm, lolp, ic_flows, system_price, niv
        return {
            "dfs_events": None,
            "dfs_utilisation": None,
            "drm": None,
            "lolp": None,
            "ic_flows": None,
            "system_price": None,
            "niv": None,
        }


    def render_placeholder_plot(self, title: str, help_text: str = "") -> None:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if help_text:
                st.caption(help_text)
            st.info(
                "Plot placeholder – insert your Plotly figure here.\n\n"
                "Example: `st.plotly_chart(fig, use_container_width=True)`",
                icon="🛠️",
            )


    def render_home(self) -> None:
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


    def render_dashboard(self) -> None:
        st.title("Dashboard")
        st.caption("Historic context and morning-of snapshot")

        start, end = st.session_state.get("date_range", (None, None))
        use_live = st.session_state.get("use_live_sources", False)
        data = self.load_historic_data(start, end, use_live)

        # --- Row 1: DFS + System Tightness ---
        st.subheader("Events & System Tightness")
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            self.render_placeholder_plot(
                "DFS events / utilisation",
                "Event timeline, accepted MW, and marginal accepted price per event window.",
            )

        with c2:
            self.render_placeholder_plot(
                "Derated Margin (DRM) & LoLP",
                "Overlay DRM & LoLP around evening SPs; consider max/mean in the peak block.",
            )

        with c3:
            self.render_placeholder_plot(
                "Demand forecast vs. outturn (optional)",
                "Helps explain tightness drivers alongside DRM/LoLP.",
            )

        st.markdown("---")

        # --- Row 2: Interconnectors + Prices/NIV ---
        st.subheader("Interconnectors & Imbalance Context")
        d1, d2, d3 = st.columns(3, gap="large")

        with d1:
            self.render_placeholder_plot(
                "Interconnector stance",
                "DA scheduled flows, % of technical capacity, outages; highlight full-import periods.",
            )

        with d2:
            self.render_placeholder_plot(
                "System price",
                "Historic imbalance price distribution around DFS events (back-cast only).",
            )

        with d3:
            self.render_placeholder_plot(
                "Net Imbalance Volume (NIV)",
                "NIV patterns on event vs non-event days; use descriptively, not as a forward input at 10:00.",
            )

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
                    """,
        )

    def route(self) -> None:
        if self.page == "Home":
            self.render_home()
        elif self.page == "Dashboard":
            self.render_dashboard()
        else:
            self.render_next_steps()
