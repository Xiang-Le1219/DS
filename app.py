"""
BMDS2003 Data Science - Deployment Prototype
Telcom Customer Churn Risk Scorer

Run locally:  streamlit run app.py

Layout and interaction live here; the palette is in theme.py, the computations
in analysis.py and the figures in charts.py. Every chart is generated live from
the data and the trained models - the notebook's static PNG exports are gone.
"""
import time

import numpy as np
import streamlit as st

import numpy as np
import streamlit as st

st.set_page_config(page_title="Telcom Churn Risk Scorer", page_icon="\U0001F4E1",
                   layout="wide")

import analysis as A
import charts as C
import theme as T

st.title("Telcom Customer Churn Risk Scorer")
st.caption("BMDS2003 Data Science - Group Assignment Deployment Prototype")

bundle = A.load_bundle()
df = A.load_cleaned()

# Overall churn rate, used as the baseline reference throughout the app.
BASELINE_CHURN = float((df["Churn"] == "Yes").mean())


def show(fig):
    """Render a Plotly figure. theme=None keeps our own styling from theme.py
    rather than letting Streamlit re-theme the chart on top of it."""
    st.plotly_chart(fig, width="stretch", theme=None, config=T.PLOTLY_CONFIG)


@st.cache_data
def compute_factor_rates(_df):
    """Historical churn rate for each risk-factor segment, computed from the data
    (leading underscore keeps Streamlit from trying to hash the DataFrame)."""
    def rate(mask):
        segment = _df[mask]
        return float((segment["Churn"] == "Yes").mean()) if len(segment) else 0.0

    return {
        "Month-to-month contract": rate(_df["Contract"] == "Month-to-month"),
        "Tenure in first 12 months": rate(_df["tenure"] <= 12),
        "Fiber optic internet": rate(_df["InternetService"] == "Fiber optic"),
        "Electronic check payment": rate(_df["PaymentMethod"] == "Electronic check"),
        "No online security & no tech support": rate(
            (_df["OnlineSecurity"] == "No") & (_df["TechSupport"] == "No")
        ),
    }


FACTOR_RATES = compute_factor_rates(df)


def render_speedometer_html(pct, bar_color, baseline_pct, animate=True):
    """A speedometer-style gauge as inline SVG: a semicircular green/amber/red
    dial with a needle at the predicted churn risk, plus a short dark tick
    marking the overall baseline churn rate.

    Built as SVG/CSS (via st.components.v1.html) rather than Matplotlib so the
    needle sweep, the fill arc and the percentage read-out can animate with
    real browser-native transitions - no per-frame server round trips.
    """
    cx, cy, r_outer, r_inner = 160, 150, 120, 86
    zones = [
        (0, T.MEDIUM_RISK_AT, T.ZONE_TINTS["low"]),
        (T.MEDIUM_RISK_AT, T.HIGH_RISK_AT, T.ZONE_TINTS["medium"]),
        (T.HIGH_RISK_AT, 100, T.ZONE_TINTS["high"]),
    ]

    def pt(pct_val, r):
        ang = np.radians(180 - (pct_val / 100) * 180)
        return cx + r * np.cos(ang), cy - r * np.sin(ang)

    def donut_path(lo, hi):
        x1, y1 = pt(lo, r_outer); x2, y2 = pt(hi, r_outer)
        x3, y3 = pt(hi, r_inner); x4, y4 = pt(lo, r_inner)
        return (f"M{x1:.2f},{y1:.2f} A{r_outer},{r_outer} 0 0,1 {x2:.2f},{y2:.2f} "
                f"L{x3:.2f},{y3:.2f} A{r_inner},{r_inner} 0 0,0 {x4:.2f},{y4:.2f} Z")

    zone_paths = "".join(
        f'<path d="{donut_path(lo, hi)}" fill="{color}" stroke="none"/>'
        for lo, hi, color in zones
    )
    boundary_lines = "".join(
        f'<line x1="{pt(v, r_inner)[0]:.2f}" y1="{pt(v, r_inner)[1]:.2f}" '
        f'x2="{pt(v, r_outer)[0]:.2f}" y2="{pt(v, r_outer)[1]:.2f}" '
        f'stroke="{T.TEXT}" stroke-width="2"/>'
        for v in [T.MEDIUM_RISK_AT, T.HIGH_RISK_AT]
    )
    tick_labels = "".join(
        f'<text x="{pt(v, r_outer + 18)[0]:.2f}" y="{pt(v, r_outer + 18)[1]:.2f}" '
        f'text-anchor="middle" dominant-baseline="middle" font-size="13" fill="{T.GRAY}">{v}%</text>'
        for v in [0, 33, 66, 100]
    )
    bx1, by1 = pt(baseline_pct, r_outer); bx2, by2 = pt(baseline_pct, r_outer + 12)
    baseline_tick = (f'<line x1="{bx1:.2f}" y1="{by1:.2f}" x2="{bx2:.2f}" y2="{by2:.2f}" '
                     f'stroke="{T.TEXT}" stroke-width="3" stroke-linecap="round"/>')

    arc_r = (r_outer + r_inner) / 2
    x1, y1 = pt(0, arc_r); x2, y2 = pt(100, arc_r)
    track_len = np.pi * arc_r
    fill_path = f'M{x1:.2f},{y1:.2f} A{arc_r},{arc_r} 0 0,1 {x2:.2f},{y2:.2f}'
    fill_len = (max(pct, 0) / 100) * track_len

    needle_len = r_inner * 0.98
    duration = 900 if animate else 0

    html = f"""
    <div style="display:flex;justify-content:center;">
    <svg viewBox="0 0 320 225" width="100%" style="max-width:420px;">
      {zone_paths}
      {boundary_lines}
      <path d="{fill_path}" fill="none" stroke="{bar_color}" stroke-width="{r_outer - r_inner:.0f}"
            stroke-dasharray="{track_len:.2f}" stroke-dashoffset="{track_len:.2f}"
            id="fillArc" style="transition: stroke-dashoffset {duration}ms cubic-bezier(0.22,1,0.36,1),
                                 stroke {duration}ms ease;"/>
      {tick_labels}
      {baseline_tick}
      <g id="needleGroup" style="transform-origin:{cx}px {cy}px;
                                  transform:rotate(0deg);
                                  transition: transform {duration}ms cubic-bezier(0.22,1,0.36,1);">
        <polygon points="{cx},{cy-7} {cx-needle_len:.1f},{cy} {cx},{cy+7}" fill="{T.TEXT}"/>
      </g>
      <circle cx="{cx}" cy="{cy}" r="9" fill="{T.TEXT}" stroke="white" stroke-width="1.5"/>
      <text id="pctText" x="{cx}" y="{cy+55}" text-anchor="middle"
            font-size="28" font-weight="bold" fill="{bar_color}">0.0%</text>
    </svg>
    </div>
    <script>
      (function() {{
        const finalPct = {pct};
        const needleAngle = (finalPct / 100) * 180;
        const fillOffset = {track_len:.2f} - {fill_len:.2f};
        const needle = document.getElementById("needleGroup");
        const arc = document.getElementById("fillArc");
        const label = document.getElementById("pctText");
        requestAnimationFrame(function() {{
          needle.style.transform = `rotate(${{needleAngle}}deg)`;
          arc.style.strokeDashoffset = fillOffset;
        }});
        const dur = {duration};
        if (dur === 0) {{
          label.textContent = finalPct.toFixed(1) + "%";
        }} else {{
          const start = performance.now();
          function tick(now) {{
            const t = Math.min((now - start) / dur, 1);
            const eased = 1 - Math.pow(1 - t, 3);
            label.textContent = (finalPct * eased).toFixed(1) + "%";
            if (t < 1) requestAnimationFrame(tick);
          }}
          requestAnimationFrame(tick);
        }}
      }})();
    </script>
    """
    return html


BEST_MODEL_NAME = bundle["best_model_name"]
AVAILABLE_MODELS = list(bundle["models"].keys())
FEATURES = bundle["feature_names"]

MODEL_DISPLAY_NAMES = {
    "Random Forest": "Random Forest (Best Model)",
    "Logistic Regression": "Logistic Regression (Baseline Model)",
}


def format_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# Material icons render inside segmented_control and pills option labels, which
# makes the profile form readable at a glance instead of a wall of dropdowns.
GENDER_ICONS = {"Female": ":material/face_3: Female", "Male": ":material/face: Male"}
INTERNET_ICONS = {
    "DSL": ":material/router: DSL",
    "Fiber optic": ":material/cable: Fiber optic",
    "No": ":material/wifi_off: None",
}
CONTRACT_ICONS = {
    "Month-to-month": ":material/event_repeat: Month-to-month",
    "One year": ":material/calendar_month: One year",
    "Two year": ":material/calendar_today: Two year",
}
# Shortened labels - the real dataset values are kept as the option values, so
# only the display text changes. Four full labels never fit on one row.
PAYMENT_ICONS = {
    "Bank transfer (automatic)": ":material/account_balance: Bank transfer",
    "Credit card (automatic)": ":material/credit_card: Credit card",
    "Electronic check": ":material/receipt_long: E-check",
    "Mailed check": ":material/mail: Mailed check",
}
# Binary attributes become chips: "which of these apply to this customer".
FLAG_ICONS = {
    "Senior citizen": ":material/elderly: Senior citizen",
    "Has partner": ":material/favorite: Has partner",
    "Has dependents": ":material/child_care: Has dependents",
    "Paperless billing": ":material/receipt: Paperless billing",
}
def _set_flags(values):
    st.session_state["p_flags"] = values


def _set_services(values):
    st.session_state["p_services"] = values


SERVICE_ICONS = {
    "Phone service": ":material/call: Phone service",
    "Multiple lines": ":material/dialpad: Multiple lines",
    "Online security": ":material/security: Online security",
    "Online backup": ":material/backup: Online backup",
    "Device protection": ":material/shield: Device protection",
    "Tech support": ":material/support_agent: Tech support",
    "Streaming TV": ":material/tv: Streaming TV",
    "Streaming movies": ":material/movie: Streaming movies",
}


DEFAULT_MODEL_NAME = "Random Forest" if "Random Forest" in AVAILABLE_MODELS else BEST_MODEL_NAME
ORDERED_MODELS = [DEFAULT_MODEL_NAME] + [
    model_name for model_name in AVAILABLE_MODELS if model_name != DEFAULT_MODEL_NAME
]

# Layout-only CSS.
#
# Everything to do with colour, corner radius and typography lives in
# .streamlit/config.toml. What is left here is the handful of *structural*
# rules Streamlit exposes no theme option for. Keeping this list short matters -
# these selectors target Streamlit's internal DOM, which is not a public API.
st.markdown(
    """
    <style>
    /* --- Full-width four-card model selector ------------------------- */
    div.st-key-model_selector div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        align-items: stretch !important;
        gap: 1rem !important;
    }

    div.st-key-model_selector div[data-testid="stColumn"] {
        flex: 1 1 0 !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    div.st-key-model_selector div[data-testid="stButton"],
    div.st-key-model_selector button {
        width: 100% !important;
        height: 100% !important;
    }

    div.st-key-model_selector button {
        min-height: 4rem !important;
        padding: 0.65rem 0.75rem !important;
        white-space: normal !important;
        line-height: 1.25 !important;
        font-weight: 600 !important;
    }

    div.st-key-model_selector button p {
        width: 100% !important;
        margin: 0 !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    /* Stack into a 2x2 grid, then a single column, on narrower screens. */
    @media (max-width: 900px) {
        div.st-key-model_selector div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }

        div.st-key-model_selector div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
        }
    }

    @media (max-width: 480px) {
        div.st-key-model_selector div[data-testid="stColumn"] {
            flex-basis: 100% !important;
            width: 100% !important;
        }
    }

    /* --- Equal-height "Customer profile" / "Result" columns ---------- */
    div.st-key-scorer_row div[data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
        align-items: stretch !important;
    }

    div.st-key-scorer_row div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }

    div.st-key-scorer_row div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
        flex: 1 1 auto !important;
        height: 100% !important;
    }

    /* --- Select-all / clear actions, shaped like the chips they act on
       (fully rounded, light weight, compact) so they read as part of the
       same chip cluster rather than as heavier form buttons. ----------- */
    div.st-key-flag_actions button,
    div.st-key-service_actions button {
        border-radius: 999px !important;
        min-height: 0 !important;
        padding: 0.3rem 0.85rem !important;
        font-weight: 400 !important;
    }

    /* --- Lift the Result panel off the page with a soft shadow ------- */
    div.st-key-result_panel {
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08) !important;
    }

    /* --- Centre the Matplotlib gauge and cap its width --------------- */
    div.st-key-gauge_chart div[data-testid="stVerticalBlock"] {
        display: flex;
        justify-content: center;
    }

    div.st-key-gauge_chart div[data-testid="stElementContainer"] {
        max-width: 380px;
        width: 100% !important;
        margin: 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def select_model(model_name):
    """Store the selected model so the highlighted choice persists after reruns."""
    st.session_state["selected_model_name"] = model_name


def highlight_model_rows(row):
    """Add distinct colours to the baseline and best-model rows."""
    if row["Model"] == "Logistic Regression (Baseline Model)":
        return [T.BASELINE_ROW_STYLE] * len(row)
    if row["Model"] == "Random Forest (Best Model)":
        return [T.BEST_ROW_STYLE] * len(row)
    return [""] * len(row)


tab_scorer, tab_model_evaluation, tab_eda = st.tabs([
    "Risk Scorer",
    "Model Evaluation",
    "Data Exploration",
])

# ----------------------------------------------------------------------
# TAB 1: Risk Scorer
# ----------------------------------------------------------------------
with tab_scorer:
    with st.container(border=True, key="model_panel"):
        st.markdown("#### 1. Choose a model")
        st.caption("Model used to score this customer")

        if st.session_state.get("selected_model_name") not in ORDERED_MODELS:
            st.session_state["selected_model_name"] = DEFAULT_MODEL_NAME

        with st.container(key="model_selector"):
            model_columns = st.columns([1, 1, 1, 1], gap="small")
            for index, (column, model_name) in enumerate(zip(model_columns, ORDERED_MODELS)):
                is_selected = st.session_state["selected_model_name"] == model_name
                with column:
                    st.button(
                        format_model_name(model_name),
                        key=f"model_choice_{index}",
                        type="primary" if is_selected else "secondary",
                        width="stretch",
                        on_click=select_model,
                        args=(model_name,),
                        help=f"Use {format_model_name(model_name)} to calculate the churn risk.",
                    )

    selected_model_name = st.session_state["selected_model_name"]
    model = bundle["models"][selected_model_name]

    with st.container(key="scorer_row"):
        left, right = st.columns([1, 1], gap="medium")

        with left:
            with st.container(border=True, key="profile_panel"):
                st.markdown("#### 2. Customer profile")

                # Two widget types only: segmented controls to pick one value
                # from a short list, pills to pick several. Dropdowns cost a
                # click to reveal options that fit on screen anyway, and a
                # toggle would be a third pattern for the same yes/no question
                # the service chips already ask.
                # Only controls that FIT a half-width column go side by side.
                # A segmented control that wraps onto a second row stops reading
                # as one group - the wrapped option looks like a separate widget -
                # so Contract (3 long labels) and Payment method (4 options) get
                # the full width instead.
                c1, c2 = st.columns(2)
                with c1:
                    tenure = st.slider("Tenure (months)", 0, 72, 12)
                with c2:
                    monthly = st.slider("Monthly charges (RM)", 18.0, 120.0, 70.0,
                                        step=0.5)

                c3, c4 = st.columns([1, 1.6])
                with c3:
                    gender = st.segmented_control(
                        "Gender", ["Female", "Male"], format_func=GENDER_ICONS.get,
                        default="Female", key="p_gender") or "Female"
                with c4:
                    internet = st.segmented_control(
                        "Internet service", ["DSL", "Fiber optic", "No"],
                        format_func=INTERNET_ICONS.get, default="DSL",
                        key="p_internet") or "DSL"

                contract = st.segmented_control(
                    "Contract", ["Month-to-month", "One year", "Two year"],
                    format_func=CONTRACT_ICONS.get, default="Month-to-month",
                    key="p_contract") or "Month-to-month"
                payment = st.segmented_control(
                    "Payment method", sorted(df["PaymentMethod"].unique()),
                    format_func=PAYMENT_ICONS.get,
                    default="Bank transfer (automatic)",
                    key="p_payment") or "Bank transfer (automatic)"

                # Select-all / clear write into the pills' session_state from an
                # on_click callback, which Streamlit runs before the widget is
                # rebuilt - the supported way to set a widget's value in code.
                #
                # Two things this deliberately avoids. A horizontal container
                # flows the caption and both actions on one line at their natural
                # widths, so there are no column ratios to get wrong at different
                # window sizes. And the callbacks replace an explicit st.rerun():
                # rerunning mid-render tore each button out from under its own
                # tooltip, which then had nothing left to fire a mouse-leave on
                # and stayed stuck on screen after the click.
                with st.container(horizontal=True, vertical_alignment="center",
                                  key="flag_actions"):
                    st.caption("Select any that apply (multi-select)")
                    st.button("Select all", key="flags_all",
                              icon=":material/done_all:", wrap=False,
                              on_click=_set_flags, args=(list(FLAG_ICONS),))
                    st.button("Clear", key="flags_clear",
                              icon=":material/close:", wrap=False,
                              on_click=_set_flags, args=([],))
                customer_flags = st.pills(
                    "Which of these apply to this customer?", list(FLAG_ICONS),
                    format_func=FLAG_ICONS.get, selection_mode="multi",
                    default=[], key="p_flags", label_visibility="collapsed") or []

                with st.container(horizontal=True, vertical_alignment="center",
                                  key="service_actions"):
                    st.caption("Services subscribed (multi-select)")
                    st.button("Select all", key="services_all",
                              icon=":material/done_all:", wrap=False,
                              on_click=_set_services, args=(list(SERVICE_ICONS),))
                    st.button("Clear", key="services_clear",
                              icon=":material/close:", wrap=False,
                              on_click=_set_services, args=([],))
                services = st.pills(
                    "Services subscribed", list(SERVICE_ICONS),
                    format_func=SERVICE_ICONS.get, selection_mode="multi",
                    default=["Phone service"], key="p_services",
                    label_visibility="collapsed") or []

                yn = lambda flag: "Yes" if flag else "No"
                total = round(tenure * monthly, 2)

                raw = {
                    "gender": gender,
                    "SeniorCitizen": yn("Senior citizen" in customer_flags),
                    "Partner": yn("Has partner" in customer_flags),
                    "Dependents": yn("Has dependents" in customer_flags),
                    "tenure": tenure,
                    "PhoneService": yn("Phone service" in services),
                    "MultipleLines": yn("Multiple lines" in services),
                    "InternetService": internet,
                    "OnlineSecurity": yn("Online security" in services),
                    "OnlineBackup": yn("Online backup" in services),
                    "DeviceProtection": yn("Device protection" in services),
                    "TechSupport": yn("Tech support" in services),
                    "StreamingTV": yn("Streaming TV" in services),
                    "StreamingMovies": yn("Streaming movies" in services),
                    "Contract": contract,
                    "PaperlessBilling": yn("Paperless billing" in customer_flags),
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                }

                st.caption(f"Estimated total charges: RM {total:,.2f}  (tenure x monthly charges)")

                if st.button("Score this customer", type="primary", width="stretch"):
                    row = A.encode_profile(raw, bundle)
                    row = A.scaled(row, selected_model_name)

                    proba = float(model.predict_proba(row)[0, 1])
                    pred = int(model.predict(row)[0])

                    flags = []
                    if contract == "Month-to-month":
                        flags.append("Month-to-month contract")
                    if tenure <= 12:
                        flags.append("Tenure in first 12 months")
                    if internet == "Fiber optic":
                        flags.append("Fiber optic internet")
                    if payment == "Electronic check":
                        flags.append("Electronic check payment")
                    if ("Online security" not in services
                            and "Tech support" not in services):
                        flags.append("No online security & no tech support")

                    peers = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
                    peer_rate = float((peers["Churn"] == "Yes").mean()) if len(peers) > 0 else 0.0

                    # Stored in session_state so the result survives reruns and can
                    # render in the right-hand column, which is built first.
                    st.session_state["result"] = {
                        "model_name": selected_model_name,
                        "proba": proba,
                        "pred": pred,
                        "flags": flags,
                        "factor_rates": FACTOR_RATES,
                        "peer_count": len(peers),
                        "peer_rate": peer_rate,
                        "contract": contract,
                        "internet": internet,
                        "raw": raw,
                    }
                    # Consumed once by the Result panel below to decide whether
                    # to sweep the gauge from 0 or just draw the final value -
                    # every OTHER rerun (changing a slider elsewhere, etc.)
                    # should redraw the stored result statically, not replay it.
                    st.session_state["just_scored"] = True

        with right:
            with st.container(border=True, key="result_panel"):
                st.markdown("#### 3. Result")
                result = st.session_state.get("result")
                # Consumed (popped) so the sweep animation plays exactly once,
                # on the run where Score was actually clicked - not on every
                # later rerun the app happens to do for an unrelated reason.
                just_scored = st.session_state.pop("just_scored", False)

                if result:
                    st.caption(f"Prediction generated using "
                               f"**{format_model_name(result['model_name'])}**")
                else:
                    st.caption(
                        "No prediction yet - fill in the profile on the left and "
                        "click **Score this customer**."
                    )

                pct = result["proba"] * 100 if result else 0.0
                band, bar_color = T.risk_band(pct) if result else (None, T.GRAY)

                k1, k2, k3 = st.columns(3)
                if result:
                    k1.metric("Churn risk", f"{pct:.1f}%",
                              f"{pct - BASELINE_CHURN * 100:+.1f} pts vs baseline",
                              delta_color="inverse")
                    k2.metric("Risk band", band.upper())
                    k3.metric("Similar customers", f"{result['peer_rate']:.1%}",
                              f"{result['peer_count']:,} in data", delta_color="off")
                else:
                    k1.metric("Churn risk", "-")
                    k2.metric("Risk band", "-")
                    k3.metric("Similar customers", "-")

                with st.container(key="gauge_chart"):
                    st.components.v1.html(
                        render_speedometer_html(pct, bar_color, BASELINE_CHURN * 100,
                                                animate=just_scored),
                        height=280,
                    )

                if result:
                    st.caption(f"Dark tick marks the overall baseline churn rate "
                               f"({BASELINE_CHURN:.1%}).")

                    if band == "high":
                        st.error("**HIGH RISK - flag for retention contact.**")
                    elif band == "medium":
                        st.warning("**MEDIUM RISK - monitor customer closely.**")
                    else:
                        st.success("**LOW RISK - no retention action required.**")

                    flags = result["flags"]
                    if flags:
                        st.markdown("**Risk factors present in this profile** "
                                    "(from the notebook's EDA):")
                        rates = [result["factor_rates"][f] for f in flags]
                        show(C.risk_factor_bar(flags, rates, BASELINE_CHURN))
                    else:
                        st.info("No major risk factor from the top drivers "
                                "identified in the EDA.")
                else:
                    st.caption(f"Dark tick marks the overall baseline churn rate "
                               f"({BASELINE_CHURN:.1%}).")
                    st.info(
                        "Fill in the customer profile on the left and click "
                        "**Score this customer** to see the risk result here."
                    )

    # ------------------------------------------------------------------
    # What-if analysis - the score alone is a dead end; this makes it explorable.
    # ------------------------------------------------------------------
    result = st.session_state.get("result")
    if result:
        # ------------------------------------------------------------------
        # Why this score - the risk-factor chart above shows how often the
        # SEGMENTS this customer belongs to churn historically. That is a
        # population average, not the model's reasoning about this individual.
        # This section answers the latter.
        # ------------------------------------------------------------------
        with st.container(border=True, key="explain_panel"):
            st.markdown("#### 4. Why this score?")
            breakdown, actual = A.contribution_breakdown(result["raw"],
                                                         result["model_name"])
            levers, current_risk = A.retention_levers(result["raw"],
                                                      result["model_name"])

            why_col, lever_col = st.columns(2, gap="medium")
            with why_col:
                st.markdown("**What makes this customer different**")
                if breakdown.empty:
                    st.info("This customer matches the typical profile on every "
                            "attribute, so nothing stands out to explain.")
                else:
                    show(C.contribution_chart(breakdown, actual))
                    st.caption(
                        "Each attribute is swapped to the most common value in the "
                        "dataset and the customer re-scored; the bar is the "
                        "difference. Attributes already at the typical value are "
                        "omitted because they change nothing."
                    )
            with lever_col:
                st.markdown("**What would help most**")
                best = levers.iloc[0]
                if best["Reduction"] > 0:
                    st.success(
                        f"**Biggest lever:** {best['Field']} -> "
                        f"**{best['Change to']}** would take risk from "
                        f"{current_risk:.1%} to **{best['New risk']:.1%}** "
                        f"({best['Reduction'] * 100:.1f} pts lower)."
                    )
                else:
                    st.info("No single change reduces this customer's risk.")
                show(C.lever_chart(levers, current_risk))
                st.caption(
                    "Every single change a retention team could make, re-scored. "
                    "Read these as risk, not profit: dropping a service often "
                    "lowers churn risk while also removing the revenue you were "
                    "trying to keep."
                )

        # ------------------------------------------------------------------
        # What-if
        # ------------------------------------------------------------------
        with st.container(border=True, key="whatif_panel"):
            st.markdown("#### 5. What-if analysis")
            st.caption(
                "Holds the rest of this customer's profile fixed and re-scores them "
                "across other values. The marker shows where they sit today."
            )
            numeric_tab, categorical_tab = st.tabs(["Sweep a number",
                                                    "Compare the options"])
            with numeric_tab:
                sweep_feature = st.segmented_control(
                    "Sweep which feature?", ["tenure", "MonthlyCharges"],
                    format_func=lambda f: "Tenure (months)" if f == "tenure"
                    else "Monthly charges (RM)",
                    default="tenure", key="sweep_feature") or "tenure"
                current = result["raw"][sweep_feature]
                show(C.sensitivity(result["raw"], result["model_name"],
                                   sweep_feature, current))
            with categorical_tab:
                # Contract is the strongest driver in the whole model and is
                # categorical, so a numeric sweep can never reach it.
                cat_field = st.segmented_control(
                    "Feature", list(A.ACTIONABLE_FIELDS),
                    format_func=lambda f: A.FIELD_LABELS.get(f, f),
                    default="Contract", key="whatif_field") or "Contract"
                option_risks = A.categorical_whatif(result["raw"],
                                                    result["model_name"], cat_field)
                show(C.categorical_whatif_chart(
                    option_risks, A.FIELD_LABELS.get(cat_field, cat_field),
                    result["raw"][cat_field]))

    st.divider()
    st.caption(
        "Academic prototype trained on 7,043 historical customer records. It outputs a churn "
        "RISK SCORE for retention triage, not a prediction about an individual's intentions. "
        "The associations shown are correlational, not causal, and the model must be retrained "
        "as customer behaviour and tariffs change. Not for real commercial or financial decisions."
    )

# ----------------------------------------------------------------------
# TAB 2: Model Evaluation
# ----------------------------------------------------------------------
with tab_model_evaluation:
    st.caption("Performance and diagnostics for all four trained models, computed live "
               "on the notebook's held-out test set (1,409 customers).")

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, step=0.01,
        help="A customer is flagged as a churn risk when their predicted probability "
             "is at or above this value. The notebook's published metrics all assume 0.50.",
    )
    if abs(threshold - 0.50) > 1e-9:
        st.info(f"Showing metrics at a **{threshold:.2f}** threshold. "
                "The report's published figures assume 0.50.")

    st.subheader("Metrics table")
    live = A.live_metrics(threshold).rename(index={
        "Logistic Regression": "Logistic Regression (Baseline Model)",
        "Random Forest": "Random Forest (Best Model)",
    })
    display_results = live.reset_index().rename(columns={"index": "Model"})
    st.dataframe(
        display_results.style.apply(highlight_model_rows, axis=1).format(precision=3),
        width="stretch", hide_index=True,
    )

    published = A.load_published_metrics()
    if published is not None and abs(threshold - 0.50) < 1e-9:
        recomputed = A.live_metrics(0.5)
        largest_gap = float((recomputed[published.columns] - published).abs().to_numpy().max())
        st.caption(f"Cross-check: these live numbers match "
                   f"`model_comparison_results.csv` to within {largest_gap:.6f}.")

    st.subheader("Overall comparison")
    all_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    chosen_metrics = st.multiselect("Metrics to compare", all_metrics,
                                    default=all_metrics)
    if chosen_metrics:
        show(C.model_comparison(chosen_metrics, threshold))
    else:
        st.info("Select at least one metric to compare.")

    st.subheader("ROC curves")
    st.caption("Click a model in the legend to hide it, or double-click to isolate it.")
    show(C.roc_curves())

    st.subheader("Confusion matrix and the threshold trade-off")
    cm_model = st.segmented_control(
        "Model", ORDERED_MODELS, format_func=format_model_name,
        default=DEFAULT_MODEL_NAME, key="cm_model",
    ) or DEFAULT_MODEL_NAME
    left_cm, right_cm = st.columns([1, 1], gap="medium")
    with left_cm:
        show(C.confusion_heatmap(cm_model, threshold))
        stats = A.metrics_at_threshold(cm_model, threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{stats['precision']:.3f}")
        m2.metric("Recall", f"{stats['recall']:.3f}")
        m3.metric("F1", f"{stats['f1']:.3f}")
    with right_cm:
        show(C.threshold_tradeoff(cm_model, threshold))
        ceiling = A.max_probability(cm_model)
        st.caption(
            "Lowering the threshold catches more churners (recall up) at the cost of "
            "more false alarms (precision down). For retention triage, recall usually "
            f"matters more. This model never predicts above {ceiling:.2f}, so past that "
            "point it flags nobody and precision becomes undefined rather than zero - "
            "which is why the precision line stops instead of dropping to the floor."
        )

    st.subheader("Cross-validation stability")
    cv_left, cv_right = st.columns([2, 1])
    cv_metric = cv_left.segmented_control(
        "Scoring metric", ["f1", "roc_auc", "accuracy"], default="f1", key="cv_metric",
    ) or "f1"
    show_folds = cv_right.toggle("Show individual folds", key="cv_folds")
    with st.spinner(f"Running {A.CV_FOLDS}-fold cross-validation ({cv_metric})..."):
        show(C.cv_stability(cv_metric, show_folds))
    st.caption("Bar height is the mean across folds; the error bar is +/- 1 standard "
               "deviation. A shorter error bar means a more reliable model - the score "
               "does not depend on one lucky split.")

    st.subheader("Overfitting check")
    show(C.overfitting_check())
    st.caption("A small train-test gap suggests the model generalises rather than "
               "memorising the training data.")

# ----------------------------------------------------------------------
# TAB 3: Data Exploration
# ----------------------------------------------------------------------
with tab_eda:
    sub_eda, sub_corr, sub_importance = st.tabs([
        "Exploratory Data Analysis",
        "Correlation Analysis",
        "Feature Importance",
    ])

    with sub_eda:
        quality = A.data_quality()
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Customers", f"{quality['clean_rows']:,}")
        q2.metric("Features", quality["columns"])
        q3.metric("Missing values", quality["missing"])
        q4.metric("Duplicate rows", quality["duplicates"])
        st.caption("Data-preparation checks: the cleaned dataset has no missing values "
                   "and no duplicates, so the models train on complete records.")

        st.divider()
        st.subheader("Preparation evidence and the target")
        prep_col, donut_col = st.columns([2.8, 1], gap="medium")
        with prep_col:
            st.caption("SMOTE is genuinely re-run here on the training split only, and "
                       "the scaler re-applied, so these are measured results rather "
                       "than quoted ones.")
            try:
                with st.spinner("Re-running SMOTE on the training split..."):
                    show(C.preparation_evidence())
            except ModuleNotFoundError:
                # This chart is the only thing needing imbalanced-learn. Rather than
                # take the page down - or quietly print numbers we did not actually
                # compute - say plainly what is missing.
                st.info(
                    "This chart re-runs SMOTE for real, which needs the "
                    "`imbalanced-learn` package:\n\n"
                    "```\npip install imbalanced-learn==0.14.0\n```\n\n"
                    "It is already listed in `requirements.txt`, so Streamlit Cloud "
                    "installs it automatically. Everything else here works without it."
                )
        with donut_col:
            st.caption("The target itself: about one customer in four churned, which is "
                       "why SMOTE is needed on the left.")
            show(C.churn_donut())

        st.divider()
        st.subheader("Churn by category")
        cat_feature = st.selectbox(
            "Feature", A.CATEGORICAL_FEATURES, key="cat_feature",
            help="Drives both charts below.",
        )
        bar_col, mosaic_col = st.columns(2, gap="medium")
        with bar_col:
            st.markdown("**Churn rate**")
            show(C.churn_rate_bar(cat_feature))
            st.caption("How likely each group is to churn.")
        with mosaic_col:
            st.markdown("**Mosaic view**")
            show(C.mosaic(cat_feature))
            st.caption("Column width is customer volume, so a wide red column is a "
                       "large *and* high-risk segment - rate alone cannot show that.")

        st.subheader("Which categorical features matter most?")
        st.caption("Cramer's V measures how strongly each categorical feature is "
                   "associated with churn (0 = none, 1 = perfect).")
        show(C.cramers_v_lollipop())

        st.subheader("Numeric feature distributions")
        num_feature = st.segmented_control(
            "Numeric feature", A.NUMERIC_FEATURES,
            default=A.NUMERIC_FEATURES[0], key="num_feature",
        ) or A.NUMERIC_FEATURES[0]
        violin_col, ecdf_col = st.columns(2, gap="medium")
        with violin_col:
            show(C.numeric_violin(num_feature))
            st.caption("Distribution by churn status, with the mean line (dashed) "
                       "and the median called out for each group.")
        with ecdf_col:
            show(C.ecdf(num_feature))
            st.caption("Where each curve crosses the 50% line is that group's median - "
                       "the actionable threshold the report reads off this chart.")

        st.subheader("Where does risk compound?")
        st.caption("Churn rate for every combination of two features. Pick any pair. "
                   "Green is a low-churn segment and red a high-churn one, on a fixed "
                   "0-60% scale so a colour means the same thing across every pairing.")
        i1, i2 = st.columns(2)
        row_feature = i1.selectbox("Rows", A.CATEGORICAL_FEATURES, index=0, key="int_row")
        col_feature = i2.selectbox("Columns", A.CATEGORICAL_FEATURES, index=1, key="int_col")
        if row_feature == col_feature:
            st.warning("Pick two different features to see an interaction.")
        else:
            show(C.interaction_heatmap(row_feature, col_feature))

        st.subheader("Services subscribed vs churn")
        show(C.services_dual_axis())
        st.caption("Churn falls steadily as customers add services, even though most "
                   "customers sit at the low end of the service count.")

        st.subheader("Numeric features together")
        splom_features = st.multiselect(
            "Features in the matrix", A.NUMERIC_FEATURES,
            default=A.NUMERIC_FEATURES,
            help="All four engineered numeric features. Figure 8 in the report "
                 "uses the first three; drop num_services to match it exactly.",
        )
        if len(splom_features) < 2:
            st.info("Pick at least two features to build the matrix.")
        else:
            show(C.scatter_matrix(splom_features))
            st.caption("Scatters use a 1,200-customer sample so the chart stays "
                       "responsive; the diagonal shows each feature's full "
                       "distribution split by churn. Numeric features alone only "
                       "partly separate the two groups - which is why the categorical "
                       "and engineered features are still needed.")

    with sub_corr:
        st.subheader("Correlation matrix")
        top_n = st.slider("Number of top features", 5, 27, 15, key="corr_top")
        show(C.correlation_heatmap(top_n))
        st.caption("Features ranked by absolute correlation with churn. Deep blue is a "
                   "positive relationship, red a negative one.")

        st.subheader("Ranked correlation with churn")
        ranked_n = st.slider("Features to show", 6, 27, 27, key="rank_top",
                             help="All 27 encoded features, as in Figure 9c.")
        show(C.correlation_ranked(ranked_n))
        st.caption("Two-year contracts and long tenure pull churn down; month-to-month "
                   "contracts and fiber optic push it up.")

    with sub_importance:
        imp_model = st.segmented_control(
            "Model", ORDERED_MODELS, format_func=format_model_name,
            default=DEFAULT_MODEL_NAME, key="imp_model",
        ) or DEFAULT_MODEL_NAME
        imp_n = st.slider("Features to show", 5, 27, 15, key="imp_top")

        st.subheader("Model-native importance")
        compare_all = st.toggle("Compare all four models", key="imp_compare",
                                help="The four-panel view from Figure 15.")
        if compare_all:
            show(C.importance_grid(min(imp_n, 10)))
            st.caption("Top features per model. Contract, tenure and internet service "
                       "surface in all four - that agreement is the real finding.")
        else:
            show(C.importance_bar(imp_model, imp_n))
            st.caption("How much each feature contributes inside the model itself. "
                       "Logistic Regression coefficients are signed - red raises churn "
                       "risk, blue lowers it - while tree importances are always "
                       "positive, so only their magnitude is meaningful.")

        st.subheader("Permutation importance")
        with st.spinner("Shuffling each feature to measure its contribution..."):
            show(C.permutation_bar(imp_model, imp_n))
        st.caption("Measured on the held-out test set: how much ROC-AUC is lost when a "
                   "feature is randomly shuffled, with the value printed on each bar. "
                   "Error bars show the spread over repeats.")

        st.subheader("Correlation vs learned importance")
        show(C.correlation_vs_importance(imp_model, imp_n))
        st.caption("A sanity check - when the model's ranking agrees with the raw "
                   "correlations, it is picking up genuine signal rather than noise. "
                   "The number in parentheses is how many ranks a feature moved "
                   "between the correlation ranking and the model's importance ranking.")
