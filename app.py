"""
app.py
--------
🌸 AI Flower Classification System — Premium Streamlit Web Application (v2).

A modern AI SaaS-style dashboard for classifying Iris flowers using
K-Nearest Neighbors (and comparative models), with live predictions,
confidence scoring, interactive visualizations, model comparison, and
a portfolio-grade About page.

All machine learning logic (data loading, preprocessing, training,
evaluation, prediction) lives untouched in `src/` — this module is
presentation-only.

Run with:
    streamlit run app.py

Author:   Rocky Dutta
"""

from __future__ import annotations

import os
import random
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# --------------------------------------------------------------------
# ML logic imports — UNCHANGED, sourced entirely from src/
# --------------------------------------------------------------------
from data_loader import FEATURE_NAMES, TARGET_NAMES, load_iris_dataframe  # noqa: E402
from predict import predict_species  # noqa: E402
from utils import load_metrics, load_model  # noqa: E402

# =====================================================================
# CONSTANTS, DEVELOPER INFO & CONFIG
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "css", "style.css")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

APP_VERSION = "v2.0.0"

NAV_PAGES = ["🔮 Predict", "📊 Dataset & EDA", "ℹ️ About"]

ML_TIPS = [
    "KNN classifies a point by majority vote of its K nearest neighbors.",
    "Feature scaling matters most for distance-based algorithms like KNN.",
    "Cross-validation gives a more honest estimate of accuracy than a single split.",
    "A confusion matrix reveals *which* classes a model confuses — accuracy alone hides this.",
    "Stratified splits keep class balance consistent between train and test sets.",
    "Low K in KNN can overfit; high K can underfit — tuning finds the sweet spot.",
    "F1 score balances precision and recall — useful when classes are imbalanced.",
    "The Iris dataset has been a machine learning benchmark since 1936.",
]

SKILLS = [
    "Python", "Machine Learning", "Scikit-Learn", "Pandas", "NumPy",
    "Streamlit", "Data Visualization", "Plotly", "Model Evaluation", "EDA",
]

PIPELINE_STEPS = [
    "Load Data", "Clean & EDA", "Train/Test Split", "Feature Scaling",
    "Hyperparameter Tuning", "Model Comparison", "Evaluation", "Deployment",
]

FUTURE_IMPROVEMENTS = [
    "Batch CSV predictions for multiple flowers at once",
    "SHAP-based model explainability for individual predictions",
    "FastAPI REST layer alongside the Streamlit UI",
    "Docker containerization for consistent deployment",
    "CI/CD pipeline via GitHub Actions to run tests on every push",
]

PROJECT_FEATURES = [
    "🔮 Live prediction with confidence scoring",
    "📊 Full exploratory data analysis suite",
    "🏆 Multi-model comparison with cross-validation",
    "📈 Interactive Plotly visualizations",
    "🕒 Session prediction history with CSV export",
    "🌓 Dark / light theme with URL-based persistence",
]

st.set_page_config(
    page_title="AI Flower Classification System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# REUSABLE HELPER FUNCTIONS (presentation layer only)
# =====================================================================
def load_css() -> None:
    """Inject the premium stylesheet into the Streamlit app."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def apply_theme_variables(theme: str) -> None:
    """Override root CSS variables for light mode.

    Toggling a DOM class via an injected <script> is unreliable inside
    Streamlit (script tags inserted through st.markdown are not executed
    by the browser). Overriding the CSS custom properties directly on
    :root via a second <style> block is 100% CSS-driven and always works.
    """
    if theme == "light":
        st.markdown(
            """
            <style>
            :root {
                --bg-gradient: linear-gradient(160deg, #FDFBF6 0%, #F3EAFB 40%, #EADCF5 70%, #DCC9EE 100%);
                --glass-bg: rgba(255, 255, 255, 0.55);
                --glass-bg-strong: rgba(255, 255, 255, 0.75);
                --glass-border: rgba(107, 91, 149, 0.18);
                --text-primary: #241B36;
                --text-secondary: rgba(36, 27, 54, 0.75);
                --text-muted: rgba(36, 27, 54, 0.5);
                --shadow-soft: 0 8px 32px rgba(107, 91, 149, 0.12);
                --shadow-strong: 0 20px 50px rgba(107, 91, 149, 0.18);
                --shadow-glow: 0 0 40px rgba(181, 101, 167, 0.2);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def glass_card(content_html: str, extra_class: str = "") -> None:
    """Render a glassmorphism card with arbitrary inner HTML."""
    st.markdown(f"<div class='glass-card {extra_class}'>{content_html}</div>", unsafe_allow_html=True)


def section_header(label: str, title: str = "", desc: str = "") -> None:
    """Render a consistent section label / title / description block."""
    html = f"<div class='section-label'>{label}</div>"
    if title:
        html += f"<div class='section-title'>{title}</div>"
    if desc:
        html += f"<div class='section-desc'>{desc}</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_splash_screen() -> None:
    """Show a one-time animated splash screen on the first load of a session."""
    if not st.session_state.get("splash_shown", False):
        splash = st.empty()
        splash.markdown(
            """
            <div class="splash-wrap">
                <div class="splash-logo">🌸</div>
                <div class="splash-text">Initializing AI Flower Classification System</div>
                <div class="splash-sub">Loading model bundle &amp; visual assets...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(1.1)
        splash.empty()
        st.session_state.splash_shown = True


def render_sidebar_brand() -> None:
    """Render the animated brand block at the top of the sidebar."""
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo-mark">🌸</div>
            <div class="brand-text">
                <b>AI Flower Classification</b>
                <span>Iris Species Predictor · ML</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_status(model_ready: bool) -> None:
    """Render a small pulsing status pill showing whether the model is loaded."""
    status_class = "" if model_ready else "offline"
    status_text = "Model Online" if model_ready else "Model Offline"
    st.markdown(
        f"<div class='ai-status {status_class}'><span class='dot'></span> {status_text}</div>",
        unsafe_allow_html=True,
    )


def render_live_clock() -> None:
    """Render a server-time clock in the sidebar (updates on every rerun)."""
    now_str = datetime.now().strftime("%A, %b %d · %I:%M %p")
    st.markdown(f"<div class='live-clock'>🕐 {now_str}</div>", unsafe_allow_html=True)


def render_floating_widgets() -> None:
    """Best-effort floating scroll-to-top button + live clock injected into
    the parent document via a small HTML component. Wrapped in try/catch on
    the JS side so it fails silently if the browser sandbox restricts
    cross-frame DOM access — the server-rendered sidebar clock above is the
    reliable fallback either way.
    """
    components.html(
        """
        <script>
        try {
            const doc = window.parent.document;

            const oldBtn = doc.getElementById('sa-scroll-top');
            if (oldBtn) oldBtn.remove();

            const btn = doc.createElement('button');
            btn.id = 'sa-scroll-top';
            btn.innerHTML = '↑';
            btn.title = 'Scroll to top';
            btn.style.cssText = 'position:fixed; bottom:24px; right:24px; z-index:999999;' +
                'width:46px; height:46px; border-radius:50%; border:none; cursor:pointer;' +
                'font-size:20px; font-weight:700; color:#fff;' +
                'background:linear-gradient(120deg,#6B5B95,#B565A7);' +
                'box-shadow:0 8px 22px rgba(107,91,149,0.45); transition: transform 0.2s ease;';
            btn.onmouseover = function() { btn.style.transform = 'translateY(-3px) scale(1.08)'; };
            btn.onmouseout = function() { btn.style.transform = 'translateY(0) scale(1)'; };
            btn.onclick = function() { window.parent.scrollTo({top: 0, behavior: 'smooth'}); };
            doc.body.appendChild(btn);

            // Keyboard shortcut: "Home" scrolls to top
            if (!window.parent.__saKeyBound) {
                window.parent.document.addEventListener('keydown', function(e) {
                    if (e.key === 'Home') { window.parent.scrollTo({top: 0, behavior: 'smooth'}); }
                });
                window.parent.__saKeyBound = true;
            }

            // SEO: set a descriptive meta tag on the parent document head
            if (!doc.querySelector('meta[name="description"]')) {
                const meta = doc.createElement('meta');
                meta.name = 'description';
                meta.content = 'AI Flower Classification System — an end-to-end ML app classifying Iris flowers with KNN and comparative models.';
                doc.head.appendChild(meta);
            }
        } catch (e) { /* Fails silently if cross-frame access is restricted */ }
        </script>
        """,
        height=0,
    )


def render_ml_tip() -> None:
    """Render a random ML tip, fixed for the duration of the session."""
    if "ml_tip" not in st.session_state:
        st.session_state.ml_tip = random.choice(ML_TIPS)
    st.markdown(
        f"<div class='tip-card'>💡 <b>ML Tip:</b> {st.session_state.ml_tip}</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the premium footer."""
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="footer-made">Made with ❤️ using Python, Streamlit and Scikit-Learn</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_chart(probabilities: dict, theme: str) -> go.Figure:
    """Construct a polished horizontal Plotly bar chart of class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=["#6B5B95", "#B565A7", "#D4A574"],
                line=dict(color="rgba(255,255,255,0.35)", width=1),
            ),
            text=[f"{v * 100:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(family="Inter, sans-serif", size=13),
        )
    )
    fig.update_layout(
        title="Prediction Probability Breakdown",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin=dict(l=10, r=50, t=55, b=10),
        xaxis=dict(range=[0, 1], title="Probability"),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def build_confidence_gauge(confidence: float, theme: str) -> go.Figure:
    """Construct an animated-feeling radial gauge for the prediction confidence."""
    font_color = "#F7F3FB" if theme == "dark" else "#241B36"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round((confidence or 0) * 100, 1),
            number={"suffix": "%", "font": {"size": 34, "color": font_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": font_color},
                "bar": {"color": "#D4A574"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(196,102,92,0.25)"},
                    {"range": [50, 80], "color": "rgba(212,165,116,0.22)"},
                    {"range": [80, 100], "color": "rgba(74,124,89,0.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=font_color),
    )
    return fig


# =====================================================================
# SESSION STATE INITIALIZATION & THEME PERSISTENCE (via URL query params)
# =====================================================================
if "history" not in st.session_state:
    st.session_state.history = []

# Theme persists across page reloads using Streamlit's query params (?theme=light)
if "theme" not in st.session_state:
    st.session_state.theme = st.query_params.get("theme", "dark")

# =====================================================================
# CSS INJECTION
# =====================================================================
load_css()
apply_theme_variables(st.session_state.theme)
render_floating_widgets()
render_splash_screen()

# =====================================================================
# SIDEBAR
# =====================================================================
try:
    _model_bundle_probe = load_model("model.pkl")
    _model_ready_probe = True
except FileNotFoundError:
    _model_ready_probe = False

with st.sidebar:
    render_sidebar_brand()
    render_ai_status(_model_ready_probe)
    render_live_clock()

    page = st.radio(
        "Navigate",
        NAV_PAGES,
        label_visibility="collapsed",
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    theme_choice = st.toggle("🌙 Dark theme", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.query_params["theme"] = new_theme
        st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    render_ml_tip()

    st.markdown(f"<div class='version-tag'>AI Flower Classification · {APP_VERSION}</div>", unsafe_allow_html=True)

# =====================================================================
# HEADER / HERO SECTION
# =====================================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">⚡ Machine Learning · Supervised Classification</div>
        <div class="hero-title">🌸 AI Flower Classification System</div>
        <div class="hero-subtitle">
            A production-grade machine learning application that classifies Iris flowers —
            <b>Setosa</b>, <b>Versicolor</b>, and <b>Virginica</b> — from petal and sepal
            measurements. Powered by a fully compared suite of classical ML algorithms,
            complete with live confidence scoring, rich exploratory visualizations,
            and transparent model evaluation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# Load trained model bundle (ML logic unchanged — src/utils.load_model)
# --------------------------------------------------------------------
try:
    model_bundle = load_model("model.pkl")
    metrics = load_metrics()
    model_ready = True
except FileNotFoundError:
    model_ready = False
    st.error(
        "⚠️ No trained model found. Please run `python src/train.py` first to "
        "generate the model bundle before using this app."
    )

# =====================================================================
# PROJECT STATISTICS ROW
# =====================================================================
if model_ready:
    stat_cols = st.columns(4)
    stats = [
        ("🌱", "150", "Total Samples"),
        ("🏷️", "3", "Flower Classes"),
        ("🏆", model_bundle.get("model_name", "N/A"), "Best Model"),
        ("🎯", f"{metrics.get('test_accuracy', 0) * 100:.1f}%", "Test Accuracy"),
    ]
    for i, (col, (icon, num, label)) in enumerate(zip(stat_cols, stats)):
        with col:
            glass_card(
                f"<div class='stat-card'><div class='stat-icon'>{icon}</div>"
                f"<div class='stat-number'>{num}</div>"
                f"<div class='stat-label'>{label}</div></div>",
                extra_class=f"delay-{i + 1}",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # Welcome / How-to-use glassmorphism section
    # -----------------------------------------------------------
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to the Dashboard</h3>
            <p style="color:var(--text-secondary); font-size:1rem; max-width:640px; margin:0 auto;">
                This AI system predicts the species of an Iris flower using a trained,
                cross-validated Machine Learning model. Explore the dataset, compare model
                performance, or jump straight into making a live prediction.
            </p>
            <div class="welcome-steps">
                <div class="welcome-step">
                    <div class="step-num">STEP 01</div>
                    <div class="step-title">🔮 Go to Predict</div>
                </div>
                <div class="welcome-step">
                    <div class="step-num">STEP 02</div>
                    <div class="step-title">📏 Set Measurements</div>
                </div>
                <div class="welcome-step">
                    <div class="step-num">STEP 03</div>
                    <div class="step-title">✨ Get Instant Results</div>
                </div>
            </div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
# PAGE: 🔮 PREDICT
# =====================================================================
if page == "🔮 Predict" and model_ready:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        section_header("Input", "Flower Measurements", "Adjust the sliders to describe the flower.")
        with st.form("prediction_form"):
            sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider("🌿 Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
            petal_length = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
            petal_width = st.slider("🌸 Petal Width (cm)", 0.1, 2.6, 1.3, 0.1)
            col_a, col_b = st.columns(2)
            with col_a:
                submitted = st.form_submit_button("🌸 Predict Species", use_container_width=True)
            with col_b:
                reset = st.form_submit_button("↺ Reset History", use_container_width=True)

        if reset:
            st.session_state.history = []
            st.rerun()

        if st.session_state.history:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Timeline", "Recent Predictions")
            recent_html = "<div class='timeline'>"
            for item in st.session_state.history[:5]:
                conf_pct = (item["confidence"] or 0) * 100
                recent_html += (
                    "<div class='timeline-item'>"
                    f"<div class='tl-title'>🌺 {item['species'].title()} "
                    f"<span style='color:var(--accent); font-size:0.8rem;'>({conf_pct:.1f}%)</span></div>"
                    f"<div class='tl-sub'>{item['time']}</div>"
                    "</div>"
                )
            recent_html += "</div>"
            glass_card(recent_html)

    with right:
        if submitted:
            # Animated "thinking" spinner before revealing the result.
            with st.spinner("🔬 Analyzing measurements..."):
                time.sleep(0.6)
                result = predict_species(sepal_length, sepal_width, petal_length, petal_width)

            st.session_state.history.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "species": result["predicted_species"],
                    "confidence": result["confidence"],
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                },
            )

            conf_pct = (result["confidence"] or 0) * 100
            st.success("✅ Prediction complete!")

            section_header("Result", "")
            st.markdown(
                f"""
                <div class='prediction-hero scale-in'>
                    <div class='prediction-check'>✅</div>
                    <div class='prediction-species'>🌺 {result['predicted_species']}</div>
                    <div class='confidence-badge'>Confidence: {conf_pct:.1f}%</div>
                    <div class='prediction-summary'>
                        <span class='summary-chip'>Sepal L: <b>{sepal_length} cm</b></span>
                        <span class='summary-chip'>Sepal W: <b>{sepal_width} cm</b></span>
                        <span class='summary-chip'>Petal L: <b>{petal_length} cm</b></span>
                        <span class='summary-chip'>Petal W: <b>{petal_width} cm</b></span>
                        <span class='summary-chip'>Model: <b>{result['model_used']}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            gauge_col, chart_col = st.columns([0.8, 1.2])
            with gauge_col:
                st.plotly_chart(
                    build_confidence_gauge(result["confidence"], st.session_state.theme),
                    use_container_width=True,
                )
            with chart_col:
                if result["probabilities"]:
                    fig = build_probability_chart(result["probabilities"], st.session_state.theme)
                    st.plotly_chart(fig, use_container_width=True)

            history_df = pd.DataFrame(st.session_state.history)
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction History (CSV)",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            glass_card(
                "<div class='section-label'>Ready</div>"
                "<p style='color:var(--text-secondary); font-family:var(--font-body);'>"
                "Adjust the sliders and click <b>Predict Species</b> to classify a flower "
                "and see its confidence score and probability breakdown.</p>"
            )

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Log", "Full Prediction History", "Every prediction made this session.")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )

# =====================================================================
# PAGE: 📊 DATASET & EDA  (Analytics Dashboard)
# =====================================================================
elif page == "📊 Dataset & EDA" and model_ready:
    df = load_iris_dataframe()

    section_header(
        "Overview",
        "Dataset Analytics Dashboard",
        "A quick statistical snapshot of the Iris dataset used to train this model.",
    )
    glass_card(
        "<p style='font-family:var(--font-body); color:var(--text-secondary); font-size:0.98rem; line-height:1.6;'>"
        "The <b>Iris dataset</b> contains 150 samples across 3 balanced species classes "
        "(Setosa, Versicolor, Virginica), each described by 4 numeric features: "
        "sepal length, sepal width, petal length, and petal width — all measured in centimeters.</p>"
    )

    feat_cols = st.columns(4)
    feat_info = [
        ("📏", "Sepal Length", "4.0 – 8.0 cm"),
        ("📐", "Sepal Width", "2.0 – 4.5 cm"),
        ("🌸", "Petal Length", "1.0 – 7.0 cm"),
        ("🌺", "Petal Width", "0.1 – 2.6 cm"),
    ]
    for col, (icon, name, rng) in zip(feat_cols, feat_info):
        with col:
            glass_card(
                f"<div class='stat-card'><div class='stat-icon'>{icon}</div>"
                f"<div style='font-weight:700; color:var(--text-primary);'>{name}</div>"
                f"<div class='stat-label'>{rng}</div></div>"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View Raw Dataset (150 records)", expanded=False):
        st.dataframe(df.drop(columns=["target"]), use_container_width=True, hide_index=True)

    with st.expander("📈 Dataset Summary Statistics", expanded=False):
        st.dataframe(df[FEATURE_NAMES].describe(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Visual Analysis",
        "Exploratory Visualizations",
        "Distribution, correlation, and separability plots generated by the training pipeline.",
    )

    plot_groups = {
        "📊 Distributions": [
            ("histograms.png", "Feature Histograms"),
            ("feature_distribution.png", "Feature Distribution by Species"),
        ],
        "🏷️ Class & Correlation": [
            ("count_plot.png", "Class Distribution"),
            ("correlation_heatmap.png", "Correlation Matrix"),
        ],
        "📦 Spread & Separability": [
            ("box_plots.png", "Box Plots by Species"),
            ("scatter_plot.png", "Petal Length vs Width"),
        ],
        "🔗 Pairwise Relationships": [
            ("pair_plot.png", "Pair Plot"),
        ],
    }
    for group_title, plots in plot_groups.items():
        with st.expander(group_title, expanded=(group_title == "📊 Distributions")):
            cols = st.columns(2)
            for i, (fname, title) in enumerate(plots):
                path = os.path.join(IMAGES_DIR, fname)
                if os.path.exists(path):
                    with cols[i % 2]:
                        st.markdown(f"**{title}**")
                        st.image(path, use_container_width=True)

# =====================================================================
# PAGE: 🏆 MODEL COMPARISON  (Performance Dashboard)
# =====================================================================


# =====================================================================
# PAGE: ℹ️ ABOUT  (Portfolio Page)
# =====================================================================
elif page == "ℹ️ About":


    # ---- Project Overview ----
    section_header("Project", "Project Overview")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        glass_card(
            "<div class='section-label'>🤖 About the AI</div>"
            "<p style='font-family:var(--font-body); color:var(--text-secondary); line-height:1.6;'>"
            "This system uses supervised learning to classify Iris flowers based on "
            "physical measurements. Multiple algorithms — KNN, Decision Tree, Random Forest, "
            "and Logistic Regression — are trained and compared, with the best performer "
            "automatically selected and deployed.</p>"
        )
    with col2:
        glass_card(
            "<div class='section-label'>🌸 About the Dataset</div>"
            "<p style='font-family:var(--font-body); color:var(--text-secondary); line-height:1.6;'>"
            "The Iris dataset, introduced by Ronald Fisher in 1936, is one of the most "
            "well-known datasets in pattern recognition. It contains 150 balanced samples "
            "across 3 species with 4 numeric features each.</p>"
        )
    with col3:
        model_name = model_bundle.get("model_name", "N/A") if model_ready else "N/A"
        best_k = model_bundle.get("best_k", "N/A") if model_ready else "N/A"
        glass_card(
            "<div class='section-label'>🏆 About the Model</div>"
            f"<p style='font-family:var(--font-body); color:var(--text-secondary); line-height:1.6;'>"
            f"The production model is a <b>{model_name}</b> classifier"
            f"{f' (K={best_k})' if model_name == 'K-Nearest Neighbors' else ''}, "
            "selected automatically after cross-validated comparison against three "
            "alternative algorithms.</p>"
        )

    # ---- Features ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Highlights", "Project Features")
    glass_card(
        "<div style='display:flex; flex-wrap:wrap; gap:0.6rem;'>"
        + "".join(
            f"<span class='summary-chip' style='font-size:0.85rem;'>{feat}</span>"
            for feat in PROJECT_FEATURES
        )
        + "</div>"
    )

    # ---- ML Pipeline Workflow Diagram ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Pipeline", "Machine Learning Workflow")
    workflow_html = "<div class='workflow-row'>"
    for i, step in enumerate(PIPELINE_STEPS):
        workflow_html += f"<div class='workflow-step'>{i + 1}. {step}</div>"
        if i < len(PIPELINE_STEPS) - 1:
            workflow_html += "<span class='workflow-arrow'>→</span>"
    workflow_html += "</div>"
    glass_card(workflow_html)

    # ---- Timeline view of the same pipeline ----
    with st.expander("🗓️ View Pipeline as a Timeline", expanded=False):
        timeline_html = "<div class='timeline'>"
        for i, step in enumerate(PIPELINE_STEPS):
            timeline_html += (
                "<div class='timeline-item'>"
                f"<div class='tl-title'>Stage {i + 1}: {step}</div>"
                "<div class='tl-sub'>Implemented in src/ pipeline modules</div>"
                "</div>"
            )
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

    # ---- Tech Stack ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Stack", "Technologies Used")
    glass_card(
        "<p style='font-family:var(--font-body); color:var(--text-secondary);'>"
        "Python · Scikit-Learn · Pandas · NumPy · Matplotlib · Seaborn · "
        "Streamlit · Plotly · Joblib</p>"
    )

    # ---- Future Improvements ----
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Roadmap", "Future Improvements")
    glass_card(
        "<ul style='font-family:var(--font-body); color:var(--text-secondary); line-height:1.9; margin:0; padding-left:1.2rem;'>"
        + "".join(f"<li>{item}</li>" for item in FUTURE_IMPROVEMENTS)
        + "</ul>"
    )

    # ---- Keyboard Shortcuts (documentation) ----
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.markdown(
            "- **Home** — scroll back to the top of the page\n"
            "- Use the sidebar radio buttons to switch pages instantly"
        )

# =====================================================================
# FOOTER
# =====================================================================
render_footer()
