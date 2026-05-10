"""
Developer Burnout Risk Classifier — Streamlit App
ECON 3916 Final Project

Screening aid for engineering team leads and wellness coordinators.
NOT for performance reviews or adverse employment decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Developer Burnout Risk Classifier",
    page_icon="⚠️",
    layout="wide",
)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
@st.cache_resource
def load_full_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_no_stress_model():
    return joblib.load("model_no_stress.pkl")

model = load_full_model()
model_no_stress = load_no_stress_model()

FEATURE_COLS = [
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day",
    "meetings_per_day", "screen_time", "exercise_hours", "stress_level",
]

TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#c0392b"}

# ------------------------------------------------------------
# Sample profiles (representative inputs within training distribution)
# ------------------------------------------------------------
PROFILES = {
    "Sustainable senior dev": {
        "age": 38, "experience_years": 12, "daily_work_hours": 8.0,
        "sleep_hours": 7.5, "caffeine_intake": 2, "bugs_per_day": 3,
        "commits_per_day": 8, "meetings_per_day": 3, "screen_time": 8.5,
        "exercise_hours": 1.0, "stress_level": 30,
    },
    "Mid-level under load": {
        "age": 30, "experience_years": 6, "daily_work_hours": 11.0,
        "sleep_hours": 6.0, "caffeine_intake": 4, "bugs_per_day": 9,
        "commits_per_day": 14, "meetings_per_day": 5, "screen_time": 12.0,
        "exercise_hours": 0.5, "stress_level": 55,
    },
    "Junior burning out": {
        "age": 25, "experience_years": 2, "daily_work_hours": 13.5,
        "sleep_hours": 4.5, "caffeine_intake": 6, "bugs_per_day": 16,
        "commits_per_day": 5, "meetings_per_day": 8, "screen_time": 14.5,
        "exercise_hours": 0.1, "stress_level": 85,
    },
}

DEFAULTS = {
    "age": 30, "experience_years": 5, "daily_work_hours": 9.0,
    "sleep_hours": 7.0, "caffeine_intake": 2, "bugs_per_day": 2,
    "commits_per_day": 4, "meetings_per_day": 3, "screen_time": 9.0,
    "exercise_hours": 0.5, "stress_level": 50,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_profile():
    name = st.session_state.profile
    if name in PROFILES:
        for k, v in PROFILES[name].items():
            st.session_state[k] = v

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Developer Burnout Risk Classifier")
st.caption(
    "**Macro F1 = 0.991** on 5-fold CV · **98.6% recall** on held-out High-tier cases "
    "(351 / 356) · ~70% of predictive weight from `stress_level` — see the behavior-only "
    "panel below for the ablation that tested this dependence."
)
st.markdown(
    "Screening tool for engineering team leads. Given a developer's recent "
    "work patterns and self-reported stress, predict burnout tier "
    "(Low / Medium / High) and surface candidates for a wellness check-in."
)
st.warning(
    "**Intended use:** human-in-the-loop screening aid only. "
    "Predictions must not drive performance reviews, compensation, or any "
    "adverse employment decision. Feature importance is predictive, not causal."
)

# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
st.sidebar.header("Developer profile")
st.sidebar.selectbox(
    "Sample profile",
    ["Custom"] + list(PROFILES.keys()),
    key="profile",
    on_change=apply_profile,
    help="Pick a preset to see typical Low / Medium / High predictions without "
         "moving the sliders manually. You can still tweak any slider after.",
)
st.sidebar.divider()

with st.sidebar:
    st.slider("Age", 20, 65, key="age")
    st.slider("Years of experience", 0, 40, key="experience_years")

    st.markdown("**Work patterns**")
    st.slider("Daily work hours", 4.0, 16.0, step=0.5, key="daily_work_hours")
    st.slider("Daily screen time (hrs)", 4.0, 20.0, step=0.5, key="screen_time")
    st.slider("Meetings per day", 0, 12, key="meetings_per_day")
    st.slider("Commits per day", 0, 30, key="commits_per_day")
    st.slider("Bugs per day", 0, 20, key="bugs_per_day")

    st.markdown("**Lifestyle**")
    st.slider("Sleep hours", 3.0, 10.0, step=0.5, key="sleep_hours")
    st.slider("Exercise hours/day", 0.0, 3.0, step=0.1, key="exercise_hours")
    st.slider("Caffeine (servings/day)", 0, 10, key="caffeine_intake")

    st.markdown("**Self-reported**")
    st.slider("Stress level (0-100)", 0, 100, key="stress_level")

# Build inputs from session state
input_dict = {k: st.session_state[k] for k in FEATURE_COLS}
input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
input_df_no_stress = input_df.drop(columns=["stress_level"])

# ------------------------------------------------------------
# Predictions (both models)
# ------------------------------------------------------------
def predict_with(m, X):
    pred = m.predict(X)[0]
    proba = m.predict_proba(X)[0]
    proba_map = dict(zip(m.classes_, proba))
    return pred, proba_map

pred_full, proba_full = predict_with(model, input_df)
pred_ns, proba_ns = predict_with(model_no_stress, input_df_no_stress)


def render_tier_card(pred, proba_map, sub_caption):
    color = TIER_COLORS.get(pred, "#333")
    conf = proba_map.get(pred, 0)
    st.markdown(
        f"<div style='padding: 1.5rem; background: {color}; "
        f"border-radius: 8px; color: white; text-align: center;'>"
        f"<div style='font-size: 2.2rem; font-weight: 700;'>{pred}</div>"
        f"<div style='font-size: 0.95rem; opacity: 0.9;'>"
        f"Confidence: {conf:.0%}</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(sub_caption)

# ------------------------------------------------------------
# Side-by-side tier cards: full model vs behavior-only
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Full model")
    render_tier_card(
        pred_full, proba_full,
        "Random Forest trained on all 11 features. Macro F1 = 0.991 (5-fold CV).",
    )
with col2:
    st.subheader("Behavior-only model")
    render_tier_card(
        pred_ns, proba_ns,
        "Same model refit without `stress_level` (notebook §3.5). CV F1 drops "
        "0.991 → 0.76; High recall 98.6% → 66.8%. Drag the stress_level slider "
        "to see how much that one self-report is pulling.",
    )

# ------------------------------------------------------------
# Probability bars + recommended action (full model)
# ------------------------------------------------------------
st.divider()

col_prob, col_action = st.columns([1.3, 1])

with col_prob:
    st.subheader("Class probabilities (full model)")
    display_order = ["Low", "Medium", "High"]
    ordered_probs = [proba_full.get(c, 0) for c in display_order]
    ordered_colors = [TIER_COLORS[c] for c in display_order]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(display_order, ordered_probs, color=ordered_colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability")
    for bar, prob in zip(bars, ordered_probs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", fontsize=10,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with col_action:
    st.subheader("Recommended action")
    action_map = {
        "Low": "No action needed. Continue regular 1:1 cadence.",
        "Medium": "Schedule an informal check-in within 2 weeks. "
                  "Review current workload.",
        "High": "Prioritize a wellness conversation this week. "
                "Consider workload reassignment or recovery time.",
    }
    st.info(action_map.get(pred_full, ""))

# ------------------------------------------------------------
# Low-margin warning (on full-model prediction)
# ------------------------------------------------------------
st.divider()
st.subheader("Interpreting this prediction")

sorted_probs = sorted(proba_full.values(), reverse=True)
margin = sorted_probs[0] - sorted_probs[1]

if margin < 0.15:
    st.markdown(
        f"⚠️ **Low-margin prediction** ({margin:.0%} gap between the top two "
        "classes). This developer sits near a decision boundary. Treat the "
        "prediction with extra caution and weight the team lead's judgment."
    )
else:
    st.markdown(
        f"✓ **Confident prediction** ({margin:.0%} margin over the next tier)."
    )

# ------------------------------------------------------------
# Model details and limitations
# ------------------------------------------------------------
with st.expander("Model details and limitations", expanded=True):
    st.markdown(
        """
**Model:** Random Forest Classifier (200 trees, balanced class weights, `random_state=42`).

**Cross-validation performance:** macro F1 = 0.991 ± 0.002 (5-fold CV, n = 7,000).

**Held-out test:** 98.6% recall on High tier (351 / 356); all 5 misses labeled Medium, none as Low.

**Ablation:** refitting without `stress_level` drops CV macro F1 to 0.76 and High recall to 66.8%. The behavior-only model also flips the precision/recall asymmetry the wrong way for a triage tool (High precision 0.82 > High recall 0.67). See notebook §3.5.

**Key caveats:**

- The `stress_level` self-report feature carries ~70% of the model's predictive weight and correlates 0.60 / 0.55 / 0.49 with work hours, screen time, and bugs per day. The model is largely restating self-reported stress rather than predicting independent risk from behavior.
- Training data is from a public Kaggle dataset that is likely synthetic (uniform 2% missingness across every column, zero Tukey outliers across 7,000 rows). Real-world performance on a specific engineering org has not been validated.
- Feature importance reflects predictive signal, not causal effect.
- Tier boundaries (Low / Medium / High) are analyst choices applied to a continuous Burn Rate, not ground-truth categories.

**Do not use this tool for:** performance reviews, compensation decisions, hiring or firing, or any other adverse employment action.
        """
    )
