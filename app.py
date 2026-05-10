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
# Load model once and cache (don't reload on every slider tick)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Feature order must match the DataFrame used during training
FEATURE_COLS = [
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day",
    "meetings_per_day", "screen_time", "exercise_hours", "stress_level",
]

TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#c0392b"}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Developer Burnout Risk Classifier")
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

with st.sidebar:
    age = st.slider("Age", 20, 65, 30)
    experience_years = st.slider("Years of experience", 0, 40, 5)

    st.markdown("**Work patterns**")
    daily_work_hours = st.slider("Daily work hours", 4.0, 16.0, 9.0, 0.5)
    screen_time = st.slider("Daily screen time (hrs)", 4.0, 16.0, 9.0, 0.5)
    meetings_per_day = st.slider("Meetings per day", 0, 12, 3)
    commits_per_day = st.slider("Commits per day", 0, 20, 4)
    bugs_per_day = st.slider("Bugs per day", 0, 20, 2)

    st.markdown("**Lifestyle**")
    sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    exercise_hours = st.slider("Exercise hours/day", 0.0, 3.0, 0.5, 0.1)
    caffeine_intake = st.slider("Caffeine (servings/day)", 0, 10, 2)

    st.markdown("**Self-reported**")
    stress_level = st.slider("Stress level (1-10)", 1, 10, 5)

input_df = pd.DataFrame([[
    age, experience_years, daily_work_hours, sleep_hours,
    caffeine_intake, bugs_per_day, commits_per_day,
    meetings_per_day, screen_time, exercise_hours, stress_level,
]], columns=FEATURE_COLS)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
pred_class = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0]
classes = model.classes_
proba_map = dict(zip(classes, pred_proba))
top_prob = proba_map[pred_class]

# ------------------------------------------------------------
# Main layout: tier card + probability chart
# ------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Predicted tier")
    tier_color = TIER_COLORS.get(pred_class, "#333")
    st.markdown(
        f"<div style='padding: 1.5rem; background: {tier_color}; "
        f"border-radius: 8px; color: white; text-align: center;'>"
        f"<div style='font-size: 2.5rem; font-weight: 700;'>{pred_class}</div>"
        f"<div style='font-size: 1rem; opacity: 0.9;'>"
        f"Model confidence: {top_prob:.0%}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("**Recommended action**")
    action_map = {
        "Low": "No action needed. Continue regular 1:1 cadence.",
        "Medium": "Schedule an informal check-in within 2 weeks. "
                  "Review current workload.",
        "High": "Prioritize a wellness conversation this week. "
                "Consider workload reassignment or recovery time.",
    }
    st.info(action_map.get(pred_class, ""))

with col2:
    st.subheader("Class probabilities")

    display_order = ["Low", "Medium", "High"]
    ordered_probs = [proba_map.get(c, 0) for c in display_order]
    ordered_colors = [TIER_COLORS[c] for c in display_order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(display_order, ordered_probs, color=ordered_colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability")
    ax.set_title("How the model distributes probability across tiers")

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

# ------------------------------------------------------------
# Uncertainty flagging
# ------------------------------------------------------------
st.divider()
st.subheader("Interpreting this prediction")

sorted_probs = sorted(ordered_probs, reverse=True)
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

with st.expander("Model details and limitations", expanded=True):
    st.markdown(
        """
**Model:** Random Forest Classifier (balanced class weights)

**Cross-validation performance:** macro F1 = 0.991 ± 0.002
(5-fold CV on the Kaggle developer burnout dataset)

**Key caveats:**

- The `stress_level` self-report feature carries roughly 70% of the model's
  predictive weight and correlates 0.60 / 0.55 / 0.49 with work hours, screen
  time, and bugs per day. The model may be partly restating stress rather than
  predicting independent risk from behavior.
- Training data is from a public Kaggle dataset that is likely synthetic.
  Real-world performance on a specific engineering org has not been validated.
- Feature importance reflects predictive signal, not causal effect. Changing
  a feature (e.g. reducing caffeine) will not necessarily change burnout tier.
- Tier boundaries (Low/Medium/High) are analyst choices applied to a
  continuous Burn Rate, not ground-truth categories.

**Do not use this tool for:** performance reviews, compensation decisions,
hiring or firing, or any other adverse employment action.
        """
    )
