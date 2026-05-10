# Developer Burnout Risk Classifier

Random Forest classifier that predicts a developer's burnout risk tier (Low / Medium / High) from work patterns and a self-reported stress score. Deployed as a Streamlit app. Final project for ECON 3916 — Statistical & Machine Learning for Economics.

**[Live demo →](https://econ3916-final-project-bpqngzwfdfzo6ycitevyaw.streamlit.app/)** · [Notebook](3916_final_project_starter.ipynb) · [Open in Colab](https://colab.research.google.com/github/ian-menachery/Burnout-App/blob/main/3916_final_project_starter.ipynb)

![Streamlit app screenshot](screenshot.png)

## The problem

Developer burnout is well-documented but hard to spot early — by the time it shows up in attrition or missed deadlines, the recovery cost is already high. A team lead with twelve direct reports doesn't have time to manually review work-pattern data on each one. The goal here is a screening aid: a model that flags developers who look like they're trending toward burnout, so a human can decide whether to schedule a check-in.

This is a screening tool, not a decision tool. It is not designed for performance reviews, compensation, or any adverse employment action.

## Headline result

The Random Forest reaches **macro F1 = 0.991 ± 0.002** on 5-fold cross-validation (n = 7,000). The number on its own is misleading, and the more interesting finding is the reason why:

About **70% of the model's predictive weight comes from a single feature — the self-reported `stress_level` score** (1–10). That feature also correlates 0.49–0.60 with the behavioral features the model is supposedly learning from (daily work hours, screen time, bugs per day). So in practice the classifier is mostly restating self-reported stress rather than predicting independent risk from work patterns.

Put differently: a manager who already has the `stress_level` number probably doesn't need this model. The behavioral features add real but small signal on top.

![Feature importance](feature_importance.png)

## Approach

- **Data:** [Developer Burnout Prediction Dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) on Kaggle (n = 7,000, accessed April 2026). Likely synthetic — provenance against a real engineering org could not be verified.
- **Features (11):** age, years of experience, daily work hours, sleep hours, caffeine intake, bugs per day, commits per day, meetings per day, screen time, exercise hours, self-reported stress level.
- **Target:** continuous `Burn Rate` from the dataset, bucketed into Low / Medium / High tiers. The tier boundaries are an analyst choice, not ground truth.
- **Models:** Logistic Regression baseline → `RandomForestClassifier` (200 trees, `class_weight="balanced"`, `random_state=42`).
- **Validation:** 5-fold cross-validation, macro F1 reported (averaged across the three classes).

Full EDA, preprocessing, model selection, and evaluation are in [`3916_final_project_starter.ipynb`](3916_final_project_starter.ipynb).

## Limitations

These are the same caveats surfaced inside the deployed app.

- **The stress_level feature dominates and leaks.** ~70% of model weight on a self-reported measure that correlates 0.49–0.60 with the work-pattern features means the model is largely restating self-reported stress, not learning a separable risk signal from behavior.
- **The training data is likely synthetic.** Real-world performance on a specific engineering org has not been validated.
- **Feature importance is predictive, not causal.** Reducing caffeine intake will not reduce someone's predicted burnout tier in the way the model's importance scores might suggest.
- **Tier boundaries are an analyst choice.** Low / Medium / High are bucketed from the continuous `Burn Rate`. Pushing those thresholds shifts the metric.
- **Not for adverse decisions.** Performance review, compensation, hiring, and firing are explicitly out of scope.

## Tech stack

Python · scikit-learn · Streamlit · pandas · matplotlib · Jupyter

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

`model.pkl` is included in this repo (~6.6 MB), so no separate download step. Tested on Python 3.10+. To re-run the analysis end-to-end, open `3916_final_project_starter.ipynb` (locally or in Colab) — runtime is ~2 minutes on a laptop, no GPU.

## Repo layout

| File | What it is |
|---|---|
| `app.py` | Streamlit app |
| `model.pkl` | Trained Random Forest |
| `3916_final_project_starter.ipynb` | EDA, preprocessing, modeling, evaluation |
| `developer_burnout_dataset_7000.csv` | Training data (Kaggle source) |
| `requirements.txt` | Python dependencies |
| `feature_importance.png` | Feature importance chart used in this README |
| `screenshot.png` | App screenshot used in this README |

## Course context

Built for ECON 3916 — Statistical & Machine Learning for Economics, Spring 2026, Northeastern University. Code is MIT-licensed; the underlying Kaggle dataset is subject to its original author's terms.
