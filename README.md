---
title: Medical Insurance Charge Predictor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Medical Insurance Charge Predictor

Predicts a person's yearly medical insurance charge from six simple inputs
(age, sex, BMI, number of children, smoker status, region). Built for a
University of Petra Machine Learning course project.

**Author:** Abdelrahman Elkurdi (ID 202410905), Data Science & AI, University of Petra.

## What this is

The app takes details about a person and estimates the yearly medical insurance
cost they would be billed. It is a regression problem (the answer is a dollar
amount), so the underlying model predicts a continuous number.

## The data

Medical Cost Personal dataset (1338 records, 7 columns):
age, sex, bmi, children, smoker, region, and the target `charges`.

## How it works

When the app starts it trains a model on the dataset using this pipeline:

1. **Missing values** filled with the median.
2. **Numeric features** (age, bmi, children) scaled with StandardScaler.
3. **Categorical features** (sex, smoker, region) turned into 0/1 columns with one-hot encoding.
4. **Model:** Gradient Boosting Regressor (300 trees, learning rate 0.03, depth 2).

Gradient Boosting was the best of four models tested in the project.

## Results from the project (test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 4,031 | 5,833 | 0.77 |
| Decision Tree | 3,131 | 5,067 | 0.83 |
| Random Forest | 3,159 | 4,789 | 0.84 |
| **Gradient Boosting** | **2,852** | **4,743** | **0.85** |

The strongest cost drivers are smoking, then BMI, then age. The cost of being a
smoker grows sharply as BMI rises.

## Run locally

```bash
pip install -r requirements.txt gradio
python app.py
```

## Files

- `app.py` — trains the model and serves the Gradio interface
- `insurance.csv` — the dataset
- `requirements.txt` — Python dependencies
