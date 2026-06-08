"""
Medical Insurance Charge Predictor
----------------------------------
Trains a Gradient Boosting regression model on the Medical Cost Personal
dataset and serves a Gradio web app that estimates a person's yearly medical
insurance charge from six inputs (age, sex, BMI, children, smoker, region).

Author: Abdelrahman Elkurdi (ID 202410905),
        Data Science & AI, University of Petra.
"""

import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "insurance.csv"

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

# Choices and ranges derived from the dataset.
SEX_CHOICES = ["female", "male"]
SMOKER_CHOICES = ["no", "yes"]
REGION_CHOICES = ["southwest", "southeast", "northwest", "northeast"]


def build_pipeline() -> Pipeline:
    """Preprocessing + Gradient Boosting pipeline, exactly as documented.

    1. Missing values filled with the median (numeric) / most frequent (categorical).
    2. Numeric features (age, bmi, children) scaled with StandardScaler.
    3. Categorical features (sex, smoker, region) one-hot encoded into 0/1 columns.
    4. Gradient Boosting Regressor (300 trees, learning rate 0.03, depth 2).
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def train_model() -> Pipeline:
    """Load the dataset and fit the pipeline once when the app starts."""
    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


# Train at startup so every request reuses the same fitted model.
PIPELINE = train_model()


def predict_charge(age, sex, bmi, children, smoker, region) -> str:
    """Predict the yearly insurance charge for one person."""
    row = pd.DataFrame(
        [
            {
                "age": int(age),
                "sex": sex,
                "bmi": float(bmi),
                "children": int(children),
                "smoker": smoker,
                "region": region,
            }
        ]
    )
    charge = float(PIPELINE.predict(row)[0])
    charge = max(charge, 0.0)  # a billed charge can never be negative
    return f"${charge:,.2f} per year"


def build_demo() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="Medical Insurance Charge Predictor") as demo:
        gr.Markdown(
            """
            # 🏥 Medical Insurance Charge Predictor

            Estimate a person's **yearly medical insurance charge** from six simple
            details. Powered by a Gradient Boosting model trained on the Medical
            Cost Personal dataset (1338 records, test-set R² ≈ 0.85).

            💡 *Tip: keep age and BMI fixed and switch smoker from **no** to **yes**
            — watch the predicted cost jump. That is the main finding of the project.*
            """
        )
        with gr.Row():
            with gr.Column():
                age = gr.Slider(18, 64, value=30, step=1, label="Age")
                sex = gr.Radio(SEX_CHOICES, value="male", label="Sex")
                bmi = gr.Slider(15.0, 55.0, value=28.0, step=0.1, label="BMI")
                children = gr.Slider(
                    0, 5, value=0, step=1, label="Number of children"
                )
                smoker = gr.Radio(SMOKER_CHOICES, value="no", label="Smoker")
                region = gr.Dropdown(
                    REGION_CHOICES, value="southwest", label="Region"
                )
                predict_btn = gr.Button("Predict charge", variant="primary")
            with gr.Column():
                output = gr.Textbox(
                    label="Predicted yearly charge", interactive=False
                )
                gr.Examples(
                    examples=[
                        [40, "male", 28.0, 0, "no", "southwest"],
                        [40, "male", 28.0, 0, "yes", "southwest"],
                        [55, "female", 35.0, 2, "no", "northeast"],
                        [55, "female", 35.0, 2, "yes", "northeast"],
                    ],
                    inputs=[age, sex, bmi, children, smoker, region],
                    outputs=output,
                    fn=predict_charge,
                    cache_examples=False,
                    run_on_click=True,
                    label="Examples (non-smoker vs. smoker, same person)",
                )

        inputs = [age, sex, bmi, children, smoker, region]
        predict_btn.click(predict_charge, inputs=inputs, outputs=output)

    return demo


if __name__ == "__main__":
    build_demo().launch()
