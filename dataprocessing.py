from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"

TARGET = "Churn"

# Original numeric columns + newly engineered numeric columns
NUM_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "AvgMonthlyCharges",
    "ChargeDifference",
]

CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Sub-service columns where "No internet service" / "No phone service" is redundant
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_data(path=DATA_PATH):
    """Load the CSV and perform initial target and type cleaning."""
    df = pd.read_csv(path)

    # customerID is just an identifier, not useful for prediction
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges has blank strings (" ") for new customers (tenure == 0) -> convert to NaN then 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # SeniorCitizen is 0/1 -> convert safely to "Yes"/"No" across all Pandas versions
    df["SeniorCitizen"] = (
        df["SeniorCitizen"]
        .map({0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
        .fillna("No")
    )

    # Target: Yes -> 1 , No -> 0 (works for both object and Pandas 3.0 StringDtype)
    if TARGET in df.columns:
        df[TARGET] = (
            df[TARGET]
            .map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            .astype(int)
        )

    return df


def split_xy(df):
    """Separate features (X) and target (y)."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y


def engineer_features(df):
    """
    Clean redundant categories and create domain-specific features.
    Placed inside the Pipeline so it runs automatically on train, test,
    and single-row predictions in Streamlit (app.py).
    """
    df = df.copy()

    # 1. Domain fix for TotalCharges: new customers (tenure == 0) have 0 total charges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    # 2. Ensure SeniorCitizen is consistent ("Yes"/"No") without overwriting existing "Yes"/"No" strings
    df["SeniorCitizen"] = (
        df["SeniorCitizen"]
        .map({0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
        .fillna("No")
    )

    # 3. Remove redundant categories to prevent multicollinearity & independence violations
    # "No internet service" is already captured by InternetService == "No"
    for col in SERVICE_COLS:
        df[col] = df[col].replace({"No internet service": "No"})

    # "No phone service" is already captured by PhoneService == "No"
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    # 4. Feature Engineering:
    # a) TotalServices: count how many services the customer is subscribed to (0 to 7)
    all_services = ["PhoneService"] + SERVICE_COLS
    df["TotalServices"] = (df[all_services] == "Yes").sum(axis=1)

    # b) AvgMonthlyCharges: historical average monthly payment (avoid division by zero when tenure == 0)
    safe_tenure = df["tenure"].replace(0, 1)
    df["AvgMonthlyCharges"] = df["TotalCharges"] / safe_tenure
    df.loc[df["tenure"] == 0, "AvgMonthlyCharges"] = df["MonthlyCharges"]

    # c) ChargeDifference: current monthly bill minus historical average
    # Positive value indicates recent price increase or expired promo discount
    df["ChargeDifference"] = df["MonthlyCharges"] - df["AvgMonthlyCharges"]

    return df


def build_preprocessor():
    """
    Step 1: Feature engineering & category consolidation (FunctionTransformer)
    Step 2: ColumnTransformer:
      - Numeric columns     -> fill missing with 0, then standardize (StandardScaler)
      - Categorical columns -> fill missing with most frequent, then One-Hot Encode
        (using drop="if_binary" so binary Yes/No columns produce 1 column instead of 2)
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])

    return Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features)),
        ("columns", col_transformer),
    ])