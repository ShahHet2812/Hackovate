import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json
from math import sqrt


def train_models():
    """
    Trains XGBoost models for:
      - Milk yield regression
      - Disease detection classification (with class weighting)

    Saves:
      - models/*.joblib (models + encoders + feature lists)
      - models/metrics.joblib (yield_rmse, disease_f1, disease_classification_report)
    """
    print("--- Starting Local Model Training (XGBoost CPU) ---")
    os.makedirs("models", exist_ok=True)

    # Feature candidate list (kept for clarity)
    feature_columns = [
        "Breed",
        "Age_Months",
        "Weight_kg",
        "Lactation_Stage",
        "Parity",
        "Milking_Interval_hrs",
        "Feed_Type",
        "Feed_Quantity_kg",
        "Walking_Distance_km",
        "Grazing_Duration_hrs",
        "Rumination_Time_hrs",
        "Resting_Hours",
        "FMD_Vaccine",
        "Brucellosis_Vaccine",
        "HS_Vaccine",
        "BQ_Vaccine",
        "Anthrax_Vaccine",
        "IBR_Vaccine",
        "BVD_Vaccine",
        "Rabies_Vaccine",
        "Body_Temperature_C",
        "Heart_Rate_bpm",
        "Ambient_Temperature_C",
        "Humidity_percent",
        "Season",
        "Housing_Score",
        "Region",
        "Climate_Zone",
        "Days_in_Milk",
        "Water_Intake_L",
        "Respiratory_Rate",
        "Previous_Week_Avg_Yield",
    ]

    metrics = {}

    # -------------------------
    # 1) Milk Yield Regression
    # -------------------------
    print("\n[1/2] Loading milk yield dataset...")
    yield_csv = "global_cattle_milk_yield_prediction_dataset.csv"
    if not os.path.exists(yield_csv):
        print(
            f"ERROR: '{yield_csv}' not found. Place dataset in project folder and retry."
        )
        return

    df_yield = pd.read_csv(yield_csv)
    # Keep only available features from feature_columns + target
    yield_cols_to_keep = [c for c in feature_columns if c in df_yield.columns] + [
        "Milk_Yield_L"
    ]
    df_yield = df_yield[yield_cols_to_keep].dropna().reset_index(drop=True)

    # Encode categorical features and save encoders
    print("Encoding categorical features for yield model...")
    yield_encoders = {}
    df_yield_encoded = df_yield.copy()
    categorical_cols_yield = df_yield_encoded.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in categorical_cols_yield:
        if col == "Milk_Yield_L":
            continue
        le = LabelEncoder()
        df_yield_encoded[col] = le.fit_transform(df_yield_encoded[col].astype(str))
        yield_encoders[col] = le

    # Features used for training (exclude target)
    yield_features = [c for c in yield_cols_to_keep if c != "Milk_Yield_L"]
    X_yield = df_yield_encoded[yield_features]
    y_yield = df_yield_encoded["Milk_Yield_L"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_yield, y_yield, test_size=0.2, random_state=42
    )

    # Train XGBoost regressor
    print("Training XGBoost Regressor for milk yield (this may take a moment)...")
    yield_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
    yield_model.fit(X_train, y_train)
    print("Yield model trained.")

    # Evaluate
    y_pred = yield_model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Milk Yield RMSE (test): {rmse:.4f} L")
    metrics["yield_rmse"] = float(rmse)

    # Save artifacts
    joblib.dump(yield_model, "models/yield_model.joblib")
    joblib.dump(yield_encoders, "models/yield_encoders.joblib")
    joblib.dump(yield_features, "models/yield_features.joblib")
    print("Saved yield_model, yield_encoders, yield_features.")

    # -------------------------
    # 2) Disease Classification
    # -------------------------
    print("\n[2/2] Loading disease dataset...")
    disease_csv = "global_cattle_disease_detection_dataset.csv"
    if not os.path.exists(disease_csv):
        print(
            f"ERROR: '{disease_csv}' not found. Place dataset in project folder and retry."
        )
        return

    df_d = pd.read_csv(disease_csv)
    # Keep features present plus 'Milk_Yield_L' and 'Disease_Status'
    disease_cols_to_keep = [c for c in feature_columns if c in df_d.columns] + [
        "Milk_Yield_L",
        "Disease_Status",
    ]
    df_d = df_d[disease_cols_to_keep].dropna().reset_index(drop=True)

    # Encode categorical features (exclude target)
    print("Encoding categorical features for disease model...")
    disease_encoders = {}
    df_d_encoded = df_d.copy()
    categorical_cols_disease = df_d_encoded.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if "Disease_Status" in categorical_cols_disease:
        categorical_cols_disease.remove("Disease_Status")
    for col in categorical_cols_disease:
        le = LabelEncoder()
        df_d_encoded[col] = le.fit_transform(df_d_encoded[col].astype(str))
        disease_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    df_d_encoded["Disease_Status"] = target_encoder.fit_transform(
        df_d_encoded["Disease_Status"].astype(str)
    )

    X_disease = df_d_encoded.drop("Disease_Status", axis=1)
    y_disease = df_d_encoded["Disease_Status"]
    disease_features = X_disease.columns.tolist()

    # Train-test split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_disease, y_disease, test_size=0.2, random_state=42
    )

    # Compute sample weights for class imbalance
    print("Computing sample weights for class balancing...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_d)

    # Train classifier
    print("Training XGBoost Classifier for disease detection...")
    disease_model = xgb.XGBClassifier(
        n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss", random_state=42
    )
    disease_model.fit(X_train_d, y_train_d, sample_weight=sample_weights)
    print("Disease model trained.")

    # Evaluate
    y_pred_d = disease_model.predict(X_test_d)
    f1 = f1_score(y_test_d, y_pred_d, average="weighted")
    print(f"✅ Disease Detection weighted F1 (test): {f1:.4f}")
    class_report = classification_report(y_test_d, y_pred_d, output_dict=True)
    metrics["disease_f1"] = float(f1)
    metrics["disease_classification_report"] = class_report

    # Save artifacts
    joblib.dump(disease_model, "models/disease_model.joblib")
    joblib.dump(disease_encoders, "models/disease_encoders.joblib")
    joblib.dump(target_encoder, "models/disease_target_encoder.joblib")
    joblib.dump(disease_features, "models/disease_features.joblib")
    print(
        "Saved disease_model, disease_encoders, disease_target_encoder, disease_features."
    )

    # Save metrics
    joblib.dump(metrics, "models/metrics.joblib")
    # Also write a human-readable json copy for easy inspection
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics (metrics.joblib, metrics.json).")
    print("\n--- Training complete. All artifacts saved to 'models/' ---")


if __name__ == "__main__":
    train_models()
