import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os


def train_models():
    """
    Trains and saves XGBoost models on a local machine (Mac).
    This version includes class weighting to handle imbalanced data for better accuracy.
    """
    print("--- Starting Local Model Training (XGBoost for Mac/CPU) ---")
    os.makedirs("models", exist_ok=True)

    # --- Define the specific feature set to be used ---
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

    # --- 1. Train Milk Yield Model ---
    print("\n[1/2] Training Milk Yield Prediction Model...")
    try:
        yield_df_full = pd.read_csv("global_cattle_milk_yield_prediction_dataset.csv")
        yield_cols_to_keep = [
            col for col in feature_columns if col in yield_df_full.columns
        ] + ["Milk_Yield_L"]
        yield_df = yield_df_full[yield_cols_to_keep].copy()
        yield_df.dropna(inplace=True)
    except FileNotFoundError:
        print(
            "Error: 'global_cattle_milk_yield_prediction_dataset.csv' not found in the project folder."
        )
        return

    # Encode categorical features
    yield_encoders = {}
    yield_df_encoded = yield_df.copy()
    categorical_cols_yield = yield_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in categorical_cols_yield:
        le = LabelEncoder()
        yield_df_encoded[col] = le.fit_transform(yield_df_encoded[col])
        yield_encoders[col] = le

    yield_features_list = [
        col
        for col in yield_cols_to_keep
        if col
        not in [
            "Milk_Yield_L",
            "Body_Temperature_C",
            "Heart_Rate_bpm",
            "Respiratory_Rate",
        ]
    ]
    X_yield = yield_df_encoded[yield_features_list]
    y_yield = yield_df_encoded["Milk_Yield_L"]

    # XGBoost Regressor for CPU. Removed 'device=cuda' for Mac compatibility.
    yield_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)
    yield_model.fit(X_yield, y_yield)
    print("Milk Yield Model trained successfully.")

    joblib.dump(yield_model, "models/yield_model.joblib")
    joblib.dump(yield_encoders, "models/yield_encoders.joblib")
    joblib.dump(yield_features_list, "models/yield_features.joblib")
    print("Milk Yield artifacts saved.")

    # --- 2. Train Disease Detection Model ---
    print("\n[2/2] Training Disease Detection Model with Class Balancing...")
    try:
        disease_df_full = pd.read_csv("global_cattle_disease_detection_dataset.csv")
        disease_cols_to_keep = [
            col for col in feature_columns if col in disease_df_full.columns
        ] + ["Milk_Yield_L", "Disease_Status"]
        disease_df = disease_df_full[disease_cols_to_keep].copy()
        disease_df.dropna(inplace=True)
    except FileNotFoundError:
        print(
            "Error: 'global_cattle_disease_detection_dataset.csv' not found in the project folder."
        )
        return

    disease_encoders = {}
    disease_df_encoded = disease_df.copy()
    categorical_cols_disease = disease_df.select_dtypes(
        include=["object", "category"]
    ).columns.drop("Disease_Status", errors="ignore")
    for col in categorical_cols_disease:
        le = LabelEncoder()
        disease_df_encoded[col] = le.fit_transform(disease_df_encoded[col])
        disease_encoders[col] = le

    target_encoder = LabelEncoder()
    disease_df_encoded["Disease_Status"] = target_encoder.fit_transform(
        disease_df_encoded["Disease_Status"]
    )

    X_disease = disease_df_encoded.drop("Disease_Status", axis=1)
    y_disease = disease_df_encoded["Disease_Status"]
    disease_features_list = list(X_disease.columns)

    # **FIX FOR INACCURACY**: Calculate and apply sample weights to handle class imbalance
    print("Calculating class weights to improve disease prediction accuracy...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_disease)
    print("Class weights calculated.")

    # XGBoost Classifier for CPU.
    disease_model = xgb.XGBClassifier(n_jobs=-1, random_state=42)

    # Pass the sample_weights to the fit method
    print("Training balanced disease model...")
    disease_model.fit(X_disease, y_disease, sample_weight=sample_weights)
    print("Disease Detection Model trained successfully.")

    joblib.dump(disease_model, "models/disease_model.joblib")
    joblib.dump(disease_encoders, "models/disease_encoders.joblib")
    joblib.dump(target_encoder, "models/disease_target_encoder.joblib")
    joblib.dump(disease_features_list, "models/disease_features.joblib")
    print("Disease Detection artifacts saved.")

    print(
        "\n--- Local model training complete. All files saved to 'models' directory. ---"
    )


if __name__ == "__main__":
    train_models()
