import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🐄 Cattle Analytics Platform", page_icon="🐄", layout="wide"
)


# -----------------------------
# Utility helpers
# -----------------------------
def safe_set_env_api_key(key: str):
    """Set GOOGLE_API_KEY env var if provided (keeps key out of code)."""
    if key:
        os.environ["GOOGLE_API_KEY"] = key


def safe_transform_column(le, series: pd.Series) -> pd.Series:
    """
    Try to use encoder.transform. If unseen categories cause errors, fallback to
    mapping using le.classes_ when possible, otherwise fall back to -1 for unknowns.
    """
    if series is None:
        return series
    try:
        return pd.Series(le.transform(series), index=series.index)
    except Exception:
        if hasattr(le, "classes_"):

            def map_val(v):
                try:
                    return int(np.where(le.classes_ == v)[0][0])
                except Exception:
                    return -1

            return series.map(map_val).astype(int)
        return series


def reindex_with_defaults(df: pd.DataFrame, columns: list, default=0):
    """Return df reindexed to include columns; fill missing columns with default."""
    for c in columns:
        if c not in df.columns:
            df[c] = default
    return df[columns]


# -----------------------------
# LLM: Veterinary disease advisor
# -----------------------------
def get_disease_info(disease: str) -> dict:
    """Query the LLM to return structured disease info (returns dict or error)."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    except Exception as e:
        return {"error": f"Failed to initialize language model. Details: {e}"}

    prompt = PromptTemplate(
        input_variables=["disease"],
        template="""
You are an expert veterinary cattle disease advisor.

The user will give you a cattle disease name: **{disease}**.

Return a JSON response according to these rules:

1. If the input is "Healthy" (case-insensitive), return exactly:
{{
  "disease": "Healthy",
  "message": "The cattle is healthy and has no signs of any disease."
}}

2. If the input is a known cattle disease, return this structure:
{{
  "disease": "{{disease}}",
  "overview": "<Brief overview>",
  "harmful_effects": ["<Effect 1>", "<Effect 2>", "<Effect 3>"],
  "prevention": ["<Measure 1>", "<Measure 2>", "<Measure 3>"],
  "treatment": ["<Treatment 1>", "<Treatment 2>", "<Treatment 3>"]
}}

3. If unknown, return:
{{
  "disease": "{{disease}}",
  "message": "No specific information found. Please consult a professional veterinarian."
}}
""",
    )
    agent = prompt | llm
    try:
        response_content = agent.invoke({"disease": disease}).content
        cleaned = response_content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        return {"error": f"Could not retrieve advice: {str(e)}"}


# -----------------------------
# Cached artifacts & UI data loaders
# -----------------------------
@st.cache_resource
def load_artifacts():
    """Load all model artifacts from models/ directory (joblib)."""
    models_dir = "models"
    artifacts = {}
    try:
        artifacts["yield_model"] = joblib.load(
            os.path.join(models_dir, "yield_model.joblib")
        )
        artifacts["yield_encoders"] = joblib.load(
            os.path.join(models_dir, "yield_encoders.joblib")
        )
        artifacts["yield_features"] = joblib.load(
            os.path.join(models_dir, "yield_features.joblib")
        )
        artifacts["disease_model"] = joblib.load(
            os.path.join(models_dir, "disease_model.joblib")
        )
        artifacts["disease_encoders"] = joblib.load(
            os.path.join(models_dir, "disease_encoders.joblib")
        )
        artifacts["disease_target_encoder"] = joblib.load(
            os.path.join(models_dir, "disease_target_encoder.joblib")
        )
        artifacts["disease_features"] = joblib.load(
            os.path.join(models_dir, "disease_features.joblib")
        )
        # metrics (optional)
        try:
            artifacts["metrics"] = joblib.load(
                os.path.join(models_dir, "metrics.joblib")
            )
        except Exception:
            artifacts["metrics"] = None
    except Exception as e:
        raise e
    return artifacts


@st.cache_data
def load_ui_data():
    """Load CSV used to populate dropdowns."""
    df = pd.read_csv("global_cattle_milk_yield_prediction_dataset.csv")
    return {
        "Breed": sorted(df["Breed"].unique()),
        "Region": sorted(df["Region"].unique()),
        "Climate_Zone": sorted(df["Climate_Zone"].unique()),
        "Lactation_Stage": sorted(df["Lactation_Stage"].unique()),
        "Feed_Type": sorted(df["Feed_Type"].unique()),
        "Season": sorted(df["Season"].unique()),
    }


# -----------------------------
# PDF report generator
# -----------------------------
def generate_report(
    input_data: dict,
    predicted_yield: float,
    predicted_disease: str,
    advice: dict,
    metrics: dict,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("🐄 Cattle Analytics & Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Input Data
    story.append(Paragraph("📌 Input Data Provided", styles["Heading2"]))
    input_table = [["Field", "Value"]] + [[k, str(v)] for k, v in input_data.items()]
    table = Table(input_table, colWidths=[200, 260])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # Predictions
    story.append(Paragraph("📈 AI Predictions", styles["Heading2"]))
    story.append(
        Paragraph(
            f"<b>Predicted Milk Yield:</b> {predicted_yield:.2f} L/day",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Predicted Health Status:</b> {predicted_disease}", styles["Normal"]
        )
    )
    story.append(Spacer(1, 12))

    # Validation Metrics (flat values only ✅)
    story.append(Paragraph("📊 Model Validation Metrics", styles["Heading2"]))
    if metrics:
        story.append(
            Paragraph(
                f"Milk Yield RMSE: {metrics.get('yield_rmse', 0):.2f} L",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                f"Disease Detection F1 Score: {metrics.get('disease_f1', 0):.2f}",
                styles["Normal"],
            )
        )
    story.append(Spacer(1, 12))

    # Advice Section
    story.append(Paragraph("🩺 AI Veterinary Advisor", styles["Heading2"]))
    if "error" in advice:
        story.append(Paragraph(f"⚠️ {advice['error']}", styles["Normal"]))
    elif "message" in advice:
        story.append(Paragraph(advice["message"], styles["Normal"]))
    else:
        if advice.get("overview"):
            story.append(Paragraph("<b>Overview</b>", styles["Heading3"]))
            story.append(Paragraph(advice["overview"], styles["Normal"]))
        if advice.get("harmful_effects"):
            story.append(Paragraph("<b>Harmful Effects</b>", styles["Heading3"]))
            for e in advice["harmful_effects"]:
                story.append(Paragraph(f"- {e}", styles["Normal"]))
        if advice.get("prevention"):
            story.append(Paragraph("<b>Prevention</b>", styles["Heading3"]))
            for p in advice["prevention"]:
                story.append(Paragraph(f"- {p}", styles["Normal"]))
        if advice.get("treatment"):
            story.append(Paragraph("<b>Treatment Options</b>", styles["Heading3"]))
            for t in advice["treatment"]:
                story.append(Paragraph(f"- {t}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "ℹ️ Disclaimer: This report is AI-generated and should not replace professional veterinary advice.",
            styles["Italic"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer


# -----------------------------
# Sidebar / Top UI
# -----------------------------
st.sidebar.title("Cattle Analytics")

# API key (via .env)
api_key_input = os.getenv("API_KEY")
if api_key_input:
    safe_set_env_api_key(api_key_input)

page = st.sidebar.radio(
    "Select Page", options=["Prediction Platform", "Veterinary Chatbot"], index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("Tips:")
st.sidebar.markdown("- Enter correct cattle details then click **Predict Now**.")
st.sidebar.markdown("- Switch to **Veterinary Chatbot** for follow-up questions.")


# -----------------------------
# PREDICTION PAGE
# -----------------------------
if page == "Prediction Platform":
    st.title("🐄 Cattle Analytics & Prediction Platform")
    st.markdown(
        "Enter cattle details below to predict **milk yield** and **health status**. Predictions are local and model-based."
    )

    # Load artifacts and UI data
    try:
        with st.spinner("Loading models and UI data..."):
            artifacts = load_artifacts()
            ui_options = load_ui_data()
    except FileNotFoundError:
        st.error(
            "Model files or CSV not found. Ensure `models/` folder and CSV are present."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    # Sidebar inputs (full)
    st.sidebar.header("Enter Cattle Details")
    breed = st.sidebar.selectbox("Breed", ui_options["Breed"])
    age = st.sidebar.number_input("Age (Months)", min_value=1, max_value=240, value=60)
    weight = st.sidebar.number_input(
        "Weight (kg)", min_value=100.0, max_value=1000.0, value=550.0, step=0.1
    )
    lactation = st.sidebar.selectbox("Lactation Stage", ui_options["Lactation_Stage"])
    parity = st.sidebar.number_input("Parity (calvings)", 0, 15, 3)
    days_in_milk = st.sidebar.number_input("Days in Milk", 1, 400, 150)
    prev_yield = st.sidebar.number_input(
        "Previous Week Avg. Yield (L)", 0.0, 50.0, 15.0, 0.1
    )

    st.sidebar.subheader("Feed & Behavior")
    feed_type = st.sidebar.selectbox("Feed Type", ui_options["Feed_Type"])
    feed_qty = st.sidebar.number_input("Feed Quantity (kg/day)", 1.0, 50.0, 15.0, 0.1)
    water = st.sidebar.number_input("Water Intake (L/day)", 10.0, 150.0, 60.0, 0.1)
    milking_interval = st.sidebar.selectbox(
        "Milking Interval (hours)", [8, 12, 24], index=1
    )
    walking = st.sidebar.number_input("Walking Distance (km/day)", 0.0, 15.0, 3.0, 0.1)
    grazing = st.sidebar.number_input("Grazing Duration (hrs/day)", 0.0, 12.0, 4.0, 0.1)
    rumination = st.sidebar.number_input(
        "Rumination Time (hrs/day)", 0.0, 12.0, 8.0, 0.1
    )
    resting = st.sidebar.number_input("Resting Hours (hrs/day)", 0.0, 16.0, 9.0, 0.1)

    st.sidebar.subheader("Environment & Health Vitals")
    region = st.sidebar.selectbox("Region", ui_options["Region"])
    climate = st.sidebar.selectbox("Climate Zone", ui_options["Climate_Zone"])
    season = st.sidebar.selectbox("Season", ui_options["Season"])
    housing_score = st.sidebar.number_input(
        "Housing Score",
        min_value=0.00,
        max_value=1.00,
        value=0.70,
        step=0.01,
        format="%.2f",
    )
    ambient_temp = st.sidebar.number_input(
        "Ambient Temperature (°C)", -10.0, 50.0, 25.0, 0.1
    )
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0, 0.1)
    body_temp = st.sidebar.number_input("Body Temperature (°C)", 35.0, 45.0, 38.5, 0.1)
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 40, 120, 60)
    resp_rate = st.sidebar.number_input("Respiratory Rate (breaths/min)", 10, 50, 30)

    st.sidebar.subheader("Vaccination Status")
    vac_cols = st.sidebar.columns(2)
    fmd_vac = vac_cols[0].checkbox("FMD", True)
    bruc_vac = vac_cols[1].checkbox("Brucellosis", True)
    hs_vac = vac_cols[0].checkbox("HS", True)
    bq_vac = vac_cols[1].checkbox("BQ", True)
    anthrax_vac = vac_cols[0].checkbox("Anthrax", True)
    ibr_vac = vac_cols[1].checkbox("IBR", True)
    bvd_vac = vac_cols[0].checkbox("BVD", True)
    rabies_vac = vac_cols[1].checkbox("Rabies", True)

    col_pred, col_clear = st.columns([2, 1])
    predict_now = col_pred.button("✨ Predict Now", use_container_width=True)
    clear_pred = col_clear.button("🧹 Clear Prediction", use_container_width=True)

    input_data = {
        "Breed": breed,
        "Age_Months": age,
        "Weight_kg": weight,
        "Lactation_Stage": lactation,
        "Parity": parity,
        "Milking_Interval_hrs": milking_interval,
        "Feed_Type": feed_type,
        "Feed_Quantity_kg": feed_qty,
        "Walking_Distance_km": walking,
        "Grazing_Duration_hrs": grazing,
        "Rumination_Time_hrs": rumination,
        "Resting_Hours": resting,
        "FMD_Vaccine": int(fmd_vac),
        "Brucellosis_Vaccine": int(bruc_vac),
        "HS_Vaccine": int(hs_vac),
        "BQ_Vaccine": int(bq_vac),
        "Anthrax_Vaccine": int(anthrax_vac),
        "IBR_Vaccine": int(ibr_vac),
        "BVD_Vaccine": int(bvd_vac),
        "Rabies_Vaccine": int(rabies_vac),
        "Body_Temperature_C": body_temp,
        "Heart_Rate_bpm": heart_rate,
        "Ambient_Temperature_C": ambient_temp,
        "Humidity_percent": humidity,
        "Season": season,
        "Housing_Score": housing_score,
        "Region": region,
        "Climate_Zone": climate,
        "Days_in_Milk": days_in_milk,
        "Water_Intake_L": water,
        "Respiratory_Rate": resp_rate,
        "Previous_Week_Avg_Yield": prev_yield,
    }

    if clear_pred:
        for k in ["prediction_result", "shap_df"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cleared previous prediction.")
        st.experimental_rerun()

    # Prediction flow
    if predict_now:
        with st.spinner("Processing predictions..."):
            # DataFrames for models
            yield_df = pd.DataFrame([input_data])
            disease_df = pd.DataFrame([input_data])

            # Encode categorical for yield model
            for col, le in artifacts["yield_encoders"].items():
                if col in yield_df.columns:
                    yield_df[col] = safe_transform_column(le, yield_df[col])

            # Reindex to required features
            yield_ready = yield_df.reindex(
                columns=artifacts["yield_features"], fill_value=0
            )

            # Predict yield
            try:
                predicted_yield = float(
                    artifacts["yield_model"].predict(yield_ready)[0]
                )
            except Exception as e:
                st.error(f"Yield prediction failed: {e}")
                predicted_yield = 0.0

            # Add predicted yield to disease df
            disease_df["Milk_Yield_L"] = predicted_yield

            # Encode categorical for disease model
            for col, le in artifacts["disease_encoders"].items():
                if col in disease_df.columns:
                    disease_df[col] = safe_transform_column(le, disease_df[col])

            disease_ready = disease_df.reindex(
                columns=artifacts["disease_features"], fill_value=0
            )

            # Disease prediction
            try:
                pred_enc = artifacts["disease_model"].predict(disease_ready)[0]
                predicted_disease = artifacts[
                    "disease_target_encoder"
                ].inverse_transform([pred_enc])[0]
            except Exception as e:
                st.error(f"Disease prediction failed: {e}")
                predicted_disease = "Unknown"
                pred_enc = None

            # Save results
            st.session_state.prediction_result = {
                "input_data": input_data,
                "predicted_yield": predicted_yield,
                "predicted_disease": predicted_disease,
                "pred_enc": int(pred_enc) if pred_enc is not None else None,
            }

            # Fetch advice (LLM)
            with st.spinner("Fetching AI veterinary advice..."):
                advice = get_disease_info(predicted_disease)
            st.session_state.prediction_result["advice"] = advice

            # SHAP explanation (if applicable)
            if str(predicted_disease).lower() != "healthy":
                try:
                    explainer = shap.TreeExplainer(artifacts["disease_model"])
                    explanation = explainer(disease_ready)
                    if hasattr(explanation, "values"):
                        vals = explanation.values
                        if (
                            vals.ndim == 3
                            and st.session_state.prediction_result.get("pred_enc")
                            is not None
                        ):
                            class_vals = vals[
                                0, :, st.session_state.prediction_result["pred_enc"]
                            ]
                        else:
                            class_vals = vals[0, :]
                        feature_names = getattr(
                            explanation, "feature_names", disease_ready.columns.tolist()
                        )
                        shap_df = pd.DataFrame(
                            {"feature": feature_names, "contribution": class_vals}
                        )
                        shap_df = shap_df.sort_values("contribution", ascending=False)
                        st.session_state.shap_df = shap_df
                except Exception as e:
                    st.warning(f"SHAP explanation could not be generated: {e}")

        st.success("Prediction complete — results saved.")

    # Show stored prediction and metrics
    if "prediction_result" in st.session_state:
        res = st.session_state.prediction_result
        st.subheader("📈 AI Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Milk Yield", f"{res['predicted_yield']:.2f} L/day")
        col2.metric("Predicted Health Status", str(res["predicted_disease"]))

        # Show model validation metrics if available
        metrics = artifacts.get("metrics")
        if metrics:
            st.subheader("📊 Model Validation Metrics (test set)")
            st.metric(
                "Milk Yield RMSE (test)", f"{metrics.get('yield_rmse', 'N/A'):.3f} L"
            )
            st.metric(
                "Disease Detection F1 (test)", f"{metrics.get('disease_f1', 'N/A'):.3f}"
            )

            # classification report expander
            if metrics.get("disease_classification_report"):
                with st.expander("🔍 Disease classification report (detailed)"):
                    cr = metrics["disease_classification_report"]
                    # Convert to DataFrame for display
                    try:
                        cr_df = pd.DataFrame(cr).T
                        st.dataframe(cr_df)
                    except Exception:
                        st.write(cr)
        else:
            st.warning(
                "Model validation metrics not found. Run `train_models.py` to generate metrics.joblib."
            )

        # SHAP visualization
        if "shap_df" in st.session_state and st.session_state.shap_df is not None:
            shap_df = st.session_state.shap_df.copy()
            positive = shap_df[shap_df["contribution"] > 0].head(7)
            if not positive.empty:
                st.subheader("🔬 Top Factors Pushing Prediction Toward Disease")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(positive["feature"], positive["contribution"])
                ax.set_xlabel("Impact on Prediction (SHAP value)")
                plt.gca().invert_yaxis()
                st.pyplot(fig, use_container_width=True)

        st.divider()
        st.subheader("🩺 AI Veterinary Advisor")
        advice = res.get("advice", {})
        if "error" in advice:
            st.error(advice["error"])
        elif "message" in advice:
            st.info(advice["message"])
        else:
            if advice.get("overview"):
                with st.expander("📖 Overview", expanded=False):
                    st.write(advice["overview"])
            if advice.get("harmful_effects"):
                st.error("⚠️ Harmful Effects")
                for e in advice["harmful_effects"]:
                    st.markdown(f"- {e}")
            if advice.get("prevention"):
                st.warning("🛡️ Prevention")
                for p in advice["prevention"]:
                    st.markdown(f"- {p}")
            if advice.get("treatment"):
                st.success("💊 Treatment Options")
                for t_item in advice["treatment"]:
                    st.markdown(f"- {t_item}")

        st.info(
            "ℹ️ Disclaimer: This is an AI prediction and not a substitute for professional veterinary advice."
        )
        st.divider()

        # Report download (include metrics)
        pdf_buffer = generate_report(
            res["input_data"],
            res["predicted_yield"],
            res["predicted_disease"],
            res["advice"],
            artifacts["metrics"],
        )
        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_buffer,
            file_name=f"Cattle_Report_{res['predicted_disease']}.pdf",
            mime="application/pdf",
        )


# -----------------------------
# CHATBOT PAGE
# -----------------------------
elif page == "Veterinary Chatbot":
    st.title("💬 AI Veterinary Assistant")
    st.markdown(
        "Ask about cattle health, symptoms, feed, or management. The chat is kept separate from the prediction page so your results stay intact."
    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm an AI assistant for cattle health. How can I help you today? Please remember I am not a substitute for a professional veterinarian.",
            }
        ]

    chat_col, tips_col = st.columns([3, 1])
    with chat_col:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input
        if user_input := st.chat_input("Ask about cattle health, symptoms, or feed..."):
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash", temperature=0.3
                        )

                        system_prompt = """
You are an AI Veterinary Assistant specializing in cattle. Provide concise, practical, farmer-friendly advice. If outside scope, advise consulting a veterinarian.

At the end of every reply include:

---
Disclaimer: I am an AI assistant and not a substitute for professional veterinary advice. Always consult a qualified veterinarian for diagnosis and treatment.
"""
                        chat_history = [SystemMessage(content=system_prompt)]
                        for msg in st.session_state.chat_messages:
                            if msg["role"] == "user":
                                chat_history.append(
                                    HumanMessage(content=msg["content"])
                                )
                            elif msg["role"] == "assistant":
                                chat_history.append(AIMessage(content=msg["content"]))

                        response = llm.invoke(chat_history)
                        assistant_response = response.content
                    except Exception as e:
                        assistant_response = (
                            f"Sorry, an error occurred while contacting the AI: {e}"
                        )
                        st.error(assistant_response)

                # Ensure disclaimer
                if ("---" not in assistant_response) and (
                    "Disclaimer:" not in assistant_response
                ):
                    assistant_response = (
                        assistant_response
                        + "\n\n---\nDisclaimer: I am an AI assistant and not a substitute for professional veterinary advice. Always consult a qualified veterinarian for diagnosis and treatment."
                    )

                st.markdown(assistant_response)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

    with tips_col:
        st.markdown("### Quick Tips")
        st.markdown(
            "- Provide symptoms and recent changes (appetite, behavior, milk drop)."
        )
        st.markdown(
            "- For severe issues (high fever, bleeding, severe breathing trouble) contact a vet immediately."
        )
        if "prediction_result" in st.session_state:
            st.markdown("---")
            st.markdown("### Last Prediction")
            pr = st.session_state.prediction_result
            st.markdown(f"- **Yield:** {pr['predicted_yield']:.2f} L/day")
            st.markdown(f"- **Health:** {pr['predicted_disease']}")
            st.markdown(
                "Switch back to Prediction Platform to download the report or view SHAP details."
            )
