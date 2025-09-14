import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


# ✅ Set your default Gemini API key here
DEFAULT_API_KEY = "AIzaSyDrMKi8pEnfxHQ7XIxUeIuz81OUexhusdA"


def get_disease_info(disease: str, api_key: str = DEFAULT_API_KEY) -> dict:
    """
    Takes a disease name and queries the Gemini model,
    returns a structured dictionary with detailed veterinary advice.
    """
    if not api_key:
        return {"error": "API Key is missing."}

    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        # ✅ Updated to latest model (fast + accurate)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    except Exception as e:
        return {
            "error": f"Failed to initialize the language model. Check API key. Details: {str(e)}"
        }

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

        2. If the input is a known cattle disease, return this full structure with real data filled:
        {{
            "disease": "<disease name>",
            "overview": "<A concise, one-paragraph overview of the disease, its cause, and how it spreads>",
            "harmful_effects": [
                "<A key harmful effect on the cattle, like 'Reduced milk production'>",
                "<Another key harmful effect, like 'Weight loss and weakness'>",
                "<Another key harmful effect, like 'Can lead to secondary infections'>"
            ],
            "prevention": [
                "<A primary preventive measure, like 'Regular vaccination schedule'>",
                "<Another preventive measure, like 'Maintaining clean and dry housing'>",
                "<Another preventive measure, like 'Quarantining new animals'>"
            ],
            "treatment": [
                "<A common treatment method, like 'Administering specific antibiotics as prescribed by a vet'>",
                "<Another treatment, like 'Providing supportive care with fluids and electrolytes'>",
                "<Another treatment, like 'Isolating the sick animal to prevent spread'>"
            ]
        }}

        3. If the input is not a valid or known cattle disease, or irrelevant text, return exactly:
        {{
            "disease": "{disease}",
            "message": "No specific information found for this condition. Please consult a professional veterinarian."
        }}

        **Rules**
        - Return ONLY a valid JSON object (no ```json fences, no extra text).
        - Use clear, simple language that a farmer can easily understand.
        - Provide 3-4 items for each list (harmful_effects, prevention, treatment).
        """,
    )

    agent = prompt | llm

    try:
        response_content = agent.invoke({"disease": disease}).content
        cleaned_response = (
            response_content.strip()
            .replace("```json", "")
            .replace("```", "")
        )
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "The AI model returned an invalid JSON format. Please try again."}
    except Exception as e:
        return {
            "disease": disease,
            "error": f"An unexpected error occurred while fetching advice: {str(e)}",
        }
