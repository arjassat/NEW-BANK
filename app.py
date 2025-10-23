import streamlit as st
import requests
import base64
import os
from io import BytesIO
import pandas as pd

# --- Configuration ---
# NOTE: Place your OpenRouter API key securely in Streamlit secrets as:
# OPENROUTER_API_KEY="sk-or-v1-..."
# OpenRouter supports many models; we'll use a strong multimodal one.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-4o" # Excellent for complex vision/reasoning

# --- Helper Functions ---

def pdf_to_base64(uploaded_file):
    """Converts a Streamlit UploadedFile to a Base64 string."""
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.read()
    return base64.b64encode(bytes_data).decode("utf-8")

def generate_openrouter_prompt(file_base64_string):
    """
    Constructs the detailed multimodal prompt for OpenRouter.
    The instruction for precision exclusion is critical here.
    """
    if not file_base64_string:
        return []

    # System prompt sets the context and output format
    system_prompt = (
        "You are an expert financial data analyst. Your task is to accurately "
        "extract all bank transactions from the provided PDF statement image. "
        "Your final output MUST be a CSV table with three columns: Date, Description, Amount. "
        "The Date must be in YYYY-MM-DD format. "
        "The Amount must be a number with two decimal places. Debits must be negative, Credits positive."
    )

    # User prompt enforces the complex exclusion rule
    user_prompt = (
        "Process the attached bank statement. EXTRACT ALL individual transaction line items. "
        "CRITICAL INSTRUCTION: If the statement has a column specifically labeled 'Fees', "
        "DO NOT use the amounts from that dedicated column in the final 'Amount' column. "
        "However, INCLUDE any fee-related transactions (e.g., 'Service Fee', 'ATM Fee') "
        "that appear as a separate line item in the main debit/credit columns."
    )

    # Multimodal message structure for OpenRouter
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:application/pdf;base64,{file_base64_string}",
                "detail": "high"
            }}
        ]}
    ]

def call_openrouter_api(messages):
    """Calls the OpenRouter API with the defined messages."""
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API Key not found. Please set the OPENROUTER_API_KEY in your Streamlit secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        # OpenRouter-specific setting to enable multimodal/vision input
        "file_input_mode": "base64" 
    }

    try:
        with st.spinner(f"Extracting data using {OPENROUTER_MODEL} via OpenRouter..."):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status() # Raise exception for bad status codes
            
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        st.error(f"OpenRouter API Request Failed: {e}")
        st.info("Check if your API key is valid, the model is correct, or if you have hit a rate limit.")
        return None
    except KeyError:
        st.error("Error parsing OpenRouter response. The model may have returned an error or an unexpected format.")
        st.json(response_json) # Display the raw error response
        return None


# --- Streamlit App Layout ---

st.title("🏦 OpenRouter Bank Statement Extractor")
st.markdown("Upload a PDF statement to extract transactions using the **OpenRouter API** (via **GPT-4o**), ensuring only the dedicated 'Fees' column is excluded.")
st.caption("Your API key (set in secrets) will be used for this service.")

uploaded_file = st.file_uploader(
    "Upload your bank statement (PDF)",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    st.info(f"File uploaded: **{uploaded_file.name}**. Click below to process.")
    
    if st.button("Process Statement via OpenRouter", type="primary"):
        pdf_b64 = pdf_to_base64(uploaded_file)
        
        if pdf_b64:
            messages = generate_openrouter_prompt(pdf_b64)
            csv_output = call_openrouter_api(messages)
            
            if csv_output:
                st.subheader("✅ Extracted Transactions (CSV Format)")
                st.code(csv_output)
                
                # --- Display as DataFrame for verification and export ---
                try:
                    df = pd.read_csv(BytesIO(csv_output.encode('utf-8')))
                    st.subheader("📊 Transaction Table")
                    st.dataframe(df)

                    # Export button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv_data,
                        file_name=f"transactions_{uploaded_file.name.replace('.pdf', '')}_OpenRouter.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.warning(f"Could not convert AI output to DataFrame. Raw output may still be usable. Error: {e}")
