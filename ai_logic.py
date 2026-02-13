import os
import google.generativeai as genai

# This function will handle the AI brainstorming for your Lofi beats
def generate_lofi_insight(prompt_text):
    # This automatically grabs the key from your Streamlit Secrets
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # Using the latest Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt_text)
    return response.text
    
