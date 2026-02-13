import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

# 1. Setup Gemini AI
# We will use Streamlit Secrets to keep your key safe
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Please add your Google API Key to Streamlit Secrets!")

st.set_page_config(page_title="Y.R.S Lofi Studio", page_icon="🎵")
st.title("🎵 Y.R.S Lofi Studio")
st.write("Using Gemini AI to carve your musical dreams.")

uploaded_file = st.file_uploader("Upload your track", type=["mp3"])

if uploaded_file:
    with st.spinner("AI analyzing rhythm and structure..."):
        # Basic Audio Analysis
        y, sr = librosa.load(uploaded_file)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Stanza Detection Logic
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        boundaries = librosa.segment.agglomerative(onset_env, 5)
        times = librosa.frames_to_time(boundaries, sr=sr)

        # UI Display
        st.metric("Detected BPM", f"{round(float(tempo), 2)}")
        
        st.subheader("AI Stanza Breakdown")
        for i in range(len(times)-1):
            st.info(f"Stanza {i+1}: {round(times[i], 2)}s - {round(times[i+1], 2)}s")

        # Gemini AI Insight
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"I am making a lofi remix of a song with {tempo} BPM. Suggest 3 lofi instruments that would fit this speed.")
        st.success("Gemini Production Tip:")
        st.write(response.text)
