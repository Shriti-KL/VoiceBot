import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Set Streamlit page configuration
st.set_page_config(
    page_title="VoiceBot with Emotional Intelligence",
    layout="centered",
    )

# Load environment variables from .env file
load_dotenv()

# Import modules
from src.emotion_analysis import load_emotion_model
from src.openai_services import get_api_key, show_api_key_input, initialize_openai_client
from src.audio_processing import handle_audio_input
from src.ui_components import (
    display_logo,
    apply_custom_styles,
    display_sidebar,
    display_chat_messages,
    display_ai_audio
)

# Check for API key and show input if needed
api_key = get_api_key()
if not api_key:
    show_api_key_input()
    st.stop()

# Initialize OpenAI client
client = initialize_openai_client()
if not client:
    st.error("Failed to initialize OpenAI client. Please check your API key and try again.")
    if st.button("Reset API Key"):
        if "openai_api_key" in st.session_state:
            del st.session_state.openai_api_key
        st.rerun()
    st.stop()

# Store OpenAI client in session state
st.session_state.openai_client = client

# Load emotion model
emotion_model = load_emotion_model()
st.session_state.emotion_model = emotion_model

# --- Session State Initialization ---
# Initialize chat history if it doesn't exist in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "You are a helpful assistant. Keep your responses concise and to the point."})
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

# Initialize audio recorder counter to reset the widget
if "audio_counter" not in st.session_state:
    st.session_state.audio_counter = 0

# Initialize storage for AI audio to persist across reruns
if "ai_audio_file" not in st.session_state:
    st.session_state.ai_audio_file = None

# Initialize last processed audio length to track new recordings
if "last_processed_audio_len" not in st.session_state:
    st.session_state.last_processed_audio_len = 0

# Initialize status for fixed status bar
if "status_message" not in st.session_state:
    st.session_state.status_message = ""

# Initialize latest debug info for sidebar display
if "latest_debug_info" not in st.session_state:
    st.session_state.latest_debug_info = None

# Initialize processing state to control audiorecorder visibility
if "is_processing_audio" not in st.session_state:
    st.session_state.is_processing_audio = False

# Initialize pending audio for processing
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None

# --- Main App UI ---
# Display logo
display_logo()

# Apply custom styles
apply_custom_styles()

# Display title
st.title("VoiceBot with Emotional Intelligence")

# Display sidebar
display_sidebar()

# Display chat messages
display_chat_messages()

# Display AI audio if available
display_ai_audio()

# Handle audio input
handle_audio_input() 