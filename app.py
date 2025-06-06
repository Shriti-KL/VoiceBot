import streamlit as st
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import httpx # Required by openai
from audiorecorder import audiorecorder # New library for audio recording
from pathlib import Path # For handling file paths

# Cloud environment detection
IS_STREAMLIT_CLOUD = "STREAMLIT_CLOUD" in os.environ or "streamlit" in sys.modules

# Minimal emotion analysis imports
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from transformers import pipeline
    import librosa
    import numpy as np
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    AUDIO_LIBS_AVAILABLE = False
    st.error(f"Audio processing libraries not available: {e}")
    if IS_STREAMLIT_CLOUD:
        st.info("This might be a temporary issue with Streamlit Cloud. Please try refreshing the page.")

# --- Configuration ---
# Set Streamlit page configuration
# Re-adding 'page_icon' based on user's provided code snippet.
st.set_page_config(
    page_title="VoiceBot with Emotional Intelligence",
    layout="centered",
    )

# Load environment variables from .env file
load_dotenv()

# --- Session-based API Key Management ---
def get_api_key():
    """Get API key from session state only - no environment fallback."""
    # Check session state only
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    # No fallback to environment variables - always require manual input
    return None

def show_api_key_input():
    """Show API key input interface."""
    st.title("OpenAI API Key Required")
    st.markdown("""
    **Welcome to Voice Chat with AI!**
    
    To get started, please enter your OpenAI API key below. Your key will be stored securely for this session only and will be automatically cleared when you close your browser.
    
    Don't have an API key? Get one at [OpenAI Platform](https://platform.openai.com/api-keys)
    """)
    
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Your API key is stored only for this browser session and is not saved permanently."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Voice Chat", type="primary", use_container_width=True):
            if api_key_input and api_key_input.startswith("sk-"):
                st.session_state.openai_api_key = api_key_input
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API key (starts with 'sk-')")
    return False

# Initialize OpenAI Client based on available API key
def initialize_openai_client():
    """Initialize OpenAI client with session API key."""
    api_key = get_api_key()
    if api_key:
        try:
            client = OpenAI(api_key=api_key, http_client=httpx.Client(trust_env=False))
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None
    return None

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

# --- Simple Emotion Analysis Setup ---
@st.cache_resource
def load_emotion_model():
    """Load emotion analysis model with extended timeout and retry logic."""
    if not AUDIO_LIBS_AVAILABLE:
        st.warning("Audio processing libraries not available - emotion analysis disabled")
        return None
        
    import time
    
    try:
        # Configure for cloud environment
        if IS_STREAMLIT_CLOUD:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes for cloud
            os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
            os.environ["HF_HOME"] = "/tmp/huggingface"
        else:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes for local
        
        # Add retry logic
        max_retries = 2 if IS_STREAMLIT_CLOUD else 3
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Loading emotion model (attempt {attempt + 1}/{max_retries})..."):
                    emotion_pipeline = pipeline(
                        "audio-classification", 
                        model="superb/hubert-large-superb-er",
                        return_all_scores=True,
                        device=-1  # Force CPU usage
                    )
                    return emotion_pipeline
            except Exception as e:
                if attempt < max_retries - 1:
                    st.info(f"Model download attempt {attempt + 1} failed, retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    raise e
                    
    except Exception as e:
        st.error(f"Failed to load emotion model after {max_retries} attempts: {e}")
        st.error("**Emotion analysis will not work without the transformer model**")
        if IS_STREAMLIT_CLOUD:
            st.info("This is likely due to Streamlit Cloud resource limitations. The app will work without emotion analysis.")
        else:
            st.info("The app requires HuBERT model for 95% transformer + 5% rule-based analysis")
        return None

def extract_audio_features(audio_data, sr=16000):
    """Extract simple rule-based audio features."""
    try:
        # Energy (RMS)
        energy = np.sqrt(np.mean(audio_data**2))
        
        # Spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        
        # Zero crossing rate (speech rate indicator)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Normalize features
        energy_norm = min(energy * 10, 1.0)  # Scale energy
        brightness_norm = min(spectral_centroid / 2000, 1.0)  # Scale spectral centroid
        speech_rate_norm = min(zcr * 100, 1.0)  # Scale ZCR
        
        return {
            'energy': energy_norm,
            'brightness': brightness_norm, 
            'speech_rate': speech_rate_norm
        }
    except:
        return {'energy': 0.5, 'brightness': 0.5, 'speech_rate': 0.5}

def get_rule_based_emotions(features):
    """Get rule-based emotion predictions with confidence scores."""
    # SUPERB model uses: neu, hap, ang, sad
    emotions = {
        'neu': 0.0, 'hap': 0.0, 'ang': 0.0, 'sad': 0.0
    }
    
    energy = features['energy']
    brightness = features['brightness']
    speech_rate = features['speech_rate']
    
    # Rule-based scoring for SUPERB emotion categories
    if energy > 0.7 and speech_rate > 0.6:
        emotions['ang'] = 0.8  # High energy + fast speech = angry
    
    if energy < 0.3:
        emotions['sad'] = 0.9  # Low energy = sad
    
    if energy > 0.6 and brightness > 0.7:
        emotions['hap'] = 0.85  # High energy + brightness = happy
    
    if energy > 0.3 and energy < 0.7 and speech_rate < 0.5:
        emotions['neu'] = 0.6  # Moderate energy + slow speech = neutral
    
    # Ensure neutral has some baseline probability
    if max(emotions.values()) < 0.3:
        emotions['neu'] = 0.5
    
    return emotions

def analyze_emotion_hybrid(audio_file_path, emotion_model):
    """Simple hybrid emotion analysis with confidence scores and debug info."""
    if not emotion_model:
        st.error("Emotion analysis unavailable: Transformer model failed to load")
        st.info("Please try refreshing the page or contact support if the issue persists")
        return [("neu", 50.0), ("neu", 0.0)], None
    
    try:
        # Load audio
        audio_data, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Transformer prediction
        transformer_results = emotion_model(audio_file_path)
        transformer_emotions = {}
        if transformer_results:
            for result in transformer_results:
                transformer_emotions[result['label']] = result['score']
        else:
            transformer_emotions = {'neu': 0.5}
        
        # Rule-based features and predictions
        features = extract_audio_features(audio_data, sr)
        rule_emotions = get_rule_based_emotions(features)
        
        # Combine both approaches (weighted average)
        combined_emotions = {}
        all_emotion_keys = set(list(transformer_emotions.keys()) + list(rule_emotions.keys()))
        
        for emotion in all_emotion_keys:
            transformer_score = transformer_emotions.get(emotion, 0.0)
            rule_score = rule_emotions.get(emotion, 0.0)
            # Weight: 95% transformer, 5% rule-based
            combined_score = (transformer_score * 0.95) + (rule_score * 0.05)
            combined_emotions[emotion] = combined_score
        
        # Get top 2 emotions
        sorted_emotions = sorted(combined_emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to percentages
        top_emotion = (sorted_emotions[0][0], sorted_emotions[0][1] * 100)
        second_emotion = (sorted_emotions[1][0], sorted_emotions[1][1] * 100) if len(sorted_emotions) > 1 else ("neu", 0.0)
        
        # Create debug information
        debug_info = {
            'audio_features': features,
            'transformer_scores': transformer_emotions,
            'rule_based_scores': rule_emotions,
            'combined_scores': combined_emotions,
            'final_ranking': sorted_emotions
        }
        
        return [top_emotion, second_emotion], debug_info
        
    except Exception as e:
        st.error(f"Error during emotion analysis: {e}")
        return [("neu", 50.0), ("neu", 0.0)], None

# Load emotion model
emotion_model = load_emotion_model()

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

def display_emotion_debug_sidebar(debug_info):
    """Display only the emotion comparison table in sidebar."""
    if not debug_info:
        return
    
    # Get all emotions
    all_emotions = set()
    all_emotions.update(debug_info['transformer_scores'].keys())
    all_emotions.update(debug_info['rule_based_scores'].keys())
    all_emotions.update(debug_info['combined_scores'].keys())
    
    # Emotion emoji mapping
    emotion_emoji = {
        "neu": "😐", "hap": "😊", "ang": "😠", "sad": "😢"
    }
    
    # Create comparison data
    comparison_data = []
    for emotion in sorted(all_emotions):
        transformer_score = debug_info['transformer_scores'].get(emotion, 0.0) * 100
        rule_score = debug_info['rule_based_scores'].get(emotion, 0.0) * 100
        combined_score = debug_info['combined_scores'].get(emotion, 0.0) * 100
        
        emoji = emotion_emoji.get(emotion, "🎭")
        emotion_name = {
            'neu': 'Neutral',
            'hap': 'Happy', 
            'ang': 'Angry',
            'sad': 'Sad'
        }.get(emotion, emotion.title())
        
        comparison_data.append({
            'Emotion': f"{emoji} {emotion_name}",
            'Transformer (95%)': f"{transformer_score:.1f}%",
            'Rule-Based (5%)': f"{rule_score:.1f}%", 
            'Combined Score': f"{combined_score:.1f}%"
        })
    
    # Sort by combined score
    comparison_data.sort(key=lambda x: float(x['Combined Score'].replace('%', '')), reverse=True)
    st.dataframe(comparison_data, hide_index=True)
    
    # Show calculation explanation
    st.caption("**Calculation:** Combined Score = (Transformer × 0.95) + (Rule-Based × 0.05)")
    
    # Show winning emotion
    top_emotion = comparison_data[0]
    st.success(f"**Result:** {top_emotion['Emotion']} with {top_emotion['Combined Score']} confidence")

# --- UI Components ---
# Display logo
import base64
with open("KL-Logo.png", "rb") as logo_file:
    logo_base64 = base64.b64encode(logo_file.read()).decode()

st.markdown(f"""
<div class="logo-container">
    <img src="data:image/png;base64,{logo_base64}" class="app-logo" alt="KadelLabs Logo">
</div>
""", unsafe_allow_html=True)

st.title("VoiceBot with Emotional Intelligence")

# Add gradient background to main page
st.markdown("""
<style>
.stApp {
    background: white;
}

/* Position logo outside normal markdown flow */
.logo-container {
    position: fixed !important;
    top: 10px !important;
    right: 15px !important;
    z-index: 1000 !important;
    margin: 0 !important;
    padding: 40px 15px 0 0 !important;
    transition: right 0.3s ease !important;
}

/* Logo will move with main content when sidebar opens - no additional CSS needed */

.app-logo {
    height: 75px !important;
    width: auto !important;
    display: block !important;
}

/* Constrain audiorecorder widget width */
div[data-testid="stVerticalBlock"] > div:has(button[aria-label*="record"]) {
    max-width: 100px !important;
    width: 100px !important;
}

/* Hide extra space in audiorecorder */
.audiorecorder {
    max-width: 100px !important;
}

/* Make sure the recording button container doesn't take full width */
iframe[title="audiorecorder.audiorecorder"] {
    max-width: 200px !important;
    width: 200px !important;
}

/* Change all button backgrounds to Soft Lavender */
.stButton > button {
    background-color: #E6E6FA !important;
    border: 1px solid #D8D8F0 !important;
}

.stButton > button:hover {
    background-color: #DCDCF7 !important;
    border: 1px solid #C8C8E8 !important;
}

/* Target audiorecorder button specifically (stubborn one) */
iframe[title="audiorecorder.audiorecorder"] button,
.audiorecorder button,
button[aria-label*="record"] {
    background-color: #E6E6FA !important;
    border: 1px solid #D8D8F0 !important;
}

iframe[title="audiorecorder.audiorecorder"] button:hover,
.audiorecorder button:hover,
button[aria-label*="record"]:hover {
    background-color: #DCDCF7 !important;
    border: 1px solid #C8C8E8 !important;
}

/* Target any buttons within the audiorecorder iframe */
iframe[title="audiorecorder.audiorecorder"] {
    filter: hue-rotate(240deg) saturate(0.3) brightness(1.1) !important;
}

/* Change sidebar background color */
.css-1d391kg, .css-1cypcdb, .css-17eq0hr, section[data-testid="stSidebar"] {
    background-color: #F3F0FF !important;
}

/* Ensure sidebar content is visible on new background */
section[data-testid="stSidebar"] > div {
    background-color: #F3F0FF !important;
}

/* Make emotional analysis table background white */
section[data-testid="stSidebar"] table {
    background-color: white !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 0.5rem !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] table thead,
section[data-testid="stSidebar"] table tbody,
section[data-testid="stSidebar"] table tr,
section[data-testid="stSidebar"] table td,
section[data-testid="stSidebar"] table th {
    background-color: white !important;
    border: none !important;
}

/* Center and format main page title */
h1 {
    text-align: center !important;
    font-size: 2.5rem !important;
    font-family: "Trans Serif", serif !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

/* Responsive title sizing for smaller screens */
@media (max-width: 768px) {
    h1 {
        font-size: 1.8rem !important;
    }
}

@media (max-width: 480px) {
    h1 {
        font-size: 1.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# API Key Management in Sidebar
with st.sidebar:
    st.markdown("### API Key Management")
    
    # Show masked API key
    current_key = get_api_key()
    if current_key:
        masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "sk-***"
        st.success(f"API Key: `{masked_key}`")
    
    # Option to change API key
    if st.button("Change API Key", use_container_width=True):
        if "openai_api_key" in st.session_state:
            del st.session_state.openai_api_key
        st.rerun()
    
    st.markdown("---")
    
    # Emotion Analysis Debug Section
    st.markdown("### Latest Emotion Analysis")
    if st.session_state.latest_debug_info:
        display_emotion_debug_sidebar(st.session_state.latest_debug_info)
    else:
        st.info("Record some audio to see emotion analysis details here.")
    
    st.markdown("---")
    
    # Clear Chat History Button
    if st.button("Clear Chat History", key="clear_chat_sidebar", use_container_width=True):
        st.session_state.messages = []
        st.session_state.messages.append({"role": "system", "content": "You are a helpful assistant. Keep your responses concise and to the point."})
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})
        st.session_state.ai_audio_file = None  # Also clear the AI audio
        st.session_state.status_message = ""  # Clear status
        st.session_state.latest_debug_info = None  # Clear debug info
        st.rerun()

# Display existing chat messages
# Iterate through messages, skipping the initial system message for display
for message in st.session_state.messages:
    if message["role"] == "user":
        # Check if message has emotion data
        if isinstance(message.get("content"), dict) and "emotion_analysis" in message["content"]:
            # SUPERB emotion mapping
            emotion_emoji = {
                "neu": "😐", "hap": "😊", "ang": "😠", "sad": "😢"
            }
            text = message["content"]["text"]
            emotion_analysis = message["content"]["emotion_analysis"]
            
            # Get top 2 emotions
            top_emotion, top_confidence = emotion_analysis[0]
            second_emotion, second_confidence = emotion_analysis[1]
            
            top_emoji = emotion_emoji.get(top_emotion, "🎭")
            
            emotion_display = f" {top_emoji}{top_confidence:.1f}% "
            st.markdown(f"**You:** {text}{emotion_display}")
        else:
            st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:** {message['content']}")

# --- Audio Recording and Processing Function ---
def handle_audio_input():
    """Handle audio recording and processing."""
    # Check if we have pending audio to process
    if st.session_state.pending_audio is not None and st.session_state.is_processing_audio:
        # Process the pending audio
        process_pending_audio()
        return
    
    # Only show audiorecorder when not processing
    if not st.session_state.is_processing_audio:
        audio = audiorecorder("Click to record", "Click to stop recording", key=f"audio_recorder_{st.session_state.audio_counter}")
        
        if len(audio) > 0:
            current_audio_len = len(audio)
            if current_audio_len != st.session_state.last_processed_audio_len:
                st.session_state.last_processed_audio_len = current_audio_len
                st.session_state.ai_audio_file = None
                
                # Store audio for processing and set processing state
                st.session_state.pending_audio = audio
                st.session_state.is_processing_audio = True
                st.rerun()  # Trigger rerun to hide audiorecorder and start processing
    else:
        # Show processing message instead of audiorecorder
        st.info("Processing your audio... Please wait.")

def process_pending_audio():
    """Process the pending audio data."""
    audio = st.session_state.pending_audio
    
    # Save the recorded audio
    audio_filename = "input.wav"
    try:
        audio.export(audio_filename, format="wav")
    except Exception as e:
        st.error(f"Error saving recorded audio: {e}")
        # Reset processing state on error
        st.session_state.is_processing_audio = False
        st.session_state.pending_audio = None
        return

    # Analyze emotion
    emotion_analysis, debug_info = analyze_emotion_hybrid(audio_filename, emotion_model)

    # Store debug info for sidebar display
    if debug_info:
        st.session_state.latest_debug_info = debug_info

    # Transcribe audio
    st.session_state.status_message = "Transcribing your speech..."
    with st.spinner("Transcribing your speech..."):
        user_text = transcribe_audio(audio_filename)
    if user_text:
        # Display with emotion analysis
        # SUPERB emotion mapping
        emotion_emoji = {
            "neu": "😐", "hap": "😊", "ang": "😠", "sad": "😢"
        }
        
        top_emotion, top_confidence = emotion_analysis[0]
        second_emotion, second_confidence = emotion_analysis[1]
        
        top_emoji = emotion_emoji.get(top_emotion, "🎭")
        
        emotion_display = f" {top_emoji}{top_confidence:.1f}% "
        st.markdown(f"**You:** {user_text}{emotion_display}")
        
        # Store message with emotion analysis and debug info
        st.session_state.messages.append({
            "role": "user", 
            "content": {
                "text": user_text,
                "emotion_analysis": emotion_analysis,
                "debug_info": debug_info
            }
        })
        st.session_state.status_message = ""
    else:
        st.warning("Could not transcribe your audio. Please try again.")
        st.session_state.status_message = ""
        # Reset processing state on error
        st.session_state.is_processing_audio = False
        st.session_state.pending_audio = None
        return

    # Get AI response
    st.session_state.status_message = "AI is thinking..."
    with st.spinner("AI is thinking..."):
        ai_reply = get_gpt_response(st.session_state.messages)
    if ai_reply:
        st.markdown(f"**Assistant:** {ai_reply}")
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.session_state.status_message = ""

        # Generate AI voice
        st.session_state.status_message = "Generating AI voice..."
        with st.spinner("Generating AI voice..."):
            ai_audio_filepath = text_to_speech(ai_reply)
        if ai_audio_filepath:
            st.session_state.ai_audio_file = ai_audio_filepath
            st.session_state.status_message = ""
        else:
            st.warning("Could not generate AI voice.")
            st.session_state.ai_audio_file = None
            st.session_state.status_message = ""
    else:
        st.warning("AI did not generate a response.")
        st.session_state.ai_audio_file = None
        st.session_state.status_message = ""

    # Reset processing state and increment counter
    st.session_state.is_processing_audio = False
    st.session_state.pending_audio = None
    st.session_state.audio_counter += 1
    st.rerun()

# --- Audio Transcription Function ---
def transcribe_audio(filename):
    """Transcribe audio using OpenAI Whisper."""
    if not filename:
        return None
    try:
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None

# --- GPT Response Function ---
def get_gpt_response(messages):
    """Get GPT response with emotion context."""
    try:
        # Convert messages with emotion data to include emotion context for GPT
        emotion_aware_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Update system prompt to be emotion-aware
                emotion_aware_messages.append({
                    "role": "system", 
                    "content": "You are a helpful and emotionally intelligent assistant. You will receive user messages along with their detected emotions (like happy, sad, angry, neutral). Please respond appropriately considering both the content and the emotional state of the user. Keep your responses concise and to the point, while being empathetic to their emotions."
                })
            elif isinstance(msg.get("content"), dict) and "text" in msg["content"]:
                # Extract text and emotion data
                text = msg["content"]["text"]
                emotion_analysis = msg["content"].get("emotion_analysis", [])
                
                if emotion_analysis:
                    # Get top emotion
                    top_emotion, top_confidence = emotion_analysis[0]
                    
                    # Map emotion codes to readable names
                    emotion_names = {
                        'neu': 'neutral',
                        'hap': 'happy', 
                        'ang': 'angry',
                        'sad': 'sad'
                    }
                    emotion_name = emotion_names.get(top_emotion, top_emotion)
                    
                    # Include emotion context in the message
                    enhanced_content = f"{text}\n[User emotion: {emotion_name} ({top_confidence:.1f}% confidence)]"
                    emotion_aware_messages.append({"role": msg["role"], "content": enhanced_content})
                else:
                    emotion_aware_messages.append({"role": msg["role"], "content": text})
            else:
                emotion_aware_messages.append(msg)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=emotion_aware_messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while getting AI response: {e}")
        return "I'm sorry, I couldn't generate a response at this moment."

# --- Text-to-Speech Function ---
def text_to_speech(text, output_filename="reply.mp3"):
    """Convert text to speech using OpenAI TTS."""
    try:
        speech_file_path = Path(__file__).parent / output_filename
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", 
            input=text
        )
        with open(speech_file_path, "wb") as f:
            f.write(response.content)
        st.success("AI response converted to speech!")
        return str(speech_file_path)
    except Exception as e:
        st.error(f"Error generating AI voice: {e}")
        return None

# --- Main App Logic ---
# Display AI audio if available (persists across reruns)
if st.session_state.ai_audio_file:
    st.markdown("### AI Voice Response")
    st.audio(st.session_state.ai_audio_file, format='audio/mp3')

handle_audio_input() # Call the function that contains the audiorecorder widget