import streamlit as st
import base64
from pathlib import Path

def display_logo():
    """Display the application logo."""
    logo_path = Path(__file__).parent.parent / "KL-Logo.png"
    with open(logo_path, "rb") as logo_file:
        logo_base64 = base64.b64encode(logo_file.read()).decode()

    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" class="app-logo" alt="KadelLabs Logo">
    </div>
    """, unsafe_allow_html=True)

def apply_custom_styles():
    """Apply custom CSS styles to the application."""
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

    /* Hide inactive audiorecorder buttons */
    div[data-testid="stAudioRecorder"]:not(:last-child) {
        display: none !important;
    }
    /* Ensure the active button is visible */
    div[data-testid="stAudioRecorder"]:last-child {
        display: block !important;
    }

    /* Main container styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styles */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Status message styles */
    .status-message {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        text-align: center;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    
    /* Hide inactive audiorecorder buttons completely */
    button[aria-label="Click to record"]:disabled,
    button[aria-label="Click to record"][disabled] {
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with API key management and emotion analysis."""
    with st.sidebar:
        st.markdown("### API Key Management")
        
        # Show masked API key
        from .openai_services import get_api_key
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
            from .emotion_analysis import display_emotion_debug_sidebar
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

def display_chat_messages():
    """Display the chat messages with emotion analysis."""
    # Create a dedicated container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Iterate through messages, skipping the initial system message for display
        for i, message in enumerate(st.session_state.messages):
            # Skip the most recent AI message if it's pending
            if (message["role"] == "assistant" and 
                i == len(st.session_state.messages) - 1 and 
                hasattr(st.session_state, 'pending_ai_response')):
                continue
                
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

def display_ai_audio():
    """Display the AI audio player if available."""
    if st.session_state.ai_audio_file and hasattr(st.session_state, 'pending_ai_response'):
        # Display the text response
        st.markdown(f"**Assistant:** {st.session_state.pending_ai_response}")
        
        # Create a container for the audio player
        audio_container = st.container()
        
        with audio_container:
            # Add the autoplay audio element with controls
            st.markdown(f"""
            <audio id="ai_audio" autoplay controls>
                <source src="data:audio/mp3;base64,{get_base64_audio(st.session_state.ai_audio_file)}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            <script>
                document.getElementById('ai_audio').onended = function() {{
                    // Trigger a custom event when audio ends
                    window.dispatchEvent(new Event('audioEnded'));
                }};
            </script>
            """, unsafe_allow_html=True)
        
        # Clear the pending response after displaying
        del st.session_state.pending_ai_response

def get_base64_audio(file_path):
    """Convert audio file to base64 for embedding."""
    import base64
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode() 