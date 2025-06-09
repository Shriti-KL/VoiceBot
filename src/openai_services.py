import streamlit as st
from openai import OpenAI
from pathlib import Path

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

def initialize_openai_client():
    """Initialize OpenAI client with session API key."""
    api_key = get_api_key()
    if api_key:
        try:
            import httpx
            client = OpenAI(api_key=api_key, http_client=httpx.Client(trust_env=False))
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None
    return None

def transcribe_audio(filename):
    """Transcribe audio using OpenAI Whisper."""
    if not filename:
        return None
    try:
        with open(filename, "rb") as audio_file:
            transcript = st.session_state.openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None

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
        
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4",
            messages=emotion_aware_messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while getting AI response: {e}")
        return "I'm sorry, I couldn't generate a response at this moment."

def text_to_speech(text, output_filename="reply.mp3"):
    """Convert text to speech using OpenAI TTS."""
    try:
        speech_file_path = Path(__file__).parent.parent / output_filename
        response = st.session_state.openai_client.audio.speech.create(
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