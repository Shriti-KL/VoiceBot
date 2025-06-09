import streamlit as st
from audiorecorder import audiorecorder
from pathlib import Path

def handle_audio_input():
    """Handle audio recording and processing."""
    # Add JavaScript to handle audio end event
    st.markdown("""
    <script>
        window.addEventListener('audioEnded', function() {
            // Force a rerun when audio ends
            window.streamlitRerun();
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Create a dedicated container for audio input
    audio_input_container = st.container()
    
    # Check if we have pending audio to process
    if st.session_state.pending_audio is not None and st.session_state.is_processing_audio:
        # Process the pending audio
        process_pending_audio()
        return
    
    # Show audiorecorder when not processing and no AI audio is being displayed
    if not st.session_state.is_processing_audio and not hasattr(st.session_state, 'pending_ai_response'):
        with audio_input_container:
            # Clear any previous content in the container
            audio_input_container.empty()
            
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
        with audio_input_container:
            # Clear any previous content in the container
            audio_input_container.empty()
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
    from .emotion_analysis import analyze_emotion_hybrid
    emotion_analysis, debug_info = analyze_emotion_hybrid(audio_filename, st.session_state.emotion_model)

    # Store debug info for sidebar display
    if debug_info:
        st.session_state.latest_debug_info = debug_info

    # Transcribe audio
    st.session_state.status_message = "Transcribing your speech..."
    with st.spinner("Transcribing your speech..."):
        from .openai_services import transcribe_audio
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
        from .openai_services import get_gpt_response
        ai_reply = get_gpt_response(st.session_state.messages)
    if ai_reply:
        # Store the AI response in session state without displaying it yet
        st.session_state.pending_ai_response = ai_reply
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.session_state.status_message = ""

        # Generate AI voice
        st.session_state.status_message = "Generating AI voice..."
        with st.spinner("Generating AI voice..."):
            from .openai_services import text_to_speech
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