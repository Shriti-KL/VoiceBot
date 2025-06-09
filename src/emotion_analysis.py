import streamlit as st
import os
import sys
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

# Cloud environment detection
IS_STREAMLIT_CLOUD = "STREAMLIT_CLOUD" in os.environ or "streamlit" in sys.modules

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