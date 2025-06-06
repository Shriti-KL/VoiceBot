# VoiceBot with Emotional Intelligence

A sophisticated voice chat application that combines AI conversation with real-time emotion analysis, providing an emotionally intelligent conversational experience.

## ✨ Features

### 🎤 Voice Interaction
- **Real-time Audio Recording**: Click-to-record interface for seamless voice input
- **Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **Text-to-Speech**: Natural AI voice responses using OpenAI TTS

### 🧠 Emotional Intelligence
- **Hybrid Emotion Analysis**: Combines transformer-based (HuBERT) and rule-based approaches
- **Real-time Detection**: Analyzes emotions from voice tone, energy, and speech patterns
- **Emotion-Aware Responses**: AI adapts responses based on detected emotional state
- **Detailed Analytics**: Comprehensive emotion breakdown with confidence scores

### 🔒 Security & Privacy
- **Session-only API Keys**: No permanent storage of credentials
- **Manual Key Input**: Always requires fresh API key entry (no environment fallbacks)
- **Secure Storage**: Keys cleared when browser session ends

### 🎨 User Experience
- **Modern UI**: Beautiful gradient background with soft lavender color scheme
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Processing States**: Clear feedback during audio processing
- **Clean Interface**: Sidebar for controls, main area for conversation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VoiceBot-AudioAnalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`
   - Enter your OpenAI API key when prompted
   - Start chatting with voice!

## 📋 Requirements

The application requires the following dependencies (see `requirements.txt`):

- `streamlit==1.32.0` - Web framework
- `openai==1.17.0` - OpenAI API client
- `streamlit-audiorecorder==0.0.6` - Audio recording widget
- `transformers==4.44.0` - Emotion analysis models
- `torch>=2.0.0` - Neural network framework
- `librosa==0.10.1` - Audio processing
- `numpy>=1.24.0` - Numerical computations
- `python-dotenv==1.0.1` - Environment management
- `httpx==0.27.0` - HTTP client for OpenAI

## 🎯 How It Works

### 1. Audio Processing Pipeline
```
🎤 Record Audio → 🧠 Emotion Analysis → 📝 Speech-to-Text → 🤖 AI Response → 🔊 Text-to-Speech
```

### 2. Emotion Analysis
- **Transformer Model**: HuBERT-large-superb-er (95% weight)
- **Rule-based Analysis**: Audio features like energy, brightness, speech rate (5% weight)
- **Emotion Categories**: Neutral, Happy, Angry, Sad
- **Real-time Feedback**: Displayed with emojis and confidence percentages

### 3. AI Integration
- **Model**: GPT-4o for intelligent responses
- **Context Awareness**: Includes emotion data in prompts
- **Empathetic Responses**: AI adapts tone based on user's emotional state

## 🎨 UI Components

### Main Page
- **Centered Title**: "VoiceBot with Emotional Intelligence" in Trans Serif font
- **Voice Recorder**: Click-to-record button with processing states
- **Chat History**: Conversation display with emotion indicators
- **AI Audio Player**: Playback of AI voice responses

### Sidebar
- **API Key Management**: Secure key input and display
- **Emotion Analytics**: Detailed breakdown of latest analysis
- **Clear Chat**: Reset conversation history

## 🔧 Configuration

### Color Scheme
- **Main Background**: Peachy gradient fade
- **Sidebar**: Light lavender (#F3F0FF)
- **Buttons**: Soft lavender (#E6E6FA)
- **Tables**: White with rounded borders

### Models Used
- **Speech-to-Text**: OpenAI Whisper-1
- **Text-to-Speech**: OpenAI TTS-1 (Alloy voice)
- **Chat**: GPT-4o
- **Emotion**: HuBERT-large-superb-er

## 🛠️ Technical Architecture

### Session State Management
- Secure API key storage
- Audio processing states
- Chat history persistence
- Emotion analysis cache

### Error Handling
- Graceful fallbacks for failed API calls
- Processing state recovery
- User-friendly error messages

### Performance Optimizations
- Cached emotion model loading
- Efficient audio processing
- Minimal UI reruns

## 📱 Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🔐 Security Notes
- API keys are never stored permanently
- Audio files are processed locally and temporarily
- No data persistence beyond browser session
- Secure HTTPS communication with OpenAI

## 🐛 Troubleshooting

### Common Issues
1. **"Failed to load emotion model"**: Ensure stable internet connection for model download
2. **"Invalid API Key"**: Verify your OpenAI API key starts with "sk-"
3. **Audio not recording**: Check browser microphone permissions
4. **Slow processing**: Large models may take time on first load

### Support
For technical issues or questions, please check:
- Browser console for error messages
- Microphone permissions in browser settings
- OpenAI API key validity and credits

## 📄 License
This project is proprietary software. All rights reserved.

---

**Built with ❤️ using Streamlit, OpenAI, and HuggingFace Transformers** 