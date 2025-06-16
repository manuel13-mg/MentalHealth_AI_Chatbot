import streamlit as st
import os
import sys
import subprocess
from groq import Groq
import traceback
import re
import warnings
from datetime import datetime

# --- Dependency Imports with Graceful Fallbacks ---
try:
    import pyttsx3
    TTS_AVAILABLE = True
except (ImportError, RuntimeError):
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="InnerCompass 🌟",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced CSS Styling - Black & Blue Theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1e40af 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main > div {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        padding: 2rem;
        margin: 1rem;
    }

    /* Title styling */
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        background: linear-gradient(135deg, #60a5fa, #3b82f6, #1d4ed8) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center !important;
        letter-spacing: -0.02em !important;
    }

    /* Subtitle styling */
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 30px;
        opacity: 0.9;
    }

    /* Crisis alert styling */
    .crisis-alert {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #fca5a5;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
    }

    /* Stats box styling */
    .stats-box, .mood-tracker, .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin-top: 10px;
        height: 100%;
        color: #93c5fd;
    }

    /* Chat input styling */
    .stChatInput > div {
        background: rgba(30, 41, 59, 0.9) !important;
        border-radius: 25px !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
    }
    .stChatInput input {
        color: #e2e8f0 !important;
    }

    /* Chat message styling */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        border-radius: 20px 20px 8px 20px !important;
        margin-left: 30% !important;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #1e293b, #334155) !important;
        border-radius: 20px 20px 20px 8px !important;
        margin-right: 25% !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
        border-radius: 25px !important;
        width: 100%;
    }

    /* Sidebar styling */
    .stSidebar {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    .stSidebar > div:first-child {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    div[data-testid="stMarkdownContainer"] p, .stats-box p, .mood-tracker p, .info-box p { color: inherit !important; }
    .crisis-alert p, .crisis-alert h3, .crisis-alert li { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper Classes & Functions ---

def speak_in_subprocess(text_to_speak, rate):
    if not TTS_AVAILABLE or not text_to_speak.strip(): return
    clean_text = re.sub(r'[\*#`]', '', text_to_speak)
    tts_script = """
import sys, pyttsx3
try:
    text, rate = sys.argv[1], int(sys.argv[2])
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()
except Exception: sys.exit(1)
"""
    try:
        subprocess.run([sys.executable, "-c", tts_script, clean_text, str(rate)], check=True, timeout=60)
    except Exception as e:
        st.error(f"Failed to play audio: {e}", icon="🔊")

class VoiceRecognitionManager:
    def __init__(self):
        self.is_initialized = False
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.is_initialized = True
            except Exception:
                pass

    # THE FIX IS HERE: The missing method is now restored.
    def listen_and_transcribe(self):
        if not self.is_initialized:
            return None, "Voice recognition not available."
        try:
            with self.microphone as source:
                st.toast("Listening...", icon="🎤")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
            text = self.recognizer.recognize_google(audio)
            return text, None
        except sr.WaitTimeoutError:
            return None, "No speech detected."
        except sr.UnknownValueError:
            return None, "Could not understand audio."
        except Exception as e:
            return None, f"An unknown error occurred: {e}"

class CrisisDetector:
    def __init__(self):
        self.crisis_keywords = ['suicide', 'kill myself', 'end it all', 'hurt myself', 'self harm', 'want to die', 'end my life']
    def detect_crisis(self, text):
        return any(re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()) for keyword in self.crisis_keywords)

class MoodTracker:
    def __init__(self):
        self.mood_keywords = {
            'very_positive': ['amazing', 'fantastic', 'wonderful', 'excellent', 'ecstatic'],
            'positive': ['good', 'happy', 'glad', 'pleased', 'content', 'optimistic'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'average'],
            'negative': ['sad', 'down', 'upset', 'disappointed', 'anxious'],
            'very_negative': ['terrible', 'awful', 'horrible', 'devastated', 'depressed', 'hopeless']
        }
    def analyze_mood(self, text):
        text_lower = text.lower()
        for mood, keywords in self.mood_keywords.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords):
                scores = {'very_positive': 5, 'positive': 4, 'neutral': 3, 'negative': 2, 'very_negative': 1}
                return mood, scores.get(mood, 3)
        return 'neutral', 3

class ConversationExporter:
    @staticmethod
    def export_to_text(messages, stats):
        duration = str(datetime.now() - stats['session_start']).split('.')[0]
        export_text = f"InnerCompass Conversation\n- User Messages: {stats['user_messages']} | Bot Responses: {stats['bot_responses']}\n- Duration: {duration}\n\n{'='*50}\n\n"
        for msg in messages:
            role = "You" if msg["role"] == "user" else "InnerCompass"
            export_text += f"[{role}]:\n{msg['content']}\n\n"
        return export_text

# --- API & Component Initialization ---
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY") or "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")
    if 'voice_manager' not in st.session_state: st.session_state.voice_manager = VoiceRecognitionManager()
    if 'crisis_detector' not in st.session_state: st.session_state.crisis_detector = CrisisDetector()
    if 'mood_tracker' not in st.session_state: st.session_state.mood_tracker = MoodTracker()
except Exception as e:
    st.error(f"Failed to initialize. Please check your API key and dependencies. Error: {e}")
    st.stop()

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state: st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
if "mood_history" not in st.session_state: st.session_state.mood_history = []
if "transcribed_text" not in st.session_state: st.session_state.transcribed_text = None
if "response_to_speak" not in st.session_state: st.session_state.response_to_speak = None
if "app_settings" not in st.session_state:
    st.session_state.app_settings = {
        "model": "llama3-8b-8192", "temperature": 0.7,
        "voice_input_enabled": SPEECH_RECOGNITION_AVAILABLE and st.session_state.voice_manager.is_initialized,
        "tts_enabled": TTS_AVAILABLE, "auto_speak": True, "voice_rate": 180,
        "mood_tracking": True, "crisis_detection": True,
        "system_prompt": "You are InnerCompass, a kind, empathetic AI companion. Your goal is to provide emotional support. Listen, validate feelings, and offer gentle, positive advice. Keep responses concise, conversational, and non-judgmental. If crisis language is detected, prioritize safety and suggest professional resources."
    }

# --- Sidebar UI ---
with st.sidebar:
    st.title("🎛️ Settings")
    st.subheader("🤖 AI Model")
    GROQ_MODELS = {"llama3-8b-8192": "Llama 3 8B", "llama3-70b-8192": "Llama 3 70B", "mixtral-8x7b-32768": "Mixtral 8x7B"}
    st.session_state.app_settings["model"] = st.selectbox("Choose Model:", options=list(GROQ_MODELS.keys()), format_func=lambda x: GROQ_MODELS[x])
    st.session_state.app_settings["temperature"] = st.slider("Creativity:", 0.1, 1.5, st.session_state.app_settings["temperature"])
    
    st.subheader("🚀 Features")
    st.session_state.app_settings["mood_tracking"] = st.checkbox("Enable Mood Tracking", st.session_state.app_settings["mood_tracking"])
    st.session_state.app_settings["crisis_detection"] = st.checkbox("Enable Crisis Detection", st.session_state.app_settings["crisis_detection"])
    
    if TTS_AVAILABLE:
        st.subheader("🔊 Voice Output")
        st.session_state.app_settings["tts_enabled"] = st.checkbox("Enable Voice Output", st.session_state.app_settings["tts_enabled"])
        if st.session_state.app_settings["tts_enabled"]:
            st.session_state.app_settings["auto_speak"] = st.checkbox("Auto-speak responses", st.session_state.app_settings["auto_speak"])
            st.session_state.app_settings["voice_rate"] = st.slider("Speech Speed:", 100, 300, st.session_state.app_settings["voice_rate"])

    if SPEECH_RECOGNITION_AVAILABLE:
        st.subheader("🎤 Voice Input")
        st.session_state.app_settings["voice_input_enabled"] = st.checkbox("Enable Voice Input", st.session_state.app_settings["voice_input_enabled"], disabled=not st.session_state.voice_manager.is_initialized)

# --- Main Application UI ---
st.title("InnerCompass 🌟")
st.markdown('<p class="subtitle">Your empathetic AI companion for thoughts and feelings.</p>', unsafe_allow_html=True)
crisis_placeholder = st.empty()

# --- Dashboard & Controls ---
col1, col2, col3 = st.columns([1.5, 1.5, 1])
with col1:
    stats = st.session_state.conversation_stats
    duration = str(datetime.now() - stats['session_start']).split('.')[0]
    st.markdown(f'<div class="stats-box"><h4>📊 Session Stats</h4><p><strong>Messages:</strong> You: {stats["user_messages"]} | Bot: {stats["bot_responses"]}<br><strong>Duration:</strong> {duration}</p></div>', unsafe_allow_html=True)

with col2:
    if st.session_state.app_settings["mood_tracking"] and st.session_state.mood_history:
        last_mood, last_score = st.session_state.mood_history[-1]
        mood_map = {'very_positive': '😃', 'positive': '😊', 'neutral': '😐', 'negative': '😟', 'very_negative': '😞'}
        st.markdown(f'<div class="mood-tracker"><h4>😊 Mood Snapshot</h4><p><strong>Last detected mood:</strong> {last_mood.replace("_", " ").title()} {mood_map.get(last_mood, "❓")}</p></div>', unsafe_allow_html=True)
        st.progress(last_score / 5.0)
    else:
        st.markdown('<div class="info-box"><h4>💡 Tip</h4><p>Enable mood tracking in the sidebar to see live insights about your conversation.</p></div>', unsafe_allow_html=True)

with col3:
    if st.button("✨ New Chat"):
        st.session_state.messages, st.session_state.mood_history = [], []
        st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
        st.rerun()
    if st.session_state.messages:
        export_text = ConversationExporter.export_to_text(st.session_state.messages, st.session_state.conversation_stats)
        st.download_button("📥 Export Text", export_text, f"InnerCompass_Chat_{datetime.now().strftime('%Y%m%d')}.txt")

# --- Chat Interface & Logic ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- TTS Playback Block ---
if st.session_state.response_to_speak:
    response = st.session_state.response_to_speak
    st.session_state.response_to_speak = None
    if st.session_state.app_settings["tts_enabled"] and st.session_state.app_settings["auto_speak"]:
        speak_in_subprocess(response, rate=st.session_state.app_settings["voice_rate"])

# --- Input Block ---
prompt = None
if st.session_state.app_settings["voice_input_enabled"]:
    if st.button("🎤 Click to Speak"):
        text, error = st.session_state.voice_manager.listen_and_transcribe()
        if text:
            st.session_state.transcribed_text = text
            st.rerun() # Rerun to process the transcribed text
        elif error:
            st.toast(f"Could not hear you: {error}", icon="❌")

prompt = st.chat_input("How are you feeling today?") or st.session_state.transcribed_text

# --- Processing Block ---
if prompt:
    st.session_state.transcribed_text = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_stats["user_messages"] += 1
    
    is_crisis = st.session_state.app_settings["crisis_detection"] and st.session_state.crisis_detector.detect_crisis(prompt)
    
    if is_crisis:
        with crisis_placeholder.container():
            st.markdown('<div class="crisis-alert"><h3>🚨 Immediate Help is Available</h3><p>It sounds like you are in distress. Please reach out to a professional who can support you.</p><ul><li><strong>USA/Canada:</strong> Call or text 988</li><li><strong>Crisis Text Line:</strong> Text HOME to 741741</li></ul></div>', unsafe_allow_html=True)
        safety_response = "I am very concerned by what you've shared. Your safety is the highest priority. Please reach out to one of the crisis resources listed above. They are available 24/7 to help you."
        st.session_state.messages.append({"role": "assistant", "content": safety_response})
        st.session_state.response_to_speak = safety_response
    else:
        if st.session_state.app_settings["mood_tracking"]:
            mood, score = st.session_state.mood_tracker.analyze_mood(prompt)
            st.session_state.mood_history.append((mood, score))

        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("InnerCompass is thinking..."):
                    stream = groq_client.chat.completions.create(
                        model=st.session_state.app_settings["model"],
                        messages=[{"role": "system", "content": st.session_state.app_settings["system_prompt"]}] + st.session_state.messages,
                        temperature=st.session_state.app_settings["temperature"],
                        stream=True,
                    )
                    full_response = ""
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.response_to_speak = full_response
        except Exception as e:
            st.error(f"An API error occurred: {e}")
            traceback.print_exc()

    st.session_state.conversation_stats["bot_responses"] += 1
    st.rerun()