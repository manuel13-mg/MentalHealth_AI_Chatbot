import streamlit as st
import os
from groq import Groq
import traceback
import time
import json
from datetime import datetime
import warnings
import threading
import queue
import re
warnings.filterwarnings("ignore")

# Try to import text-to-speech - graceful fallback if not available
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    # This warning will be displayed at the top of the app if pyttsx3 is not installed
    # st.warning("⚠️ Text-to-Speech not available. Install pyttsx3 for voice features: `pip install pyttsx3`")

# --- Configuration for Streamlit Page ---
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
    .stats-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: #93c5fd;
    }

    /* Info box styling */
    .info-box {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: #c4b5fd;
    }

    /* Chat input styling */
    .stChatInput > div {
        background: rgba(30, 41, 59, 0.9) !important;
        border-radius: 25px !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2) !important;
    }

    .stChatInput input {
        background: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    .stChatInput input::placeholder {
        color: #64748b !important;
    }

    /* Chat message styling */
    .stChatMessage {
        background: transparent !important;
        margin: 10px 0 !important;
    }

    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #1e40af, #3b82f6, #60a5fa) !important;
        border-radius: 20px 20px 8px 20px !important;
        color: white !important;
        margin-left: 30% !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }

    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #1e293b, #334155) !important;
        border-radius: 20px 20px 20px 8px !important;
        color: #e2e8f0 !important;
        margin-right: 25% !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    }

    /* Sidebar styling */
    .stSidebar {
        background: rgba(15, 23, 42, 0.95) !important;
    }

    .stSidebar > div:first-child {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        margin: 10px !important;
        padding: 20px !important;
    }

    /* Text colors */
    .stMarkdown, .stText, p, div {
        color: #e2e8f0;
    }
    div[data-testid="stMarkdownContainer"] p, .stats-box p, .mood-tracker p, .info-box p {
        color: inherit !important;
    }
    .crisis-alert p, .crisis-alert h3, .crisis-alert li {
        color: white !important;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Crisis Detection System ---
class CrisisDetector:
    def __init__(self):
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'hurt myself', 'self harm',
            'not worth living', 'better off dead', 'want to die', 'end my life',
            'no point in living', 'suicidal thoughts', 'planning to die'
        ]
    
    def detect_crisis(self, text):
        text_lower = text.lower()
        crisis_detected = any(re.search(r'\b' + re.escape(keyword) + r'\b', text_lower) for keyword in self.crisis_keywords)
        return {'crisis': crisis_detected}

# --- Mood Tracking System ---
class MoodTracker:
    def __init__(self):
        self.mood_keywords = {
            'very_positive': ['amazing', 'fantastic', 'wonderful', 'excellent', 'ecstatic', 'thrilled'],
            'positive': ['good', 'happy', 'glad', 'pleased', 'content', 'cheerful', 'optimistic'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'average', 'so-so'],
            'negative': ['sad', 'down', 'upset', 'disappointed', 'frustrated', 'worried', 'anxious'],
            'very_negative': ['terrible', 'awful', 'horrible', 'devastated', 'depressed', 'hopeless']
        }
    
    def analyze_mood(self, text):
        text_lower = text.lower()
        for mood, score in [('very_positive', 5), ('positive', 4), ('very_negative', 1), ('negative', 2)]:
            if mood.replace('_', ' ') in text_lower:
                return mood, score
        for mood, keywords in self.mood_keywords.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords):
                scores = {'very_positive': 5, 'positive': 4, 'neutral': 3, 'negative': 2, 'very_negative': 1}
                return mood, scores.get(mood, 3)
        return 'neutral', 3

# --- Enhanced TTS Manager ---
class ImprovedTTSManager:
    def __init__(self):
        self.tts_engine = None
        self.is_speaking = False
        self.is_initialized = False
        self.current_thread = None
        if TTS_AVAILABLE:
            self.initialize_engine()
    
    def initialize_engine(self):
        try:
            self.tts_engine = pyttsx3.init()
            if self.tts_engine:
                self.is_initialized = True
                print("TTS Engine Initialized Successfully.")
            else:
                st.error("Failed to get a TTS engine instance.")
        except Exception as e:
            st.error(f"Failed to initialize TTS engine: {e}")
            self.is_initialized = False
    
    def set_voice_settings(self, rate=180, volume=0.9):
        if not self.is_initialized: return
        try:
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
        except Exception as e:
            st.warning(f"Could not update voice settings: {e}")
    
    def clean_text_for_speech(self, text):
        clean_text = re.sub(r'[\*#`]', '', text)
        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    def speak_text(self, text):
        if not self.is_initialized or not text.strip() or self.is_speaking:
            return
            
        def speak_worker(text_to_speak):
            print("TTS THREAD: Started.")
            try:
                self.is_speaking = True
                clean_text = self.clean_text_for_speech(text_to_speak)
                if clean_text:
                    print(f"TTS THREAD: Attempting to say: '{clean_text[:60]}...'")
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                    print("TTS THREAD: runAndWait() finished.")
            except Exception as e:
                print(f"TTS THREAD ERROR: {e}")
            finally:
                self.is_speaking = False
                print("TTS THREAD: Finished.")

        self.current_thread = threading.Thread(target=speak_worker, args=(text,), daemon=True)
        self.current_thread.start()
    
    def stop_speaking(self):
        if self.is_initialized and self.tts_engine._inLoop:
            try:
                self.tts_engine.endLoop()
            except Exception as e:
                print(f"TTS endLoop error: {e}")
        self.is_speaking = False

# --- Conversation Export System ---
class ConversationExporter:
    @staticmethod
    def export_to_text(messages, stats):
        export_text = f"InnerCompass Conversation Export\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nSession Statistics:\n- User Messages: {stats['user_messages']}\n- Bot Responses: {stats['bot_responses']}\n- Duration: {str(datetime.now() - stats['session_start']).split('.')[0]}\n\n{'='*50}\n\n"
        for msg in messages:
            role = "You" if msg["role"] == "user" else "InnerCompass"
            export_text += f"[{role}]:\n{msg['content']}\n\n"
        return export_text

# --- Initialize Components ---
if 'tts_manager' not in st.session_state:
    st.session_state.tts_manager = ImprovedTTSManager()
if 'crisis_detector' not in st.session_state:
    st.session_state.crisis_detector = CrisisDetector()
if 'mood_tracker' not in st.session_state:
    st.session_state.mood_tracker = MoodTracker()

# --- Groq API Configuration ---
try:
    client = Groq(api_key="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")
except Exception as e:
    st.error(f"Failed to initialize Groq client. Please check your API key. Error: {e}")
    st.stop()

GROQ_MODELS = { "llama3-8b-8192": "Llama 3 8B", "llama3-70b-8192": "Llama 3 70B", "mixtral-8x7b-32768": "Mixtral 8x7B", "gemma2-9b-it": "Gemma 2 9B"}

# --- Session State Initialization ---
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state: st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
if "mood_history" not in st.session_state: st.session_state.mood_history = []
if "app_settings" not in st.session_state:
    st.session_state.app_settings = {
        "model": "llama3-8b-8192", "temperature": 0.7, "max_tokens": 1024,
        "tts_enabled": TTS_AVAILABLE and st.session_state.tts_manager.is_initialized,
        "voice_rate": 180, "auto_speak": True,
        "mood_tracking": True, "crisis_detection": True,
        "system_prompt": "You are InnerCompass, a kind, empathetic AI companion. Your goal is to provide emotional support. Listen, validate feelings, and offer gentle, positive advice. Keep responses concise, conversational, and non-judgmental. If crisis language is detected, prioritize safety and suggest professional resources."
    }

# --- Sidebar UI ---
with st.sidebar:
    st.title("🎛️ Settings")
    st.subheader("🤖 AI Model")
    selected_model_key = st.selectbox("Choose Model:", options=list(GROQ_MODELS.keys()), format_func=lambda x: GROQ_MODELS[x], index=list(GROQ_MODELS.keys()).index(st.session_state.app_settings["model"]))
    st.session_state.app_settings["model"] = selected_model_key
    st.session_state.app_settings["temperature"] = st.slider("Creativity (Temperature):", 0.1, 1.5, st.session_state.app_settings["temperature"], 0.1)
    
    st.subheader("🚀 Features")
    st.session_state.app_settings["mood_tracking"] = st.checkbox("Enable Mood Tracking", st.session_state.app_settings["mood_tracking"])
    st.session_state.app_settings["crisis_detection"] = st.checkbox("Enable Crisis Detection", st.session_state.app_settings["crisis_detection"])

    if TTS_AVAILABLE:
        st.subheader("🔊 Text-to-Speech")
        if not st.session_state.tts_manager.is_initialized:
            st.error("TTS Engine failed to load. Please restart the app or check your system's TTS installation.")
        
        st.session_state.app_settings["tts_enabled"] = st.checkbox("Enable Voice Output", st.session_state.app_settings["tts_enabled"])
        if st.session_state.app_settings["tts_enabled"]:
            st.session_state.app_settings["auto_speak"] = st.checkbox("Auto-speak responses", st.session_state.app_settings["auto_speak"])
            st.session_state.app_settings["voice_rate"] = st.slider("Speech Speed:", 100, 300, st.session_state.app_settings["voice_rate"], 10)
            st.session_state.tts_manager.set_voice_settings(rate=st.session_state.app_settings["voice_rate"])
            if st.button("🔇 Stop Voice"): 
                st.session_state.tts_manager.stop_speaking()
                st.toast("🔇 Voice stopped.")
    else:
        st.warning("Install `pyttsx3` for voice output.")


# --- Main Application UI ---
st.title("InnerCompass 🌟")
st.markdown('<p class="subtitle">Your empathetic AI companion for thoughts and feelings.</p>', unsafe_allow_html=True)
crisis_placeholder = st.empty()

# --- Dashboard Columns ---
col1, col2, col3 = st.columns([1.5, 1.5, 1])
with col1:
    stats = st.session_state.conversation_stats
    duration = str(datetime.now() - stats['session_start']).split('.')[0]
    st.markdown(f'<div class="stats-box"><h4>📊 Session Stats</h4><p><strong>Messages:</strong> You: {stats["user_messages"]} | Bot: {stats["bot_responses"]}</p><p><strong>Duration:</strong> {duration}</p></div>', unsafe_allow_html=True)

with col2:
    if st.session_state.app_settings["mood_tracking"] and st.session_state.mood_history:
        last_mood, last_score = st.session_state.mood_history[-1]
        mood_map = {'very_positive': '😃', 'positive': '😊', 'neutral': '😐', 'negative': '😟', 'very_negative': '😞'}
        st.markdown(f'<div class="mood-tracker"><h4>😊 Mood Snapshot</h4><p><strong>Last mood:</strong> {last_mood.replace("_", " ").title()} {mood_map.get(last_mood, "❓")}</p></div>', unsafe_allow_html=True)
        st.progress(last_score / 5.0)
    else:
        st.markdown('<div class="info-box"><h4>💡 Tip</h4><p>Enable mood tracking in the sidebar to see insights about your conversation.</p></div>', unsafe_allow_html=True)

with col3:
    if st.button("✨ New Chat"):
        st.session_state.messages = []
        st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
        st.session_state.mood_history = []
        st.session_state.tts_manager.stop_speaking()
        st.rerun()
    if st.session_state.messages:
        export_text = ConversationExporter.export_to_text(st.session_state.messages, st.session_state.conversation_stats)
        st.download_button("📥 Export Text", export_text, f"InnerCompass_Chat_{datetime.now().strftime('%Y%m%d')}.txt")

# --- Chat Interface ---
if not st.session_state.messages:
    welcome_message = "Hello! I'm InnerCompass. How are you feeling today? I'm here to listen."
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    st.session_state.conversation_stats["bot_responses"] += 1

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.tts_manager.stop_speaking()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_stats["user_messages"] += 1
    
    with st.chat_message("user"): 
        st.markdown(prompt)

    crisis_detected = False
    if st.session_state.app_settings["crisis_detection"]:
        if st.session_state.crisis_detector.detect_crisis(prompt)['crisis']:
            crisis_detected = True
            with crisis_placeholder.container():
                st.markdown('<div class="crisis-alert"><h3>🚨 Immediate Help is Available</h3><p>It sounds like you are in distress. Please reach out to a professional who can support you. Your safety is most important.</p><ul><li><strong>USA/Canada:</strong> Call or text 988</li><li><strong>UK:</strong> Call 111</li><li><strong>Crisis Text Line:</strong> Text HOME to 741741</li></ul></div>', unsafe_allow_html=True)
            safety_response = "I am very concerned by what you've shared. Your safety is the highest priority. Please, reach out to one of the crisis resources listed above. They are available 24/7 to help you."
            
            with st.chat_message("assistant"):
                st.markdown(safety_response)
            st.session_state.messages.append({"role": "assistant", "content": safety_response})
            st.session_state.conversation_stats["bot_responses"] += 1
            if st.session_state.app_settings["tts_enabled"]:
                st.session_state.tts_manager.speak_text(safety_response)


    if not crisis_detected:
        try:
            if st.session_state.app_settings["mood_tracking"]:
                mood, score = st.session_state.mood_tracker.analyze_mood(prompt)
                st.session_state.mood_history.append((mood, score))

            api_messages = [{"role": "system", "content": st.session_state.app_settings["system_prompt"]}] + st.session_state.messages
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("InnerCompass is thinking..."):
                    stream = client.chat.completions.create(
                        model=st.session_state.app_settings["model"],
                        messages=api_messages,
                        temperature=st.session_state.app_settings["temperature"],
                        max_tokens=st.session_state.app_settings["max_tokens"],
                        stream=True,
                    )
                    full_response = ""
                    for chunk in stream:
                        if content := chunk.choices[0].delta.content:
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.conversation_stats["bot_responses"] += 1

            if st.session_state.app_settings["tts_enabled"] and st.session_state.app_settings["auto_speak"]:
                st.toast("🔊 Speaking response...")
                st.session_state.tts_manager.speak_text(full_response)

        except Exception as e:
            st.error(f"An error occurred with the AI response: {e}")
            traceback.print_exc()

    # CRITICAL FIX: Do NOT call st.rerun() here. 
    # Streamlit will automatically rerun the script because we have updated st.session_state.messages
    # Calling it manually will kill the TTS thread before it can finish speaking.
