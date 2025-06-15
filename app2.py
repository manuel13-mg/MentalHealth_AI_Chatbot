import streamlit as st
from groq import Groq
from gtts import gTTS   # Import gTTS
import io               # Import io for in-memory file handling
import traceback
import time
import json
from datetime import datetime
import warnings
import re
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="InnerCompass 🌟", page_icon="🌟", layout="wide")

# --- CSS Styling (Your existing CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); min-height: 100vh; }
    /* Paste the rest of your beautiful CSS here */
    h1 { color: white; text-align: center; } .subtitle { color: #94a3b8; text-align: center; }
</style>
""", unsafe_allow_html=True)


# --- Cloud-Based TTS Manager using gTTS ---
class GTTSManager:
    def __init__(self):
        self.is_initialized = True

    def clean_text_for_speech(self, text):
        clean_text = re.sub(r'[\*#`]', '', text)
        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # This 3000-char limit acts as a final safety net.
        # It aligns with the ~500 word prompt limit.
        if len(clean_text) > 3000:
            st.warning("Response is very long and has been truncated for audio output.", icon="⚠️")
            return clean_text[:3000]
        
        return clean_text

    def speak_text_in_browser(self, text, lang='en', slow=False):
        if not self.is_initialized or not text.strip():
            return
        
        st.toast("Generating voice output...", icon="💬")
        
        try:
            clean_text = self.clean_text_for_speech(text)
            tts = gTTS(text=clean_text, lang=lang, slow=slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.error(f"gTTS Error: {e}", icon="🔊")


# --- Helper Classes (CrisisDetector, etc.) ---
class CrisisDetector:
    def __init__(self): self.crisis_keywords = ['suicide', 'kill myself', 'hurt myself']
    def detect_crisis(self, text): return {'crisis': any(kw in text.lower() for kw in self.crisis_keywords)}

class MoodTracker:
    def analyze_mood(self, text): return 'neutral', 3

class ConversationExporter:
    @staticmethod
    def export_to_text(messages, stats): return "Chat History..."


# --- Initialize API Clients and Managers ---
try:
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Groq API key missing in .streamlit/secrets.toml.", icon="🚨")
        st.stop()
    
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = GTTSManager()
except Exception as e:
    st.error(f"Failed to initialize Groq client. Error: {e}", icon="🚨")
    st.stop()

# --- Session State Initialization ---
GROQ_MODELS = { "llama3-8b-8192": "Llama 3 8B", "llama3-70b-8192": "Llama 3 70B", "mixtral-8x7b-32768": "Mixtral 8x7B", "gemma2-9b-it": "Gemma 2 9B"}
if "app_settings" not in st.session_state:
    st.session_state.app_settings = {
        "model": "llama3-8b-8192", 
        "temperature": 0.7, 
        "max_tokens": 1024, # max_tokens is a fallback, the prompt is the primary control
        "tts_enabled": True, 
        "tts_lang": 'en', 
        "tts_slow": False,
        "auto_speak": True, 
        "mood_tracking": True, 
        "crisis_detection": True,
        # --- THE ONLY CHANGE IS HERE ---
        "system_prompt": """You are InnerCompass, a kind, empathetic AI companion. Your goal is to provide emotional support. Listen, validate feelings, and offer gentle, positive advice. Keep responses conversational, and non-judgmental. If crisis language is detected, prioritize safety and suggest professional resources.

**Crucially, keep your responses concise and friendly. Your response must be under a 500-word limit to ensure it is easy for the user to read and listen to.**
"""
    }
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state: st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
if "mood_history" not in st.session_state: st.session_state.mood_history = []


# --- Sidebar UI ---
with st.sidebar:
    st.title("🎛️ Settings")
    st.subheader("🤖 AI Model")
    selected_model_key = st.selectbox("Choose Model:", options=list(GROQ_MODELS.keys()), format_func=lambda x: GROQ_MODELS[x], index=list(GROQ_MODELS.keys()).index(st.session_state.app_settings["model"]))
    st.session_state.app_settings["model"] = selected_model_key
    st.session_state.app_settings["temperature"] = st.slider("Creativity (Temperature):", 0.1, 1.5, st.session_state.app_settings["temperature"], 0.1)

    st.subheader("🚀 Features")
    st.session_state.app_settings["mood_tracking"] = st.checkbox("Enable Mood Tracking", value=st.session_state.app_settings["mood_tracking"])
    st.session_state.app_settings["crisis_detection"] = st.checkbox("Enable Crisis Detection", value=st.session_state.app_settings["crisis_detection"])

    st.subheader("🔊 Text-to-Speech (Google)")
    st.session_state.app_settings["tts_enabled"] = st.checkbox("Enable Voice Output", value=st.session_state.app_settings["tts_enabled"])
    if st.session_state.app_settings["tts_enabled"]:
        st.session_state.app_settings["auto_speak"] = st.checkbox("Auto-speak responses", value=st.session_state.app_settings["auto_speak"])
        st.session_state.app_settings["tts_lang"] = st.selectbox(
            "Language:", options=['en', 'en-uk', 'en-us', 'en-au', 'fr', 'es', 'de'], help="Language for the voice output."
        )
        st.session_state.app_settings["tts_slow"] = st.checkbox(
            "Slow Speed", value=st.session_state.app_settings["tts_slow"]
        )

# --- Main App Logic ---
st.title("InnerCompass 🌟")
st.markdown('<p class="subtitle">Your empathetic AI companion for thoughts and feelings.</p>', unsafe_allow_html=True)

# ... Your dashboard columns and buttons ...

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # We need to construct the message list for the API call
        # including the all-important system prompt.
        api_messages = [
            {"role": "system", "content": st.session_state.app_settings["system_prompt"]},
        ]
        # Add the rest of the conversation history
        for msg in st.session_state.messages:
             api_messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("InnerCompass is thinking..."):
                stream = groq_client.chat.completions.create(
                    model=st.session_state.app_settings["model"],
                    messages=api_messages, # Pass the full list with the system prompt
                    temperature=st.session_state.app_settings["temperature"],
                    max_tokens=st.session_state.app_settings["max_tokens"],
                    stream=True,
                )
                full_response = "".join(chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content)
                message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        if st.session_state.app_settings["tts_enabled"] and st.session_state.app_settings["auto_speak"] and full_response:
            st.session_state.tts_manager.speak_text_in_browser(
                full_response,
                lang=st.session_state.app_settings["tts_lang"],
                slow=st.session_state.app_settings["tts_slow"]
            )
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🔥")
        traceback.print_exc()
