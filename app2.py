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
    h1 { color: white; } .subtitle { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# --- NEW: Cloud-Based TTS Manager using gTTS ---
class GTTSManager:
    def __init__(self):
        self.is_initialized = True # gTTS doesn't need complex initialization

    def clean_text_for_speech(self, text):
        clean_text = re.sub(r'[\*#`]', '', text)
        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text[:500] # gTTS can be slow with very long text

    def speak_text_in_browser(self, text, lang='en', slow=False):
        if not self.is_initialized or not text.strip():
            return
        try:
            clean_text = self.clean_text_for_speech(text)
            
            # Create the gTTS object
            tts = gTTS(text=clean_text, lang=lang, slow=slow)
            
            # Create an in-memory binary file
            fp = io.BytesIO()
            
            # Write the audio data to the in-memory file
            tts.write_to_fp(fp)
            
            # Rewind the file to the beginning
            fp.seek(0)
            
            # Play the audio in the user's browser using st.audio
            st.audio(fp, format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"gTTS Error: {e}", icon="🔊")


# --- Helper Classes (CrisisDetector, etc. - No changes needed) ---
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
    
    # Initialize the NEW GTTSManager
    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = GTTSManager()

except Exception as e:
    st.error(f"Failed to initialize Groq client. Error: {e}", icon="🚨")
    st.stop()

# --- Session State Initialization ---
if "app_settings" not in st.session_state:
    st.session_state.app_settings = {
        "model": "llama3-8b-8192", "temperature": 0.7, "max_tokens": 1024,
        # Updated TTS settings for gTTS
        "tts_enabled": True,
        "tts_lang": 'en', 
        "tts_slow": False,
        "auto_speak": True,
        "mood_tracking": True, "crisis_detection": True,
        "system_prompt": "You are InnerCompass, a kind AI companion..."
    }
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state: st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
if "mood_history" not in st.session_state: st.session_state.mood_history = []


# --- Sidebar UI (Updated for gTTS) ---
with st.sidebar:
    st.title("🎛️ Settings")
    # AI Model and Features sections remain the same
    st.subheader("🤖 AI Model")
    # ... your model selectbox and temperature slider

    st.subheader("🚀 Features")
    # ... your feature checkboxes

    st.subheader("🔊 Text-to-Speech (Google)")
    st.session_state.app_settings["tts_enabled"] = st.checkbox("Enable Voice Output", value=st.session_state.app_settings["tts_enabled"])
    if st.session_state.app_settings["tts_enabled"]:
        st.session_state.app_settings["auto_speak"] = st.checkbox("Auto-speak responses", value=st.session_state.app_settings["auto_speak"])
        
        # gTTS specific options
        st.session_state.app_settings["tts_lang"] = st.selectbox(
            "Language:",
            options=['en', 'fr', 'es', 'de', 'it', 'ja', 'ko'],
            help="Language for the voice output."
        )
        st.session_state.app_settings["tts_slow"] = st.checkbox(
            "Slow Speed", 
            value=st.session_state.app_settings["tts_slow"]
        )

# --- Main App Logic ---
st.title("InnerCompass 🌟")
st.markdown('<p class="subtitle">Your empathetic AI companion for thoughts and feelings.</p>', unsafe_allow_html=True)
# ... Your dashboard columns and buttons remain the same

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Crisis detection logic here ---
    
    try:
        # --- Groq API Call ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("InnerCompass is thinking..."):
                stream = groq_client.chat.completions.create(
                    model=st.session_state.app_settings["model"],
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
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
        
        # --- FINAL STEP: Call the GTTS Manager ---
        if st.session_state.app_settings["tts_enabled"] and st.session_state.app_settings["auto_speak"]:
            st.toast("Generating voice output...", icon="💬")
            st.session_state.tts_manager.speak_text_in_browser(
                full_response,
                lang=st.session_state.app_settings["tts_lang"],
                slow=st.session_state.app_settings["tts_slow"]
            )
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🔥")
        traceback.print_exc()
