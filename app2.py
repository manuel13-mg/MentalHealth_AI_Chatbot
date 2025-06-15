import streamlit as st
from groq import Groq
from gtts import gTTS
import io
import toml             # Import the toml library
import traceback
from datetime import datetime
import warnings
import re
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="InnerCompass 🌟", page_icon="🌟", layout="wide")

# --- CSS Styling ---
st.markdown("""<style> /* Your CSS Here */ </style>""", unsafe_allow_html=True)


# --- NEW: Load Configuration from TOML File ---
@st.cache_data # Cache the config to avoid reloading it on every interaction
def load_config(filepath="config.toml"):
    """Loads the application configuration from a .toml file."""
    try:
        with open(filepath, "r") as f:
            return toml.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found at {filepath}. Please create it.")
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

# --- TTS Manager and Helper Classes (No changes needed in these) ---
class GTTSManager: # Your GTTSManager class here...
    def __init__(self): self.is_initialized = True
    def clean_text_for_speech(self, text):
        clean_text = re.sub(r'[\*#`]', '', text).strip()
        if len(clean_text) > 3000:
            st.warning("Response truncated for audio.", icon="⚠️")
            return clean_text[:3000]
        return clean_text
    def speak_text_in_browser(self, text, lang='en', slow=False):
        if not text.strip(): return
        st.toast("Generating voice...", icon="💬")
        try:
            tts = gTTS(text=self.clean_text_for_speech(text), lang=lang, slow=slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.error(f"gTTS Error: {e}", icon="🔊")

class CrisisDetector: # Your class here...
    def __init__(self): self.crisis_keywords = ['suicide', 'kill myself']
    def detect_crisis(self, text): return {'crisis': any(kw in text.lower() for kw in self.crisis_keywords)}
class MoodTracker: # Your class here...
    def analyze_mood(self, text): return 'neutral', 3
class ConversationExporter: # Your class here...
    @staticmethod
    def export_to_text(messages, stats): return "Chat History..."


# --- Initialize Clients and Load Config ---
config = load_config()
if not config:
    st.stop() # Stop the app if config fails to load

try:
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Groq API key missing in .streamlit/secrets.toml.", icon="🚨")
        st.stop()
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = GTTSManager()
except Exception as e:
    st.error(f"Failed to initialize API clients. Error: {e}", icon="🚨")
    st.stop()


# --- Session State Initialization from Config File ---
GROQ_MODELS = { "llama3-8b-8192": "Llama 3 8B", "llama3-70b-8192": "Llama 3 70B", "mixtral-8x7b-32768": "Mixtral 8x7B", "gemma2-9b-it": "Gemma 2 9B"}
if "app_settings" not in st.session_state:
    # Load all settings from the config dictionary
    st.session_state.app_settings = {
        "model": config["ai"]["model"],
        "temperature": config["ai"]["temperature"],
        "max_tokens": config["ai"]["max_tokens"],
        "system_prompt": config["ai"]["system_prompt"],
        "mood_tracking": config["features"]["mood_tracking"],
        "crisis_detection": config["features"]["crisis_detection"],
        "tts_enabled": config["tts"]["enabled"],
        "tts_lang": config["tts"]["lang"],
        "tts_slow": config["tts"]["slow"],
        "auto_speak": config["tts"]["auto_speak"]
    }
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state: st.session_state.conversation_stats = {"user_messages": 0, "bot_responses": 0, "session_start": datetime.now()}
if "mood_history" not in st.session_state: st.session_state.mood_history = []


# --- Sidebar UI (Reads from session state as before) ---
with st.sidebar:
    st.title("🎛️ Settings")
    st.subheader("🤖 AI Model")
    # ... The rest of your sidebar code is unchanged as it correctly uses st.session_state ...
    selected_model_key = st.selectbox("Choose Model:", options=list(GROQ_MODELS.keys()), index=list(GROQ_MODELS.keys()).index(st.session_state.app_settings["model"]))
    st.session_state.app_settings["model"] = selected_model_key
    st.session_state.app_settings["temperature"] = st.slider("Creativity:", 0.1, 1.5, st.session_state.app_settings["temperature"], 0.1)

    st.subheader("🔊 Text-to-Speech (Google)")
    st.session_state.app_settings["tts_enabled"] = st.checkbox("Enable Voice", value=st.session_state.app_settings["tts_enabled"])
    # ... etc.

# --- Main App Logic (Unchanged, but now benefits from the loaded config) ---
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
        # Construct the message list including the system prompt from session state
        api_messages = [
            {"role": "system", "content": st.session_state.app_settings["system_prompt"]},
        ]
        api_messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages])

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("InnerCompass is thinking..."):
                stream = groq_client.chat.completions.create(
                    model=st.session_state.app_settings["model"],
                    messages=api_messages,
                    temperature=st.session_state.app_settings["temperature"],
                    max_tokens=st.session_state.app_settings["max_tokens"],
                    stream=True,
                )
                full_response = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
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
