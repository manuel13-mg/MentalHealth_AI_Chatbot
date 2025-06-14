import streamlit as st
import groq
import re
import time
import json
from datetime import datetime, timedelta
from transformers import pipeline
import os
from typing import Dict, List, Tuple, Optional

# --- Configuration ---
st.set_page_config(
    page_title="MindfulChat Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enhanced CSS Styling - Black & Blue Theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1e40af 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    .stMainBlockContainer {padding-top: 1rem;}
    
    /* Main container styling */
    .main > div {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        padding: 20px;
        margin: 20px;
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

    /* Resource box styling */
    .resource-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: #86efac;
    }

    /* Mood tracker styling */
    .mood-tracker {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
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
        padding: 14px 28px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
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

    .stSidebar > div {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        margin: 10px !important;
        padding: 20px !important;
    }

    /* Text colors */
    .stMarkdown, .stText, p, div {
        color: #e2e8f0 !important;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* Exercise box styling */
    .exercise-box {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: #c4b5fd;
    }

    /* Typing indicator specific styles */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94a3b8;
        font-style: italic;
    }

    .typing-dots {
        display: flex;
        gap: 4px;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #60a5fa;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }

    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    .typing-dot:nth-child(3) { animation-delay: 0s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }

    /* Cursor for typewriter effect */
    .cursor {
        animation: blink-caret 0.75s step-end infinite;
    }

    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: white; } /* Adjust cursor color if needed */
    }

    .typewriter {
        white-space: pre-wrap; /* Preserves whitespace and wraps text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Enhanced Models and Initialization ---
@st.cache_resource
def load_sentiment_model():
    try:
        # Using a smaller, faster model if available, or fallback
        sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        return sentiment_pipeline, True
    except Exception as e:
        st.warning(f"Sentiment analysis model loading failed: {e}. Falling back to basic sentiment detection.")
        # Fallback to a simpler, less resource-intensive method or disable
        return None, False

@st.cache_resource
def initialize_groq_client():
    try:
        # Load API key from Streamlit secrets or environment variable
        # For deployment, st.secrets["GROQ_API_KEY"] is recommended
        groq_api_key = os.getenv("GROQ_API_KEY", "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")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
            return None, False
        client = groq.Groq(api_key=groq_api_key)
        return client, True
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}. Please check your API key.")
        return None, False

sentiment_pipeline, sentiment_analysis_available = load_sentiment_model()
client, model_available = initialize_groq_client()

# --- Crisis Detection System ---
def detect_crisis_keywords(text: str) -> Tuple[bool, str]:
    """Detect crisis-related content in user input."""
    crisis_patterns = {
        'suicide': [
            r'\b(suicide|suicidal|kill myself|end my life|want to die|not worth living|better off dead|no point living|take my own life)\b'
        ],
        'self_harm': [
            r'\b(self harm|hurt myself|cut myself|harm myself|cutting|burning myself|self injury)\b'
        ],
        'emergency': [
            r'\b(emergency|crisis|help me|desperate|can\'t go on|nobody cares|completely alone|no way out)\b'
        ]
    }
    
    text_lower = text.lower()
    for category, patterns in crisis_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, category
    
    return False, None

# --- New: Generate Crisis Resources via Groq ---
def generate_crisis_resources_groq() -> str:
    if not model_available:
        return """🚨 **IMMEDIATE HELP AVAILABLE** 🚨\n\nIf you're in immediate danger or having thoughts of suicide, please reach out now:\n\n**🇺🇸 United States:**\n• National Suicide Prevention Lifeline: **988**\n• Crisis Text Line: Text **HOME** to **741741**\n• Emergency Services: **911**\n\n**🇬🇧 United Kingdom:**\n• Samaritans: **116 123** (free, 24/7)\n• Crisis Text Line: Text **SHOUT** to **85258**\n\n**🇨🇦 Canada:**\n• Talk Suicide Canada: **1-833-456-4566**\n• Crisis Text Line: Text **TALK** to **686868**\n\n**🌍 International:**\n• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n**Remember: You are not alone. These feelings can change. Help is available.**"""

    try:
        prompt = """
        You are an empathetic AI. A user has expressed a strong crisis or suicidal ideation. Provide crucial, immediate, and direct crisis helpline resources for the United States, United Kingdom, and Canada. Include general international resources if possible. Frame it urgently and empathetically. Do NOT add any conversational pleasantries or therapy advice. Just the direct resources and a concluding statement emphasizing hope and help.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=300,
            temperature=0.2, # Low temperature for direct information
            top_p=0.9,
        )
        response_content = chat_completion.choices[0].message.content.strip()
        return f"🚨 **IMMEDIATE HELP AVAILABLE** 🚨\n\n{response_content}"
    except Exception as e:
        print(f"Error generating crisis resources: {e}")
        return """🚨 **IMMEDIATE HELP AVAILABLE** 🚨\n\nIf you're in immediate danger or having thoughts of suicide, please reach out now:\n\n**🇺🇸 United States:**\n• National Suicide Prevention Lifeline: **988**\n• Crisis Text Line: Text **HOME** to **741741**\n• Emergency Services: **911**\n\n**🇬🇧 United Kingdom:**\n• Samaritans: **116 123** (free, 24/7)\n• Crisis Text Line: Text **SHOUT** to **85258**\n\n**🇨🇦 Canada:**\n• Talk Suicide Canada: **1-833-456-4566**\n• Crisis Text Line: Text **TALK** to **686868**\n\n**🌍 International:**\n• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n**Remember: You are not alone. These feelings can change. Help is available.**"""


# --- Enhanced Sentiment Analysis ---
def analyze_sentiment_advanced(text: str) -> Dict:
    """Advanced sentiment analysis with emotion detection."""
    if not sentiment_analysis_available or not sentiment_pipeline:
        # Fallback to a simple keyword-based approach if model not available
        text_lower = text.lower()
        if any(word in text_lower for word in ['sad', 'depressed', 'anxious', 'lonely', 'stressed', 'unhappy']):
            sentiment = 'negative'
        elif any(word in text_lower for word in ['happy', 'joyful', 'excited', 'optimistic', 'great', 'good']):
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        return {'sentiment': sentiment, 'confidence': 0.5, 'intensity': 'moderate'}
        
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # RoBERTa-based models typically use NEGATIVE, NEUTRAL, POSITIVE
        sentiment_map = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
        sentiment = sentiment_map.get(label, 'neutral') # Default to neutral if label not found
        
        # Determine emotional intensity based on confidence for non-neutral sentiment
        intensity = 'moderate'
        if sentiment != 'neutral':
            if score > 0.85:
                intensity = 'high'
            elif score < 0.6:
                intensity = 'low'
        
        return {
            'sentiment': sentiment,
            'confidence': score,
            'intensity': intensity
        }
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.5, 'intensity': 'moderate'}


# --- Therapeutic Techniques (Generated by Groq) ---
class TherapeuticTechniques:
    def __init__(self, groq_client):
        self.client = groq_client

    def _generate_technique(self, prompt_text: str, model_temp: float = 0.7, max_tokens: int = 250) -> str:
        if not model_available:
            return "I'm sorry, I can't generate that right now due to a technical issue."
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model="llama3-8b-8192",
                max_tokens=max_tokens,
                temperature=model_temp,
                top_p=0.9,
                stream=False
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating technique: {e}")
            return "I'm sorry, I'm having trouble generating that exercise right now. Please try again."

    def breathing_exercise(self) -> str:
        prompt = """
        Generate instructions for a simple, effective breathing exercise (e.g., box breathing, 4-7-8 breathing). Format it clearly with steps and use emojis. Keep it concise, about 4-6 sentences.
        """
        return self._generate_technique(prompt, model_temp=0.6, max_tokens=150)

    def grounding_exercise(self) -> str:
        prompt = """
        Generate instructions for the 5-4-3-2-1 grounding technique. Explain what it is and list the steps clearly. Use emojis and keep it concise, about 5-7 sentences.
        """
        return self._generate_technique(prompt, model_temp=0.6, max_tokens=180)

    def positive_affirmation(self) -> str:
        prompt = """
        Generate one short, impactful positive affirmation (1-2 sentences). It should be encouraging and focus on self-worth or resilience. Do not include any additional conversational text, just the affirmation.
        """
        return self._generate_technique(prompt, model_temp=0.9, max_tokens=50)

    def coping_strategies(self, sentiment: str) -> str:
        strategy_prompt = ""
        if sentiment == 'negative':
            strategy_prompt = """
            Generate 5-7 actionable and simple coping strategies for someone experiencing negative emotions (e.g., stress, sadness, anxiety). Frame them as bullet points or a list, using an encouraging tone.
            """
        elif sentiment == 'positive':
            strategy_prompt = """
            Generate 5-7 actionable and simple strategies for maintaining and enhancing positive well-being. Frame them as bullet points or a list, using an encouraging tone.
            """
        else: # Neutral
            strategy_prompt = """
            Generate 5-7 simple self-care ideas for someone in a neutral emotional state, encouraging gentle exploration and well-being. Frame them as bullet points or a list.
            """
        return self._generate_technique(strategy_prompt, model_temp=0.8, max_tokens=200)

# --- Dynamic Typing Effects ---
def display_typing_indicator():
    """Display typing indicator with animated dots."""
    typing_placeholder = st.empty()
    typing_html = """
    <div class="typing-indicator">
        <span>MindfulChat is thinking</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """
    typing_placeholder.markdown(typing_html, unsafe_allow_html=True)
    return typing_placeholder

def stream_response_with_typing(response_text: str, container, speed_multiplier: float = 1.0):
    """Stream response with realistic typing patterns."""
    if not response_text:
        return
    
    # Split into sentences for more natural pauses
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    
    text_placeholder = container.empty()
    full_text = ""
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        if i > 0:
            full_text += " "
        
        for char in sentence:
            full_text += char
            
            text_placeholder.markdown(
                f'<div class="typewriter">{full_text}<span class="cursor">|</span></div>', 
                unsafe_allow_html=True
            )
            
            base_speed = 0.025 * speed_multiplier
            if char in '.!?':
                time.sleep(base_speed * 8)
            elif char in ',;:':
                time.sleep(base_speed * 4)
            elif char == ' ':
                time.sleep(base_speed * 2)
            elif char in '()"\'':
                time.sleep(base_speed)
            else:
                time.sleep(base_speed)
        
        if i < len(sentences) - 1:
            time.sleep(0.3 * speed_multiplier)
    
    time.sleep(0.5 * speed_multiplier)
    text_placeholder.markdown(f'<div class="typewriter">{full_text}</div>', unsafe_allow_html=True)


# --- Enhanced Response Generation ---
def get_therapeutic_response(user_input: str, sentiment_data: Dict, chat_history: str, crisis_detected: bool = False) -> str:
    """Generate therapeutic response with crisis handling."""
    if not model_available or not client:
        return "I'm experiencing technical difficulties and cannot provide a full response right now. Please try again later or seek professional help if this is urgent."

    # Handle crisis situations first (this part remains largely static for reliability)
    if crisis_detected:
        crisis_intro = """I'm really concerned about what you're sharing. It sounds like you're going through an incredibly difficult time, and I want you to know that your feelings are valid, but you don't have to face this alone. There is help available, and things can get better."""
        # The crisis resources themselves will be generated by Groq separately in the main() function
        crisis_affirmation = """\n\nRemember: You are strong, and these feelings can change. Reaching out for help is a brave step."""
        return crisis_intro + crisis_affirmation

    try:
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        intensity = sentiment_data['intensity']

        # Determine the therapeutic approach based on sentiment
        if sentiment == 'negative':
            persona_adjustment = "You are interacting with someone experiencing negative emotions (e.g., sadness, stress, anxiety, overwhelm). Your primary goal is to provide deep empathy, validate their feelings, normalize their experience, and offer specific, actionable coping strategies or a comforting perspective. Ask a gentle follow-up question. Conclude with a relevant, hopeful quote or affirmation."
        elif sentiment == 'positive':
            persona_adjustment = "You are interacting with someone experiencing positive emotions. Your primary goal is to celebrate with them, reinforce their positive experiences, encourage gratitude, and help them sustain their well-being. Ask an open-ended question about what's going well or how they achieved this. Conclude with an uplifting quote or affirmation."
        else: # neutral
            persona_adjustment = "You are interacting with someone in a neutral emotional state. Your primary goal is to gently explore their situation, offer broad self-care ideas, and encourage self-reflection or mild exploration of their feelings. Ask an open-ended question to encourage deeper sharing. Conclude with a thoughtful quote or affirmation."

        prompt = f"""
        **Role**: MindfulChat Pro, an empathetic, non-judgmental, and wise AI companion specializing in mental wellness and evidence-based therapeutic techniques (CBT, mindfulness, solution-focused, positive psychology).

        **Objective**: Provide compassionate support, validation, practical advice, and encouragement. Your responses should be personalized, insightful, and actionable.

        **Current User State**:
        - User Input: "{user_input}"
        - Detected Emotional Sentiment: {sentiment} (Confidence: {confidence:.2f}, Intensity: {intensity})
        - Recent Conversation Context: {chat_history[-400:] if chat_history else 'No significant prior conversation.'}

        **Your Task**:
        {persona_adjustment}

        **Constraints**:
        - Keep the response concise, typically 3-5 sentences.
        - Ensure a natural conversational flow.
        - Avoid technical jargon.
        - The response MUST end with a short, relevant inspirational quote or affirmation. Separate the quote clearly.
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # Using Llama 3 8B for general responses
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )

        response = chat_completion.choices[0].message.content.strip()
        
        # Ensure the response ends with a quote, if not, add a generic one
        if not re.search(r'["“”].*["“”]$', response) and not re.search(r'\*.*\*\s*$', response):
             response += "\n\n*You've got this, one step at a time.*" # Fallback quote if Groq doesn't end with one

        return response

    except Exception as e:
        print(f"Error generating response from Groq: {e}")
        return "I'm having some technical difficulties right now. In the meantime, remember that whatever you're going through, you're not alone. Please try again in a moment, or reach out to someone you trust."

# --- Mood Tracking ---
def initialize_mood_tracking():
    """Initialize mood tracking in session state."""
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = []
    # No daily_checkin needed if it's integrated differently or removed

def log_mood(sentiment_data: Dict):
    """Log mood data for tracking."""
    mood_entry = {
        'timestamp': datetime.now().isoformat(),
        'sentiment': sentiment_data['sentiment'],
        'confidence': sentiment_data['confidence'],
        'intensity': sentiment_data['intensity']
    }
    st.session_state.mood_history.append(mood_entry)
    
    # Keep only last 30 days
    cutoff_date = datetime.now() - timedelta(days=30)
    st.session_state.mood_history = [
        entry for entry in st.session_state.mood_history
        if datetime.fromisoformat(entry['timestamp']) > cutoff_date
    ]

def get_mood_insights() -> str:
    """Provide insights based on mood history."""
    if not st.session_state.mood_history:
        return "Start chatting to track your mood patterns over time."
    
    recent_moods = st.session_state.mood_history[-7:]  # Last 7 entries
    if not recent_moods: # Handle case where mood_history is cleared or very short
        return "Not enough recent data for insights. Keep chatting!"

    positive_count = sum(1 for mood in recent_moods if mood['sentiment'] == 'positive')
    negative_count = sum(1 for mood in recent_moods if mood['sentiment'] == 'negative')
    
    total_recent = len(recent_moods)

    if positive_count > negative_count and positive_count > total_recent / 2:
        return f"📈 You've been experiencing more positive emotions lately ({positive_count}/{total_recent} recent interactions). Keep nurturing what's working!"
    elif negative_count > positive_count and negative_count > total_recent / 2:
        return f"📊 You've been navigating some challenges recently ({negative_count}/{total_recent} recent interactions). Remember, it's okay to have difficult days, and reaching out is a sign of strength."
    else:
        return f"⚖️ Your emotions have been quite balanced lately ({positive_count} positive, {negative_count} negative out of {total_recent} recent interactions). This shows emotional adaptability!"


# --- Main Application ---
def main():
    # Header
    st.markdown("""
    # MindfulChat Pro
    
    <div class="subtitle">Your advanced AI companion for mental wellness and emotional support</div>
    """, unsafe_allow_html=True)

    # Initialize systems
    initialize_mood_tracking()
    therapeutic_methods = TherapeuticTechniques(client) # Initialize with Groq client

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Generate initial welcome message via Groq
        if model_available:
            try:
                welcome_prompt = """
                You are MindfulChat Pro, an empathetic AI companion for mental wellness. Craft a warm welcome message that introduces your purpose (compassionate support, evidence-based techniques), outlines how you can help (emotional support, coping strategies, exercises, crisis resources, mood tracking, safe space), and ends with an encouraging quote from a well-known figure. Structure it as a friendly introduction.
                """
                welcome_msg_groq = client.chat.completions.create(
                    messages=[{"role": "user", "content": welcome_prompt}],
                    model="llama3-8b-8192",
                    max_tokens=300,
                    temperature=0.8,
                    top_p=0.9,
                ).choices[0].message.content.strip()
                st.session_state.messages.append({"role": "assistant", "content": welcome_msg_groq})
            except Exception as e:
                print(f"Error generating welcome message: {e}")
                fallback_welcome_msg = """Welcome to MindfulChat Pro! I'm here to provide compassionate support and evidence-based therapeutic techniques to help you navigate life's challenges.
Whether you're dealing with stress, anxiety, relationship issues, or just need someone to talk to, I'm here to listen without judgment and offer practical guidance.
**How I can help:**
• Emotional support and validation
• Coping strategies and techniques
• Mindfulness and breathing exercises
• Crisis resources when needed
• Mood tracking and insights
*"You are braver than you believe, stronger than you seem, and smarter than you think." - A.A. Milne*"""
                st.session_state.messages.append({"role": "assistant", "content": fallback_welcome_msg})
        else:
            fallback_welcome_msg = """Welcome to MindfulChat Pro! I'm here to provide compassionate support and evidence-based therapeutic techniques to help you navigate life's challenges.
Whether you're dealing with stress, anxiety, relationship issues, or just need someone to talk to, I'm here to listen without judgment and offer practical guidance.
**How I can help:**
• Emotional support and validation
• Coping strategies and techniques
• Mindfulness and breathing exercises
• Crisis resources when needed
• Mood tracking and insights
*"You are braver than you believe, stronger than you seem, and smarter than you think." - A.A. Milne*"""
            st.session_state.messages.append({"role": "assistant", "content": fallback_welcome_msg})

    # Display mood insights in sidebar
    with st.sidebar:
        st.markdown("### 📊 Mood Insights")
        mood_insight = get_mood_insights()
        st.markdown(f'<div class="mood-tracker">{mood_insight}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🧘‍♀️ Quick Exercises")
        
        if st.button("🌬️ Breathing Exercise"):
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                exercise_content = therapeutic_methods.breathing_exercise() # Groq generated
                typing_placeholder.empty()
                
                exercise_container = st.empty()
                stream_response_with_typing(exercise_content, exercise_container)
                
            st.session_state.messages.append({"role": "assistant", "content": exercise_content})
            st.rerun() # Rerun to update chat display immediately
        
        if st.button("🌱 Grounding Exercise"):
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                exercise_content = therapeutic_methods.grounding_exercise() # Groq generated
                typing_placeholder.empty()
                
                exercise_container = st.empty()
                stream_response_with_typing(exercise_content, exercise_container)
                
            st.session_state.messages.append({"role": "assistant", "content": exercise_content})
            st.rerun()
        
        if st.button("💭 Daily Affirmation"):
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                affirmation = therapeutic_methods.positive_affirmation() # Groq generated
                typing_placeholder.empty()
                
                affirmation_msg = f"🌟 **Today's Affirmation:**\n\n*{affirmation}*\n\nTake a moment to really let this sink in. You deserve these kind words."
                affirmation_container = st.empty()
                stream_response_with_typing(affirmation_msg, affirmation_container)
                
            st.session_state.messages.append({"role": "assistant", "content": affirmation_msg})
            st.rerun()

        st.markdown("---")
        
        st.markdown("""
        ### About MindfulChat Pro
        
        **Enhanced Features:**
        - 🧠 Advanced sentiment analysis
        - ❤️ Evidence-based therapeutic responses
        - 🔍 Crisis detection and resources
        - 📊 Mood tracking and insights
        - 🧘‍♀️ Guided exercises and techniques
        - 💬 Personalized coping strategies
        - 🔒 Safe, judgment-free space
        
        **Therapeutic Approaches:**
        - Cognitive Behavioral Therapy (CBT)
        - Mindfulness-Based Stress Reduction
        - Solution-Focused Brief Therapy
        - Positive Psychology techniques
        
        **Important:** This AI provides support but is not a replacement for professional mental health care. In crisis situations, please contact emergency services or a mental health professional.
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Share what's on your mind... I'm here to listen and support you."):
        # Detect crisis
        crisis_detected, crisis_type = detect_crisis_keywords(prompt)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Analyze sentiment
        sentiment_data = analyze_sentiment_advanced(prompt)
        log_mood(sentiment_data)

        # Generate chat history (plain text for the model)
        # We only send plain text content to the model, removing any HTML wrappers
        chat_history_for_model = "\n".join([
            f"{msg['role']}: {re.sub('<[^<]+?>', '', msg['content'])}" # Strip HTML tags
            for msg in st.session_state.messages[-5:-1] # Last 4 messages before current prompt
        ])

        # Generate response with dynamic typing
        with st.chat_message("assistant"):
            typing_placeholder = display_typing_indicator()
            
            if crisis_detected:
                typing_placeholder.empty()
                crisis_resources_content = generate_crisis_resources_groq() # Groq generated crisis resources
                st.markdown(f'<div class="crisis-alert">{crisis_resources_content}</div>', unsafe_allow_html=True)
                time.sleep(1) # Brief pause before main response
                typing_placeholder = display_typing_indicator() # Show thinking again

            # Generate the main therapeutic response
            response_content = get_therapeutic_response(prompt, sentiment_data, chat_history_for_model, crisis_detected)
            
            typing_placeholder.empty() # Clear typing indicator
            
            response_container = st.empty()
            stream_response_with_typing(response_content, response_container)
            
            # Add coping strategies with typing effect if sentiment is negative
            if sentiment_data['sentiment'] == 'negative' and sentiment_data['intensity'] in ['moderate', 'high']:
                time.sleep(0.5) # Brief pause
                typing_placeholder_2 = display_typing_indicator()
                coping_strategies_content = therapeutic_methods.coping_strategies(sentiment_data['sentiment']) # Groq generated coping strategies
                typing_placeholder_2.empty()
                
                coping_container = st.empty()
                coping_container.markdown(f'<div class="resource-box">{coping_strategies_content}</div>', unsafe_allow_html=True)
                time.sleep(0.3)

        # Add the full bot response (main response + any crisis/coping boxes) to history
        # For simplicity, we capture the final displayed content here for history, or you could reconstruct
        final_bot_display_content = response_content
        if crisis_detected:
            final_bot_display_content = f'{crisis_resources_content}\n\n{final_bot_display_content}'
        if sentiment_data['sentiment'] == 'negative' and sentiment_data['intensity'] in ['moderate', 'high']:
            final_bot_display_content = f'{final_bot_display_content}\n\n<div class="resource-box">{coping_strategies_content}</div>' # Add HTML directly to history if it's shown this way

        st.session_state.messages.append({"role": "assistant", "content": final_bot_display_content})

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 14px; padding: 20px;">
        🤝 Built with care using advanced AI • 🔒 Your conversations are private • 💙 Remember, seeking help is a sign of strength
        <br><br>
        <strong>Emergency Resources:</strong> US: 988 | UK: 116 123 | CA: 1-833-456-4566
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
