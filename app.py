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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Enhanced Models and Initialization ---
@st.cache_resource
def load_sentiment_model():
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        return sentiment_pipeline, True
    except Exception as e:
        st.error(f"Error initializing sentiment analysis: {e}")
        return None, False

@st.cache_resource
def initialize_groq_client():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "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")
        client = groq.Groq(api_key=groq_api_key)
        return client, True
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None, False

sentiment_pipeline, sentiment_analysis_available = load_sentiment_model()
client, model_available = initialize_groq_client()

# --- Crisis Detection System ---
def detect_crisis_keywords(text: str) -> Tuple[bool, str]:
    """Detect crisis-related content in user input."""
    crisis_patterns = {
        'suicide': [
            r'\b(suicide|suicidal|kill myself|end my life|want to die|not worth living)\b',
            r'\b(end it all|take my own life|better off dead|no point living)\b'
        ],
        'self_harm': [
            r'\b(self harm|hurt myself|cut myself|harm myself)\b',
            r'\b(cutting|burning myself|self injury)\b'
        ],
        'emergency': [
            r'\b(emergency|crisis|help me|desperate|can\'t go on)\b',
            r'\b(nobody cares|completely alone|no way out)\b'
        ]
    }
    
    text_lower = text.lower()
    for category, patterns in crisis_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, category
    
    return False, None

def provide_crisis_resources() -> str:
    """Provide immediate crisis resources."""
    return """
🚨 **IMMEDIATE HELP AVAILABLE** 🚨

If you're in immediate danger or having thoughts of suicide, please reach out now:

**🇺🇸 United States:**
• National Suicide Prevention Lifeline: **988**
• Crisis Text Line: Text **HOME** to **741741**
• Emergency Services: **911**

**🇬🇧 United Kingdom:**
• Samaritans: **116 123** (free, 24/7)
• Crisis Text Line: Text **SHOUT** to **85258**

**🇨🇦 Canada:**
• Talk Suicide Canada: **1-833-456-4566**
• Crisis Text Line: Text **TALK** to **686868**

**🌍 International:**
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Remember: You are not alone. These feelings can change. Help is available.**
"""

# --- Enhanced Sentiment Analysis ---
def analyze_sentiment_advanced(text: str) -> Dict:
    """Advanced sentiment analysis with emotion detection."""
    if not sentiment_analysis_available or not sentiment_pipeline:
        return {'sentiment': 'neutral', 'confidence': 0.5, 'intensity': 'moderate'}
    
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Convert labels and determine intensity
        if label in ['LABEL_2', 'POSITIVE']:
            sentiment = 'positive'
        elif label in ['LABEL_0', 'NEGATIVE']:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Determine emotional intensity
        if score > 0.8:
            intensity = 'high'
        elif score > 0.6:
            intensity = 'moderate'
        else:
            intensity = 'low'
        
        return {
            'sentiment': sentiment,
            'confidence': score,
            'intensity': intensity
        }
    except:
        return {'sentiment': 'neutral', 'confidence': 0.5, 'intensity': 'moderate'}

# --- Therapeutic Techniques ---
class TherapeuticTechniques:
    @staticmethod
    def breathing_exercise() -> str:
        return """
🌬️ **Box Breathing Exercise**

Let's do a simple breathing exercise together:

1. **Inhale** slowly for 4 counts (1... 2... 3... 4...)
2. **Hold** your breath for 4 counts (1... 2... 3... 4...)
3. **Exhale** slowly for 4 counts (1... 2... 3... 4...)
4. **Hold** empty for 4 counts (1... 2... 3... 4...)

Repeat this cycle 4-6 times. Focus only on your breathing.
"""

    @staticmethod
    def grounding_exercise() -> str:
        return """
🌱 **5-4-3-2-1 Grounding Technique**

This helps bring you back to the present moment:

• **5 things** you can **see** around you
• **4 things** you can **touch** or feel
• **3 things** you can **hear**
• **2 things** you can **smell**
• **1 thing** you can **taste**

Take your time with each step. This helps calm anxiety and racing thoughts.
"""

    @staticmethod
    def positive_affirmations() -> List[str]:
        return [
            "I am stronger than my challenges.",
            "This difficult moment will pass.",
            "I deserve love and compassion, especially from myself.",
            "I have overcome difficulties before, and I can do it again.",
            "My feelings are valid, and it's okay to feel them.",
            "I am growing and learning with each experience.",
            "I choose to be patient and kind with myself today."
        ]

    @staticmethod
    def coping_strategies(sentiment: str) -> str:
        strategies = {
            'negative': """
💪 **Coping Strategies for Difficult Times:**

• **Reach out** to a trusted friend or family member
• **Practice deep breathing** or meditation
• **Go for a walk** or do light exercise
• **Write down** your thoughts and feelings
• **Listen to calming music** or nature sounds
• **Take a warm bath** or shower
• **Do something creative** - draw, write, or craft
""",
            'positive': """
🌟 **Ways to Maintain Your Positive Energy:**

• **Share your joy** with someone you care about
• **Practice gratitude** - write down 3 good things today
• **Engage in activities** you love
• **Help someone else** - volunteering feels great
• **Celebrate your wins**, no matter how small
• **Take time to reflect** on your growth
• **Plan something** you're excited about
""",
            'neutral': """
🎯 **Self-Care Ideas for Today:**

• **Check in with yourself** - how are you really feeling?
• **Do one small thing** that brings you joy
• **Connect with nature** - even just look outside
• **Practice mindfulness** for 5 minutes
• **Organize** a small space around you
• **Learn something new** - watch a tutorial or read
• **Prepare a healthy meal** or snack mindfully
"""
        }
        return strategies.get(sentiment, strategies['neutral'])

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

def typewriter_effect(text: str, container, speed: float = 0.03, show_cursor: bool = True):
    """Create typewriter effect for text display."""
    if not text:
        return
    
    # Clean the text for better display
    text = text.strip()
    
    # Create placeholder for the typing text
    text_placeholder = container.empty()
    
    # Type out the text character by character
    displayed_text = ""
    
    for i, char in enumerate(text):
        displayed_text += char
        
        # Add cursor if enabled
        cursor_html = '<span class="cursor">|</span>' if show_cursor else ''
        
        # Update the display
        text_placeholder.markdown(
            f'<div class="typewriter">{displayed_text}{cursor_html}</div>', 
            unsafe_allow_html=True
        )
        
        # Variable speed based on character type
        if char in '.!?':
            time.sleep(speed * 8)  # Longer pause after sentences
        elif char in ',;:':
            time.sleep(speed * 4)  # Medium pause after clauses
        elif char == ' ':
            time.sleep(speed * 2)  # Short pause after words
        else:
            time.sleep(speed)  # Normal character speed
    
    # Final display without cursor
    if show_cursor:
        time.sleep(0.5)  # Brief pause before removing cursor
        text_placeholder.markdown(f'<div class="typewriter">{displayed_text}</div>', unsafe_allow_html=True)

def stream_response_with_typing(response_text: str, container):
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
            
        # Add sentence to full text
        if i > 0:
            full_text += " "
        
        # Type out each character in the sentence
        for char in sentence:
            full_text += char
            
            # Show with cursor
            text_placeholder.markdown(
                f'<div class="typewriter">{full_text}<span class="cursor">|</span></div>', 
                unsafe_allow_html=True
            )
            
            # Variable typing speed
            if char in '.!?':
                time.sleep(0.15)  # Longer pause after sentences
            elif char in ',;:':
                time.sleep(0.08)  # Medium pause after clauses
            elif char == ' ':
                time.sleep(0.04)  # Short pause after words
            elif char in '()"\'':
                time.sleep(0.02)  # Quick for punctuation
            else:
                time.sleep(0.025)  # Normal character speed
        
        # Pause between sentences
        if i < len(sentences) - 1:
            time.sleep(0.3)
    
    # Final display without cursor
    time.sleep(0.5)
    text_placeholder.markdown(f'<div class="typewriter">{full_text}</div>', unsafe_allow_html=True)

# --- Enhanced Response Generation ---
def get_therapeutic_response(user_input: str, sentiment_data: Dict, chat_history: str, crisis_detected: bool = False) -> str:
    """Generate therapeutic response with crisis handling."""
    if not model_available or not client:
        return "I'm experiencing technical difficulties. Please try again or seek immediate help if this is urgent."

    # Handle crisis situations first
    if crisis_detected:
        crisis_response = """I'm really concerned about you right now, and I want you to know that your life has value and meaning. What you're experiencing is incredibly difficult, but you don't have to face it alone.

Please consider reaching out for immediate support - there are people trained to help who are available 24/7. Your feelings are valid, but there are ways to work through this pain.

*"The darkest nights produce the brightest stars." - John Green*"""
        return crisis_response

    try:
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        intensity = sentiment_data['intensity']

        # Create contextual prompt based on sentiment and intensity
        therapeutic_context = f"""
You are MindfulChat Pro, an advanced AI therapist combining empathy with evidence-based therapeutic techniques.

User Input: "{user_input}"
Emotional State: {sentiment} ({intensity} intensity, {confidence:.1f} confidence)
Recent Context: {chat_history[-300:] if chat_history else 'New conversation'}

Response Guidelines:
1. Show genuine empathy and validation
2. Use therapeutic techniques (CBT, mindfulness, solution-focused)
3. Ask one thoughtful follow-up question
4. Include practical coping strategies if appropriate
5. End with an inspirational quote or affirmation
6. Keep response conversational yet professional (3-4 sentences)
7. Focus on the user's strengths and resilience

Therapeutic Approach:
- For negative emotions: Validate, normalize, provide coping strategies
- For positive emotions: Celebrate, reinforce, encourage gratitude
- For neutral states: Gently explore, encourage self-reflection
"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": therapeutic_context}],
            model="llama3-8b-8192",
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )

        response = chat_completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"I'm having some technical difficulties right now. In the meantime, remember that whatever you're going through, you're not alone. Please try again in a moment, or reach out to someone you trust."

# --- Mood Tracking ---
def initialize_mood_tracking():
    """Initialize mood tracking in session state."""
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = []
    if "daily_checkin" not in st.session_state:
        st.session_state.daily_checkin = {}

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
    positive_count = sum(1 for mood in recent_moods if mood['sentiment'] == 'positive')
    negative_count = sum(1 for mood in recent_moods if mood['sentiment'] == 'negative')
    
    if positive_count > negative_count:
        return f"📈 You've been experiencing more positive emotions lately ({positive_count}/{len(recent_moods)} recent interactions). Keep nurturing what's working!"
    elif negative_count > positive_count:
        return f"📊 You've been going through some challenges ({negative_count}/{len(recent_moods)} recent interactions). Remember, it's okay to have difficult days."
    else:
        return f"⚖️ Your emotions have been balanced lately. This shows emotional stability and resilience."

# --- Main Application ---
def main():
    # Header
    st.markdown("""
    # MindfulChat Pro
    
    <div class="subtitle">Your advanced AI companion for mental wellness and emotional support</div>
    """, unsafe_allow_html=True)

    # Initialize systems
    initialize_mood_tracking()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = """Welcome to MindfulChat Pro! I'm here to provide compassionate support and evidence-based therapeutic techniques to help you navigate life's challenges.

Whether you're dealing with stress, anxiety, relationship issues, or just need someone to talk to, I'm here to listen without judgment and offer practical guidance.

**How I can help:**
• Emotional support and validation
• Coping strategies and techniques
• Mindfulness and breathing exercises
• Crisis resources when needed
• Mood tracking and insights

*"You are braver than you believe, stronger than you seem, and smarter than you think." - A.A. Milne*"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Display mood insights in sidebar
    with st.sidebar:
        st.markdown("### 📊 Mood Insights")
        mood_insight = get_mood_insights()
        st.markdown(f'<div class="mood-tracker">{mood_insight}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🧘‍♀️ Quick Exercises")
        techniques = TherapeuticTechniques()
        
        if st.button("🌬️ Breathing Exercise"):
            # Add typing effect for exercises
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                time.sleep(1.5)
                typing_placeholder.empty()
                
                exercise = techniques.breathing_exercise()
                exercise_container = st.empty()
                stream_response_with_typing(exercise, exercise_container)
                
            st.session_state.messages.append({"role": "assistant", "content": exercise})
            st.rerun()
        
        if st.button("🌱 Grounding Exercise"):
            # Add typing effect for exercises
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                time.sleep(1.5)
                typing_placeholder.empty()
                
                exercise = techniques.grounding_exercise()
                exercise_container = st.empty()
                stream_response_with_typing(exercise, exercise_container)
                
            st.session_state.messages.append({"role": "assistant", "content": exercise})
            st.rerun()
        
        if st.button("💭 Daily Affirmation"):
            # Add typing effect for affirmations
            with st.chat_message("assistant"):
                typing_placeholder = display_typing_indicator()
                time.sleep(1)
                typing_placeholder.empty()
                
                import random
                affirmation = random.choice(techniques.positive_affirmations())
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

        # Generate chat history
        chat_history = "\n".join([
            f"{msg['role']}: {msg['content'][:150]}..." if len(msg['content']) > 150 else f"{msg['role']}: {msg['content']}"
            for msg in st.session_state.messages[-5:-1]
        ])

        # Generate response with dynamic typing
        with st.chat_message("assistant"):
            # Show typing indicator
            typing_placeholder = display_typing_indicator()
            
            if crisis_detected:
                # Show crisis resources immediately (no typing effect for urgent info)
                typing_placeholder.empty()
                crisis_resources = provide_crisis_resources()
                st.markdown(f'<div class="crisis-alert">{crisis_resources}</div>', unsafe_allow_html=True)
                
                # Brief pause before main response
                time.sleep(1)
                typing_placeholder = display_typing_indicator()
            
            # Generate the response
            response = get_therapeutic_response(prompt, sentiment_data, chat_history, crisis_detected)
            
            # Clear typing indicator and show response with typewriter effect
            typing_placeholder.empty()
            
            # Create container for the response
            response_container = st.empty()
            
            # Apply typewriter effect
            stream_response_with_typing(response, response_container)
            
            # Add coping strategies with typing effect
            if sentiment_data['sentiment'] in ['negative'] and sentiment_data['intensity'] in ['moderate', 'high']:
                time.sleep(0.5)  # Brief pause before additional resources
                
                # Show typing indicator for additional resources
                typing_placeholder_2 = display_typing_indicator()
                time.sleep(1.5)  # Simulate thinking time
                typing_placeholder_2.empty()
                
                techniques = TherapeuticTechniques()
                coping = techniques.coping_strategies(sentiment_data['sentiment'])
                
                # Type out coping strategies
                coping_container = st.empty()
                coping_container.markdown(f'<div class="resource-box">{coping}</div>', unsafe_allow_html=True)
                
                # Add a small typing effect for the resource box
                time.sleep(0.3)

        # Add assistant response to history
        full_response = response
        if crisis_detected:
            full_response = f"{provide_crisis_resources()}\n\n{response}"
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

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
