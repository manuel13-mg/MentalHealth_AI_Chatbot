import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
import speech_recognition as sr
import pyttsx3
from textblob import TextBlob
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="MindCare AI - Mental Health Support Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .crisis-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .supportive-message {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': '',
        'mood_trend': [],
        'session_count': 0,
        'crisis_alerts': 0
    }
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = 'neutral'

class EmotionAnalyzer:
    """Emotion analysis from text and facial expressions"""
    
    def __init__(self):
        # Initialize emotion analysis pipeline
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # Fallback to TextBlob for basic sentiment
            self.emotion_classifier = None
        
        # Crisis keywords for detection
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'hurt myself', 'self harm',
            'worthless', 'hopeless', 'can\'t go on', 'better off dead'
        ]
        
        # Emotion to color mapping
        self.emotion_colors = {
            'joy': '#4CAF50',
            'sadness': '#2196F3',
            'anger': '#F44336',
            'fear': '#FF9800',
            'surprise': '#9C27B0',
            'disgust': '#795548',
            'neutral': '#607D8B'
        }
    
    def analyze_text_emotion(self, text):
        """Analyze emotion from text"""
        if not text.strip():
            return {'emotion': 'neutral', 'confidence': 0.5, 'valence': 0.0}
        
        try:
            if self.emotion_classifier:
                result = self.emotion_classifier(text)
                emotion = result[0]['label'].lower()
                confidence = result[0]['score']
            else:
                # Fallback to TextBlob sentiment
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    emotion = 'joy'
                elif polarity < -0.1:
                    emotion = 'sadness'
                else:
                    emotion = 'neutral'
                confidence = min(abs(polarity) + 0.5, 1.0)
            
            # Calculate valence
            blob = TextBlob(text)
            valence = blob.sentiment.polarity
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'valence': valence
            }
        except Exception as e:
            st.error(f"Error in emotion analysis: {e}")
            return {'emotion': 'neutral', 'confidence': 0.5, 'valence': 0.0}
    
    def detect_crisis(self, text):
        """Detect potential crisis situations"""
        text_lower = text.lower()
        crisis_score = 0
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                crisis_score += 1
        
        # Additional patterns
        if re.search(r'\b(no\s+point|give\s+up|can\'t\s+take)\b', text_lower):
            crisis_score += 1
        
        return crisis_score > 0, crisis_score

class TherapeuticResponseGenerator:
    """Generate therapeutic and supportive responses"""
    
    def __init__(self):
        self.empathetic_responses = {
            'sadness': [
                "I can hear that you're going through a difficult time. Your feelings are completely valid.",
                "It sounds like you're carrying a heavy burden right now. I'm here to listen and support you.",
                "I understand you're feeling sad. Remember that it's okay to feel this way, and you're not alone."
            ],
            'anger': [
                "I can sense your frustration. It's natural to feel angry when things aren't going as expected.",
                "Your anger is telling us something important. Let's explore what might be behind these feelings.",
                "It's okay to feel angry. Let's work together to understand and process these emotions."
            ],
            'fear': [
                "I understand you're feeling anxious or scared. These feelings can be overwhelming, but you're safe here.",
                "Fear can be really difficult to deal with. Let's take this one step at a time.",
                "I hear your concerns. It's brave of you to share these feelings with me."
            ],
            'joy': [
                "I'm glad to hear some positivity in your words! It's wonderful when we can find moments of joy.",
                "That's great to hear! Celebrating positive moments is important for our wellbeing.",
                "Your positive energy is coming through. How can we build on this feeling?"
            ],
            'neutral': [
                "I'm here to listen. How are you feeling today?",
                "Thank you for sharing with me. What's on your mind?",
                "I appreciate you taking the time to talk. What would you like to explore today?"
            ]
        }
        
        self.coping_strategies = {
            'sadness': [
                "Try some gentle breathing exercises: breathe in for 4 counts, hold for 4, breathe out for 6.",
                "Consider reaching out to a trusted friend or family member.",
                "Gentle movement like a short walk can sometimes help lift our mood."
            ],
            'anger': [
                "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                "Physical exercise or even tensing and releasing your muscles can help release anger.",
                "Write down your feelings in a journal - sometimes getting thoughts out helps."
            ],
            'fear': [
                "Practice deep breathing: in through your nose for 4 counts, out through your mouth for 6 counts.",
                "Try progressive muscle relaxation, starting from your toes and working up.",
                "Ground yourself by focusing on your immediate surroundings and what you can control."
            ]
        }
    
    def generate_response(self, user_input, emotion_data, conversation_context):
        """Generate therapeutic response based on emotion and context"""
        emotion = emotion_data['emotion']
        confidence = emotion_data['confidence']
        
        # Select appropriate empathetic response
        if emotion in self.empathetic_responses:
            empathetic_response = np.random.choice(self.empathetic_responses[emotion])
        else:
            empathetic_response = np.random.choice(self.empathetic_responses['neutral'])
        
        # Add coping strategy if appropriate
        coping_strategy = ""
        if emotion in ['sadness', 'anger', 'fear'] and confidence > 0.6:
            if emotion in self.coping_strategies:
                coping_strategy = f"\n\nHere's something that might help: {np.random.choice(self.coping_strategies[emotion])}"
        
        # Add session context
        session_note = ""
        if len(conversation_context) > 3:
            session_note = "\n\nI notice we've been talking for a while. How are you feeling about our conversation so far?"
        
        return empathetic_response + coping_strategy + session_note

class VoiceSynthesizer:
    """Voice synthesis with emotional adaptation"""
    
    def __init__(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.voices = self.tts_engine.getProperty('voices')
            if self.voices:
                self.tts_engine.setProperty('voice', self.voices[1].id if len(self.voices) > 1 else self.voices[0].id)
        except:
            self.tts_engine = None
            st.warning("Text-to-speech not available on this system")
        
        self.voice_parameters = {
            'sadness': {'rate': 150, 'volume': 0.8},
            'anger': {'rate': 120, 'volume': 0.7},
            'fear': {'rate': 140, 'volume': 0.9},
            'joy': {'rate': 180, 'volume': 0.9},
            'neutral': {'rate': 160, 'volume': 0.8}
        }
    
    def speak(self, text, emotion='neutral'):
        """Convert text to speech with emotional adaptation"""
        if not self.tts_engine:
            return False
        
        try:
            params = self.voice_parameters.get(emotion, self.voice_parameters['neutral'])
            self.tts_engine.setProperty('rate', params['rate'])
            self.tts_engine.setProperty('volume', params['volume'])
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            st.error(f"Voice synthesis error: {e}")
            return False

# Initialize components
@st.cache_resource
def load_components():
    emotion_analyzer = EmotionAnalyzer()
    response_generator = TherapeuticResponseGenerator()
    voice_synthesizer = VoiceSynthesizer()
    return emotion_analyzer, response_generator, voice_synthesizer

emotion_analyzer, response_generator, voice_synthesizer = load_components()

# Main UI
def main():
    st.markdown('<h1 class="main-header">🧠 MindCare AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Personal Mental Health Support Companion</p>', unsafe_allow_html=True)
    
    # Sidebar for user profile and settings
    with st.sidebar:
        st.header("👤 User Profile")
        
        # User information
        user_name = st.text_input("Your Name", value=st.session_state.user_profile['name'])
        if user_name != st.session_state.user_profile['name']:
            st.session_state.user_profile['name'] = user_name
        
        st.metric("Sessions Completed", st.session_state.user_profile['session_count'])
        
        # Mode selection
        st.header("🎯 Interaction Mode")
        mode = st.selectbox("Choose Mode", ["Text Chat", "Voice Chat", "Video Analysis", "Mood Tracking"])
        
        # Settings
        st.header("⚙️ Settings")
        enable_voice = st.checkbox("Enable Voice Responses", value=True)
        show_emotion_analysis = st.checkbox("Show Emotion Analysis", value=True)
        crisis_detection = st.checkbox("Crisis Detection", value=True)
        
        # Quick stats
        if st.session_state.emotion_history:
            st.header("📊 Quick Stats")
            recent_emotions = [e['emotion'] for e in st.session_state.emotion_history[-10:]]
            if recent_emotions:
                most_common = max(set(recent_emotions), key=recent_emotions.count)
                st.metric("Recent Dominant Emotion", most_common.title())
    
    # Main content area
    if mode == "Text Chat":
        text_chat_interface(enable_voice, show_emotion_analysis, crisis_detection)
    elif mode == "Voice Chat":
        voice_chat_interface(show_emotion_analysis, crisis_detection)
    elif mode == "Video Analysis":
        video_analysis_interface(crisis_detection)
    elif mode == "Mood Tracking":
        mood_tracking_interface()

def text_chat_interface(enable_voice, show_emotion_analysis, crisis_detection):
    """Text-based chat interface"""
    st.header("💬 Text Chat")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.conversation_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                    <small style="color: #666; float: right;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>MindCare AI:</strong> {message['content']}
                    <small style="color: #666; float: right;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("Type your message here...", key="chat_input", placeholder="How are you feeling today?")
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    if send_button and user_input.strip():
        process_user_message(user_input, enable_voice, show_emotion_analysis, crisis_detection)
        st.rerun()
    
    # Emotion analysis display
    if show_emotion_analysis and st.session_state.emotion_history:
        display_emotion_analysis()

def process_user_message(user_input, enable_voice, show_emotion_analysis, crisis_detection):
    """Process user message and generate response"""
    timestamp = datetime.now().strftime("%H:%M")
    
    # Add user message to history
    st.session_state.conversation_history.append({
        'type': 'user',
        'content': user_input,
        'timestamp': timestamp
    })
    
    # Analyze emotion
    emotion_data = emotion_analyzer.analyze_text_emotion(user_input)
    st.session_state.current_emotion = emotion_data['emotion']
    
    # Add to emotion history
    st.session_state.emotion_history.append({
        'emotion': emotion_data['emotion'],
        'confidence': emotion_data['confidence'],
        'valence': emotion_data['valence'],
        'timestamp': datetime.now(),
        'text': user_input
    })
    
    # Crisis detection
    if crisis_detection:
        is_crisis, crisis_score = emotion_analyzer.detect_crisis(user_input)
        if is_crisis:
            st.session_state.user_profile['crisis_alerts'] += 1
            crisis_response = """
            I'm concerned about what you've shared. Your life has value and meaning. 
            Please consider reaching out to a crisis helpline:
            
            🆘 **Crisis Resources:**
            - National Suicide Prevention Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            - International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
            
            You don't have to go through this alone. Professional help is available.
            """
            
            st.session_state.conversation_history.append({
                'type': 'bot',
                'content': crisis_response,
                'timestamp': timestamp,
                'is_crisis': True
            })
            return
    
    # Generate therapeutic response
    bot_response = response_generator.generate_response(
        user_input, 
        emotion_data, 
        st.session_state.conversation_history
    )
    
    # Add bot response to history
    st.session_state.conversation_history.append({
        'type': 'bot',
        'content': bot_response,
        'timestamp': timestamp
    })
    
    # Voice synthesis
    if enable_voice:
        voice_synthesizer.speak(bot_response, emotion_data['emotion'])
    
    # Update session count
    st.session_state.user_profile['session_count'] += 1

def display_emotion_analysis():
    """Display emotion analysis visualization"""
    st.subheader("🎭 Emotion Analysis")
    
    if len(st.session_state.emotion_history) >= 1:
        # Recent emotion
        recent_emotion = st.session_state.emotion_history[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Emotion", 
                recent_emotion['emotion'].title(),
                f"{recent_emotion['confidence']:.2f} confidence"
            )
        
        with col2:
            st.metric(
                "Valence", 
                f"{recent_emotion['valence']:.2f}",
                "Positive" if recent_emotion['valence'] > 0 else "Negative"
            )
        
        with col3:
            st.metric(
                "Emotional Stability",
                "Stable" if len(set([e['emotion'] for e in st.session_state.emotion_history[-3:]])) <= 2 else "Variable"
            )
        
        # Emotion timeline
        if len(st.session_state.emotion_history) > 1:
            df = pd.DataFrame(st.session_state.emotion_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.line(df, x='timestamp', y='valence', 
                         title='Emotional Valence Over Time',
                         labels={'valence': 'Emotional Valence', 'timestamp': 'Time'})
            fig.update_traces(line_color='#667eea')
            st.plotly_chart(fig, use_container_width=True)

def voice_chat_interface(show_emotion_analysis, crisis_detection):
    """Voice chat interface"""
    st.header("🎤 Voice Chat")
    st.info("This feature would integrate with speech recognition and real-time audio processing.")
    
    # Placeholder for voice chat implementation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", type="primary"):
            st.success("Recording started... (This is a demo placeholder)")
            # In a real implementation, this would start audio recording
    
    with col2:
        if st.button("⏹️ Stop Recording"):
            st.info("Recording stopped. Processing speech... (Demo)")
            # Here you would process the audio and convert to text
    
    # Simulated voice input for demo
    st.subheader("Voice Input Simulation")
    simulated_speech = st.text_area("Simulate speech input:", placeholder="This would be the transcribed speech...")
    
    if st.button("Process Voice Input") and simulated_speech:
        process_user_message(simulated_speech, True, show_emotion_analysis, crisis_detection)
        st.rerun()

def video_analysis_interface(crisis_detection):
    """Video analysis interface with facial emotion recognition"""
    st.header("📹 Video Analysis")
    st.info("Real-time facial emotion recognition during video calls")
    
    # Video processing would go here
    st.subheader("Facial Emotion Detection")
    
    # Simulated emotion detection results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detected Emotions")
        
        # Simulated emotion data
        emotions = ['joy', 'sadness', 'surprise', 'neutral', 'fear']
        confidences = np.random.rand(5)
        
        for emotion, confidence in zip(emotions, confidences):
            st.progress(confidence, f"{emotion.title()}: {confidence:.2f}")
    
    with col2:
        st.subheader("Facial Landmarks")
        st.info("Facial landmark detection would be visualized here")
        
        # Create a simple emotion radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=confidences,
            theta=emotions,
            fill='toself',
            name='Emotion Intensity'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Current Emotional State"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def mood_tracking_interface():
    """Mood tracking and analytics interface"""
    st.header("📊 Mood Tracking & Analytics")
    
    if not st.session_state.emotion_history:
        st.info("Start chatting to see your mood analytics!")
        return
    
    # Create DataFrame from emotion history
    df = pd.DataFrame(st.session_state.emotion_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Mood overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", len(df))
    
    with col2:
        avg_valence = df['valence'].mean()
        st.metric("Average Mood", f"{avg_valence:.2f}", 
                 "😊" if avg_valence > 0 else "😔")
    
    with col3:
        most_common_emotion = df['emotion'].mode().iloc[0] if not df.empty else 'neutral'
        st.metric("Dominant Emotion", most_common_emotion.title())
    
    with col4:
        mood_variance = df['valence'].std()
        stability = "Stable" if mood_variance < 0.3 else "Variable"
        st.metric("Emotional Stability", stability)
    
    # Detailed analytics
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎭 Emotion Distribution", "📅 Daily Patterns"])
    
    with tab1:
        # Mood trend over time
        fig = px.line(df, x='timestamp', y='valence', 
                     title='Mood Trend Over Time',
                     labels={'valence': 'Mood Valence', 'timestamp': 'Time'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Neutral Line")
        st.plotly_chart(fig, use_container_width=True)
        
        # Moving average
        if len(df) > 5:
            df_sorted = df.sort_values('timestamp')
            df_sorted['moving_avg'] = df_sorted['valence'].rolling(window=3).mean()
            
            fig2 = px.line(df_sorted, x='timestamp', y='moving_avg',
                          title='3-Point Moving Average of Mood')
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Emotion distribution
        emotion_counts = df['emotion'].value_counts()
        
        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                    title='Emotion Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Emotion confidence levels
        fig2 = px.box(df, x='emotion', y='confidence',
                     title='Emotion Detection Confidence by Type')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Daily patterns
        df['hour'] = df['timestamp'].dt.hour
        hourly_mood = df.groupby('hour')['valence'].mean().reset_index()
        
        fig = px.bar(hourly_mood, x='hour', y='valence',
                    title='Average Mood by Hour of Day')
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly patterns if enough data
        if len(df) > 7:
            df['day_of_week'] = df['timestamp'].dt.day_name()
            daily_mood = df.groupby('day_of_week')['valence'].mean().reset_index()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_mood['day_of_week'] = pd.Categorical(daily_mood['day_of_week'], categories=day_order, ordered=True)
            daily_mood = daily_mood.sort_values('day_of_week')
            
            fig2 = px.bar(daily_mood, x='day_of_week', y='valence',
                         title='Average Mood by Day of Week')
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>🧠 MindCare AI</strong> - Your Mental Health Support Companion</p>
        <p>⚠️ <em>This is an AI assistant and not a replacement for professional mental health care.</em></p>
        <p>If you're experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately.</p>
        <p>Made with ❤️ using Streamlit & Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()