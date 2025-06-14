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
    """
    Enhanced detection of crisis-related content in user input.
    Uses more comprehensive patterns and contextual analysis.
    """
    crisis_patterns = {
        'suicide': [
            r'\b(suicide|suicidal|kill myself|end my life|want to die|not worth living)\b',
            r'\b(end it all|take my own life|better off dead|no point living)\b',
            r'\b(can\'t take it anymore|don\'t want to be here|don\'t want to exist)\b',
            r'\b(life is too painful|no reason to live|saying goodbye|final message)\b'
        ],
        'self_harm': [
            r'\b(self harm|hurt myself|cut myself|harm myself)\b',
            r'\b(cutting|burning myself|self injury|self-injury|injure myself)\b',
            r'\b(physical pain|punish myself|deserve pain|feel something)\b',
            r'\b(blood|scars|wounds|blades|razors|pills)\b'
        ],
        'emergency': [
            r'\b(emergency|crisis|help me|desperate|can\'t go on)\b',
            r'\b(nobody cares|completely alone|no way out|trapped)\b',
            r'\b(hopeless|helpless|unbearable|overwhelming|suffocating)\b',
            r'\b(can\'t cope|breaking down|falling apart|losing control)\b'
        ],
        'abuse': [
            r'\b(abused|abusive|violence|violent|assault|assaulted)\b',
            r'\b(afraid for my safety|threatened|in danger|unsafe)\b',
            r'\b(hurt me|hitting|beating|physical abuse|emotional abuse)\b'
        ],
        'severe_depression': [
            r'\b(severely depressed|major depression|can\'t function)\b',
            r'\b(haven\'t eaten|can\'t sleep|can\'t get out of bed|given up)\b',
            r'\b(nothing matters|empty inside|hollow|numb|void)\b'
        ]
    }
    
    # Context patterns that increase severity assessment
    severity_amplifiers = [
        r'\b(right now|tonight|today|immediately|urgent)\b',
        r'\b(plan|planned|preparing|ready to|going to)\b',
        r'\b(nobody knows|secret|hidden|alone|isolated)\b',
        r'\b(tried everything|last resort|only option|no choice)\b',
        r'\b(always|never|forever|permanent|final)\b'
    ]
    
    text_lower = text.lower()
    
    # Check for crisis patterns
    for category, patterns in crisis_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Check for severity amplifiers
                for amplifier in severity_amplifiers:
                    if re.search(amplifier, text_lower):
                        # Higher priority if amplifiers are present
                        return True, category
                return True, category
    
    # Secondary check for combinations of concerning phrases
    concern_count = 0
    all_patterns = [pattern for patterns in crisis_patterns.values() for pattern in patterns]
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            concern_count += 1
    
    # If multiple concerning phrases are detected, treat as potential crisis
    if concern_count >= 2:
        return True, "multiple_indicators"
    
    return False, None

def get_crisis_response() -> str:
    """Provide an empathetic and actionable crisis response with bullet points."""
    return """I'm genuinely concerned about what you're sharing right now.

• Your life has immense value and meaning, even if it might not feel that way at this moment
• What you're experiencing is incredibly difficult, and I want you to know that you don't have to face it alone
• Your pain is real, and your feelings are valid

Please consider reaching out for immediate professional support - there are compassionate people trained specifically to help with these feelings who are available 24/7.

*Important reminders:*

• This moment is temporary, even though the pain feels overwhelming
• Many people who have felt this way have found their way to better days
• Reaching out for help is a sign of strength, not weakness
• You deserve support and compassion

*Immediate steps you can take:*

• Call or text a crisis hotline (988 in the US)
• Reach out to someone you trust
• Focus on just getting through the next hour, not the whole day
• Remove access to anything that could be used to harm yourself

"Hope is the thing with feathers that perches in the soul and sings the tune without the words and never stops at all." - Emily Dickinson"""

def provide_crisis_resources() -> str:
    """Provide comprehensive and immediate crisis resources."""
    return """
# 🚨 *IMMEDIATE HELP AVAILABLE* 🚨

If you're in immediate danger or experiencing a crisis, please reach out now. You deserve support, and trained professionals are ready to help 24/7.

## 📞 *Crisis Hotlines by Country*

### 🇺🇸 *United States*
| Service | Contact | Description |
|---------|---------|-------------|
| *National Suicide & Crisis Lifeline* | *988* | Call or text for immediate support |
| *Crisis Text Line* | Text *HOME* to *741741* | 24/7 text-based support |
| *Emergency Services* | *911* | For immediate danger |
| *Trevor Project (LGBTQ+)* | *1-866-488-7386* | Specialized support for LGBTQ+ youth |
| *Veterans Crisis Line* | *988*, then press 1 | Support for veterans and their loved ones |

### 🇬🇧 *United Kingdom*
| Service | Contact | Description |
|---------|---------|-------------|
| *Samaritans* | *116 123* | Free, 24/7 emotional support |
| *Crisis Text Line* | Text *SHOUT* to *85258* | 24/7 text-based support |
| *CALM (for men)* | *0800 58 58 58* | Support for men in crisis |
| *Emergency Services* | *999* | For immediate danger |

### 🇨🇦 *Canada*
| Service | Contact | Description |
|---------|---------|-------------|
| *Talk Suicide Canada* | *1-833-456-4566* | 24/7 phone support |
| *Crisis Text Line* | Text to *45645* | Text support (4pm-midnight ET) |
| *Emergency Services* | *911* | For immediate danger |

### 🇦🇺 *Australia*
| Service | Contact | Description |
|---------|---------|-------------|
| *Lifeline Australia* | *13 11 14* | 24/7 crisis support |
| *Beyond Blue* | *1300 22 4636* | Mental health support |
| *Emergency Services* | *000* | For immediate danger |

## 🌍 *International Resources*
• *International Association for Suicide Prevention*: 
  https://www.iasp.info/resources/Crisis_Centres/

• *Befrienders Worldwide*: 
  https://www.befrienders.org/

## 🛡 *Specialized Support*

### For Abuse or Domestic Violence
• *National Domestic Violence Hotline (US)*: 
  *1-800-799-7233* or text *START* to *88788*

• *National Sexual Assault Hotline (US)*: 
  *1-800-656-HOPE (4673)*

### Online Support Communities
• *7 Cups*: https://www.7cups.com/ 
  Free emotional support from trained listeners

• *TalkLife*: https://talklife.com/ 
  Peer support community for mental health

## 💙 *Remember*
• You are not alone in this moment
• These feelings are temporary, even when they feel permanent
• Reaching out is an act of courage, not weakness
• Professional help can make a significant difference
• Many people have felt this way and found their way through

*Your life matters. Help is available right now.*
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
# 🌬 *Box Breathing Exercise (4-4-4-4)*

This evidence-based breathing technique is used by Navy SEALs and healthcare professionals to reduce stress and increase focus.

## *Step-by-Step Guide:*

### ⿡ *INHALE (4 seconds)* 
1. Sit comfortably with your back straight
2. Breathe in slowly through your nose 
3. Count to 4: (1... 2... 3... 4...)
4. Fill your lungs completely from bottom to top

### ⿢ *HOLD FULL (4 seconds)* 
1. Keep your lungs full
2. Count to 4: (1... 2... 3... 4...)
3. Maintain a gentle hold, not forced
4. Keep your shoulders relaxed

### ⿣ *EXHALE (4 seconds)* 
1. Release breath completely through your mouth
2. Count to 4: (1... 2... 3... 4...)
3. Empty your lungs fully
4. Feel tension leaving your body

### ⿤ *HOLD EMPTY (4 seconds)* 
1. Keep your lungs empty
2. Count to 4: (1... 2... 3... 4...)
3. Relax your chest and abdomen
4. Prepare for the next inhale

*Practice:* Repeat this cycle 4-6 times. Focus only on your breathing and the counting.

## *Benefits:*
1. Reduces stress and anxiety
2. Lowers blood pressure
3. Improves concentration and focus
4. Regulates your autonomic nervous system
5. Reduces cortisol levels
6. Activates your parasympathetic "rest and digest" response

*When to Use:*
- Before stressful situations
- During moments of anxiety
- To improve focus before important tasks
- As part of a daily relaxation routine
- To help with sleep onset

Try practicing this technique 2-3 times daily, especially during stressful moments.
"""

    @staticmethod
    def grounding_exercise() -> str:
        return """
# 🌱 *5-4-3-2-1 Grounding Technique*

This powerful mindfulness exercise helps anchor you to the present moment when feeling overwhelmed, anxious, or disconnected.

## *Instructions:* 
Take your time with each step, really noticing the details. Move through each sense deliberately and mindfully.

### 👁 *5 things you can SEE*

1. Look around and notice five distinct objects
2. For each object, observe:
   - Its colors and shades
   - Its shape and texture
   - Any patterns or details
   - How light interacts with it
3. Name each item specifically in your mind
4. Focus on details you might normally overlook

### 👐 *4 things you can TOUCH or FEEL*

1. Find four different textures around you
2. For each texture, notice:
   - Is it smooth, rough, soft, or hard?
   - Its temperature (cool, warm, neutral)
   - Any weight or pressure
   - How your skin responds to it
3. Describe the sensations to yourself in detail
4. Pay attention to how your body feels in space

### 👂 *3 things you can HEAR*

1. Listen for three distinct sounds in your environment
2. For each sound, notice:
   - Its volume (loud or soft)
   - Its pitch (high or low)
   - Its rhythm or pattern
   - Its distance from you
3. Include nearby sounds, distant sounds, and your own breathing
4. Listen for subtle sounds you might normally filter out

### 👃 *2 things you can SMELL*

1. Notice two scents in your environment
2. For each scent:
   - Is it pleasant, neutral, or unpleasant?
   - Is it strong or subtle?
   - Is it familiar or new?
   - What associations or memories does it trigger?
3. If you can't smell anything, recall two favorite scents
4. Notice how the scents affect your mood and body

### 👄 *1 thing you can TASTE*

1. Notice one taste currently in your mouth
2. If there's no distinct taste:
   - Take a small sip of water or tea
   - Or imagine tasting something you enjoy
3. Pay attention to:
   - The specific flavor qualities
   - Where you sense it on your tongue
   - Any physical response it creates
4. Notice how this completes the full sensory experience

## *Why This Works:*

1. *Neural Pathway Activation:*
   - Engages all five senses simultaneously
   - Activates different neural pathways than those used in worry
   - Creates new patterns of attention in the brain

2. *Physiological Benefits:*
   - Interrupts the fight-or-flight response
   - Reduces cortisol and adrenaline levels
   - Activates the parasympathetic nervous system
   - Slows heart rate and breathing

3. *Psychological Impact:*
   - Breaks cycles of rumination and worry
   - Creates distance from overwhelming emotions
   - Builds present-moment awareness
   - Provides a sense of control and stability

Practice this whenever you feel overwhelmed, anxious, or disconnected from the present moment. The more you use this technique, the more effective it becomes.
"""

    @staticmethod
    def progressive_muscle_relaxation() -> str:
        return """
# 💆 *Progressive Muscle Relaxation*

This evidence-based technique helps release physical tension you might not even realize you're holding in your body.

## *Preparation:*

1. *Find a comfortable position*
   - Sit in a supportive chair or lie down
   - Ensure your back is supported
   - Loosen any tight clothing
   - Remove shoes if possible

2. *Set the environment*
   - Dim lights if possible
   - Minimize distractions
   - Consider setting a gentle timer
   - Allow 15-20 minutes for the full exercise

3. *Begin with breathing*
   - Take 3-5 deep, slow breaths
   - Inhale through your nose for 4 counts
   - Exhale through your mouth for 6 counts
   - Let your body begin to settle

## *Basic Process for Each Muscle Group:*

### ⿡ *TENSE (5 seconds)* ⏫
   - Tighten the muscles firmly (but not painfully)
   - Hold for a full 5 seconds
   - Focus completely on the sensation of tension
   - Notice what tension actually feels like

### ⿢ *RELEASE* ⏬
   - Let go suddenly and completely
   - Feel the difference as tension flows out
   - Notice the pleasant sensation of relaxation
   - Rest for 10 seconds before moving on
   - Observe how relaxation differs from tension

## *Recommended Sequence:*

### 🦶 *Lower Body*
1. *Feet*
   - Curl your toes downward tightly
   - Feel the tension in the arches and soles
   - Release and notice the relaxation

2. *Calves*
   - Point toes toward your face
   - Feel the stretch and tension in your calves
   - Release and let the muscles soften

3. *Thighs*
   - Press knees together firmly
   - Feel the large muscles engage
   - Release and feel heaviness and warmth

### 🧘 *Core*
4. *Abdomen*
   - Pull your stomach in tightly
   - Feel the tension across your midsection
   - Release and feel your breathing deepen

5. *Chest*
   - Take a deep breath and hold it
   - Feel the expansion and tension
   - Release with a slow exhale

6. *Lower back*
   - Arch slightly or press back against surface
   - Feel the engagement of back muscles
   - Release and feel your spine settle

### 💪 *Upper Body*
7. *Hands*
   - Make tight fists with both hands
   - Feel tension through fingers and palms
   - Release and notice tingling sensations

8. *Arms*
   - Bend elbows and flex biceps
   - Feel the tension in upper and lower arms
   - Release and feel warmth flowing in

9. *Shoulders*
   - Raise shoulders toward your ears
   - Feel the tension across upper back
   - Release and feel shoulders drop completely

10. *Neck*
    - Gently press head back or tilt chin to chest
    - Feel the stretch along neck muscles
    - Release and feel neck lengthen

11. *Face*
    - Scrunch all facial muscles together
    - Feel tension in forehead, cheeks, jaw
    - Release and feel your face smooth out

## *Closing the Practice:*
1. Scan your body for any remaining tension
2. Take 3 deep, satisfying breaths
3. Wiggle fingers and toes gently
4. Open your eyes if they were closed
5. Move slowly when you're ready to get up

## *Benefits:*
1. Reduces physical tension and muscle pain
2. Lowers blood pressure and heart rate
3. Decreases stress hormones in the bloodstream
4. Improves sleep quality and reduces insomnia
5. Reduces symptoms of anxiety and stress
6. Increases body awareness and mindfulness
7. Signals your nervous system to shift from "fight-or-flight" to "rest-and-digest" mode

*When to Use:*
- Before bed to improve sleep
- During high-stress periods
- After sitting for long periods
- When experiencing muscle pain
- As part of a regular relaxation routine

Regular practice (even just 5-10 minutes daily) can significantly reduce overall tension levels and improve your body's relaxation response.
"""

    @staticmethod
    def thought_reframing() -> str:
        return """
# 🔄 *Cognitive Reframing Exercise*

This powerful CBT technique helps challenge and transform unhelpful thought patterns that contribute to stress, anxiety, and low mood.

## *Step-by-Step Process:*

### ⿡ *IDENTIFY the Negative Thought*
   
*First, catch the thought:*
1. Notice when you're feeling upset or stressed
2. Ask yourself: "What am I telling myself about this situation?"
3. Write down the specific thought exactly as it appears in your mind
4. Example: "I completely failed at this task. I'm terrible at everything."

### ⿢ *EXAMINE the Evidence*
   
*Look at supporting evidence:*
1. What facts actually support this thought?
2. Be specific and objective
3. Write down only verifiable facts
   
*Consider contradicting evidence:*
1. What facts don't support or contradict this thought?
2. What positive aspects are you overlooking?
3. What strengths or resources do you have?
   
*Perform a feeling vs. fact check:*
1. Are you treating feelings as facts?
2. Example: "Feeling incompetent" ≠ "Being incompetent"
3. Separate emotional reactions from objective reality

### ⿣ *EXPLORE Alternative Perspectives*
   
*Take a compassionate friend view:*
1. What would you tell a friend with this same thought?
2. Would you judge them as harshly?
3. What compassionate advice would you offer?
   
*Consider a neutral observer view:*
1. How might someone else view this situation?
2. What would they notice that you're missing?
3. How might they describe what happened?
   
*Adopt a future perspective:*
1. How important will this seem in a week?
2. How about in a month?
3. Will this matter a year from now?

### ⿤ *CREATE a Balanced Thought*
   
*Develop a new thought that:*
1. Acknowledges any real challenges
2. Incorporates the evidence you've gathered
3. Includes your strengths and resources
4. Is realistic and helpful
   
*Example transformation:*
- *Original thought:* "I completely failed at this task. I'm terrible at everything."
- *Balanced thought:* "I struggled with parts of this task, but I did complete it. I have succeeded at similar things before, and I can learn from this experience."

## *Benefits:*
1. Reduces emotional distress
2. Improves problem-solving abilities
3. Increases resilience to stress
4. Breaks cycles of negative thinking
5. Creates more flexible thinking patterns
6. Builds self-awareness and emotional intelligence

*Practice Tips:*
- Start with less intense thoughts as you learn the technique
- Write down your thoughts and responses
- Practice regularly, even with small daily concerns
- Notice patterns in your thinking over time

This technique becomes more powerful with practice. Try using it daily to build your "mental fitness."
"""

    @staticmethod
    def positive_affirmations() -> List[str]:
        return [
            "I am stronger than my challenges and growing through what I'm going through.",
            "This difficult moment will pass. All emotions are temporary visitors.",
            "I deserve love and compassion, especially from myself, exactly as I am right now.",
            "I have overcome difficulties before, and each experience has built resilience I can draw on today.",
            "My feelings are valid messengers, not permanent states. I can acknowledge them with kindness.",
            "I am growing and learning with each experience, even the difficult ones.",
            "I choose to be patient and kind with myself today, treating myself as I would a dear friend.",
            "I am allowed to set boundaries that protect my peace and wellbeing.",
            "My worth is inherent and not determined by productivity, achievement, or others' opinions.",
            "I can handle uncertainty by focusing on what I can control in this moment.",
            "I am exactly where I need to be on my journey, learning exactly what I need to learn.",
            "I release the need for perfection and embrace progress, however small.",
            "Each breath connects me to the present moment, where peace can be found.",
            "I trust in my ability to navigate life's challenges with wisdom and courage."
        ]

    @staticmethod
    def coping_strategies(sentiment: str) -> str:
        strategies = {
            'negative': """
💪 *Evidence-Based Coping Strategies for Difficult Times:*

• *Social Connection*: 
  Reach out to a trusted person - even brief contact reduces stress hormones

• *Physical Movement*: 
  A 10-minute walk can significantly reduce anxiety for several hours

• *Sensory Grounding*: 
  Hold something cold (ice cube) or smell something strong (essential oil) to interrupt distress

• *Opposite Action*: 
  Do the opposite of what negative emotions urge (e.g., go outside when depression says stay in bed)

• *Emotional Acceptance*: 
  Name your feelings specifically - "I'm feeling disappointed and anxious" - which reduces their intensity

• *Brief Mindfulness*: 
  Focus completely on a simple task like washing dishes, noticing all sensations

• *Self-Compassion Break*: 
  Place hand on heart, acknowledge suffering ("this is hard"), and offer kindness to yourself

• *Cognitive Defusion*: 
  Add "I'm having the thought that..." before negative thoughts to create helpful distance

• *Values Reminder*: 
  Do one small action aligned with your core values, even when motivation is low

• *Behavioral Activation*: 
  Schedule one enjoyable activity, however small, and commit to it regardless of mood
""",
            'positive': """
🌟 *Evidence-Based Ways to Sustain and Build on Positive Emotions:*

• *Savoring Practice*: 
  Intentionally notice and extend positive experiences by focusing on sensory details

• *Gratitude Journaling*: 
  Write 3 specific things you're grateful for, including why they matter to you

• *Strength Spotting*: 
  Identify which personal strengths you used today and how you might use them tomorrow

• *Benefit Finding*: 
  Reflect on challenges you've faced and identify unexpected positive outcomes or growth

• *Acts of Kindness*: 
  Small acts of generosity create a positive feedback loop of wellbeing for both giver and receiver

• *Flow Activities*: 
  Engage in activities that fully absorb your attention and match your skill level with challenge

• *Positive Reminiscence*: 
  Revisit and mentally relive past positive experiences in vivid detail

• *Celebration Ritual*: 
  Create a small ritual to mark achievements or positive moments (however small)

• *Awe Experience*: 
  Seek out experiences of wonder and vastness (nature, art, music) that transcend daily concerns

• *Social Sharing*: 
  Tell someone about your positive experience - this "capitalizes" on positive emotions, amplifying them
""",
            'neutral': """
🎯 *Evidence-Based Self-Care and Emotional Regulation Practices:*

• *Emotional Check-In*: 
  Take 60 seconds to scan your body for physical sensations of emotion without judgment

• *Values Clarification*: 
  Reflect on what truly matters to you and identify one small aligned action for today

• *Mindful Observation*: 
  Choose one ordinary object and observe it for 2 minutes as if seeing it for the first time

• *Habit Stacking*: 
  Attach a small wellbeing practice to an existing daily habit (e.g., deep breathing while waiting for coffee)

• *Media Nutrition*: 
  Audit your content consumption and adjust toward what genuinely nourishes your mind

• *Nature Connection*: 
  Spend 20 minutes in a natural setting - research shows this significantly reduces stress hormones

• *Boundary Practice*: 
  Identify one situation where you need clearer boundaries and plan a specific response

• *Curiosity Cultivation*: 
  Learn something new about a topic of interest through a 10-minute exploration

• *Body Scan Meditation*: 
  Progressively bring attention to each part of your body, releasing tension as you go

• *Intentional Rest*: 
  Schedule a brief period of true rest (not just switching between activities) without guilt
"""
        }
        return strategies.get(sentiment, strategies['neutral'])
        
    @staticmethod
    def mindfulness_practice() -> str:
        return """
# 🧘 *3-Minute Mindfulness Practice*

This brief but powerful practice can be done anywhere to center yourself, reduce stress, and increase present-moment awareness.

## *A Simple 3-Step Process:*

### ⏱ *MINUTE 1: Becoming Aware*
   
*Step 1: Prepare*
1. Find a comfortable position
2. Close your eyes or soften your gaze
3. Take a deep breath and ask yourself:
   "What am I experiencing right now?"
   
*Step 2: Notice without judging:*
1. Your thoughts (without getting caught in their content)
2. Your emotions (naming them simply: "anxiety," "frustration," etc.)
3. Physical sensations in your body (tension, comfort, discomfort)
4. Sounds in your environment
   
*Key point:* Simply observe what's present, like watching clouds pass in the sky. No need to change anything.

### ⏱ *MINUTE 2: Focusing Attention*
   
*Step 1: Direct your focus*
1. Bring your full attention to your breathing
2. Choose a specific sensation to focus on:
   - The feeling of air entering and leaving your nostrils
   - The rising and falling of your chest
   - The expanding and contracting of your abdomen
   
*Step 2: When your mind wanders (which is normal):*
1. Notice that your attention has drifted
2. Gently return focus to your breath without self-criticism
3. Each time you notice and return is a moment of mindfulness
   
*Key point:* No need to control the breath - just observe its natural rhythm.

### ⏱ *MINUTE 3: Expanding Awareness*
   
*Step 1: Broaden your attention*
1. Maintain awareness of your breathing
2. Gradually expand attention to include your whole body
3. Notice:
   - Your posture and how you're holding yourself
   - Any areas of tension or relaxation
   - Your facial expression
   - The points where your body contacts the surface beneath you
   
*Step 2: Closing the practice:*
1. Feel your body as a whole, present in this moment
2. Take one deep, intentional breath
3. Carry this awareness with you as you return to your day

## *Benefits:*
1. Activates your parasympathetic nervous system
2. Reduces cortisol and other stress hormones
3. Creates a pause between stimulus and response
4. Improves focus and attention
5. Reduces rumination and worry
6. Increases self-awareness

Try practicing this 3-minute exercise once or twice daily, especially during stressful periods.

Even just 3 minutes of mindfulness can reset your nervous system and bring you back to center. Try practicing at regular intervals throughout your day.
"""

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
    """Generate therapeutic response with enhanced crisis handling and personalized support."""
    if not model_available or not client:
        return "I'm experiencing technical difficulties. Please try again or seek immediate help if this is urgent."

    # Handle crisis situations first with more empathetic and actionable response
    if crisis_detected:
        crisis_response = """I'm genuinely concerned about what you're sharing right now.

*Your feelings matter deeply, and you're not alone in this.*

## Important things to remember:

1. *Your life has immense value and meaning*
   • Even if it might not feel that way at this moment
   • Your presence in this world matters more than you may realize

2. *What you're experiencing is incredibly difficult*
   • Your pain is real, and your feelings are valid
   • You don't have to face these feelings alone

3. *This moment is temporary*
   • Even though the pain feels overwhelming right now
   • Many people who have felt this way have found their way to better days

## Immediate steps you can take:

1. *Reach out for professional support*
   • Call or text a crisis hotline (988 in the US)
   • These trained professionals are available 24/7 specifically to help

2. *Connect with someone you trust*
   • A friend, family member, or mentor
   • Simply saying "I'm struggling and need support" is a brave first step

3. *Focus on just getting through the next hour*
   • Not the whole day or week
   • Use grounding techniques to stay present

"Hope is the thing with feathers that perches in the soul and sings the tune without the words and never stops at all." - Emily Dickinson

*You deserve support and compassion. Reaching out for help is a sign of strength, not weakness.*"""
        return crisis_response

    try:
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        intensity = sentiment_data['intensity']

        # Enhanced contextual prompt with more personalized therapeutic approaches and structured formatting
        therapeutic_context = f"""
You are MindfulChat Pro, an advanced AI therapist combining deep empathy with evidence-based therapeutic techniques. Your goal is to provide meaningful, personalized support that helps users process their emotions and develop healthy coping strategies.

User Input: "{user_input}"
Emotional State: {sentiment} ({intensity} intensity, {confidence:.1f} confidence)
Recent Context: {chat_history[-300:] if chat_history else 'New conversation'}

IMPORTANT: Format your response with clear structure using the following format:

1. *Empathetic Connection*: Begin with a brief, genuine acknowledgment of their feelings (1-2 sentences)

2. *Validation*: Normalize their experience with a specific statement like "What you're feeling is completely understandable because..." (1-2 sentences)

3. *Perspective*: Offer a gentle reframe or new perspective that might be helpful (1-2 sentences)

4. *Specific Suggestions*: Provide 3 numbered, concrete action steps they can take:
   
   *Suggestion 1:* [Brief action step starting with a verb]
   • [Why this helps/how to implement]
   
   *Suggestion 2:* [Brief action step starting with a verb]
   • [Why this helps/how to implement]
   
   *Suggestion 3:* [Brief action step starting with a verb]
   • [Why this helps/how to implement]

5. *Reflection Question*: End with one thoughtful question that encourages deeper insight

6. *Supportive Closing*: Add a brief encouraging statement or relevant quote

Therapeutic Approach Based on Emotional State:
- For negative emotions ({intensity} intensity):
  * Validate their feelings without judgment
  * Normalize their experience ("Many people feel this way")
  * Provide specific coping strategies for immediate relief
  * Gently challenge unhelpful thought patterns if present
  * Emphasize their past strengths and resilience

- For positive emotions ({intensity} intensity):
  * Celebrate their positive state authentically
  * Reinforce behaviors that led to this positive feeling
  * Encourage gratitude practice and savoring the moment
  * Help identify ways to build upon this positive momentum
  * Connect this positive state to their values and goals

- For neutral states:
  * Gently explore underlying feelings that may be present
  * Encourage mindful self-reflection and awareness
  * Offer perspective-taking exercises
  * Suggest small, meaningful actions aligned with their values
  * Provide space for processing and integration

IMPORTANT: Make sure your suggestions are clearly numbered and formatted for easy reading. Each suggestion should be actionable, specific, and include a brief explanation of why/how it helps.
"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": therapeutic_context}],
            model="llama3-8b-8192",
            max_tokens=350,  # Increased token limit for more comprehensive responses
            temperature=0.7,
            top_p=0.9,
            stream=False
        )

        response = chat_completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"I'm having some technical difficulties right now. In the meantime, remember that whatever you're going through, you're not alone. Your feelings matter, and there are people who care about you. Please try again in a moment, or consider reaching out to someone you trust for support."

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
        return f"⚖ Your emotions have been balanced lately. This shows emotional stability and resilience."

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

*How I can help:*
• Emotional support and validation
• Coping strategies and techniques
• Mindfulness and breathing exercises
• Crisis resources when needed
• Mood tracking and insights

"You are braver than you believe, stronger than you seem, and smarter than you think." - A.A. Milne"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Display mood insights in sidebar
    with st.sidebar:
        st.markdown("### 📊 Mood Insights")
        mood_insight = get_mood_insights()
        st.markdown(f'<div class="mood-tracker">{mood_insight}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🧘‍♀ Therapeutic Exercises")
        techniques = TherapeuticTechniques()
        
        # Create two columns for better organization of buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌬 Breathing Exercise"):
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
        
        with col2:
            if st.button("💆 Muscle Relaxation"):
                # Add typing effect for exercises
                with st.chat_message("assistant"):
                    typing_placeholder = display_typing_indicator()
                    time.sleep(1.5)
                    typing_placeholder.empty()
                    
                    exercise = techniques.progressive_muscle_relaxation()
                    exercise_container = st.empty()
                    stream_response_with_typing(exercise, exercise_container)
                    
                st.session_state.messages.append({"role": "assistant", "content": exercise})
                st.rerun()
            
            if st.button("🧘 Mindfulness Practice"):
                # Add typing effect for exercises
                with st.chat_message("assistant"):
                    typing_placeholder = display_typing_indicator()
                    time.sleep(1.5)
                    typing_placeholder.empty()
                    
                    exercise = techniques.mindfulness_practice()
                    exercise_container = st.empty()
                    stream_response_with_typing(exercise, exercise_container)
                    
                st.session_state.messages.append({"role": "assistant", "content": exercise})
                st.rerun()
        
        # Thought reframing and affirmations
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("🔄 Thought Reframing"):
                # Add typing effect for exercises
                with st.chat_message("assistant"):
                    typing_placeholder = display_typing_indicator()
                    time.sleep(1.5)
                    typing_placeholder.empty()
                    
                    exercise = techniques.thought_reframing()
                    exercise_container = st.empty()
                    stream_response_with_typing(exercise, exercise_container)
                    
                st.session_state.messages.append({"role": "assistant", "content": exercise})
                st.rerun()
        
        with col4:
            if st.button("💭 Daily Affirmation"):
                # Add typing effect for affirmations
                with st.chat_message("assistant"):
                    typing_placeholder = display_typing_indicator()
                    time.sleep(1)
                    typing_placeholder.empty()
                    
                    import random
                    affirmation = random.choice(techniques.positive_affirmations())
                    affirmation_msg = f"🌟 *Today's Affirmation:\n\n{affirmation}*\n\nTake a moment to really let this sink in. You deserve these kind words."
                    
                    affirmation_container = st.empty()
                    stream_response_with_typing(affirmation_msg, affirmation_container)
                    
                st.session_state.messages.append({"role": "assistant", "content": affirmation_msg})
                st.rerun()

        st.markdown("---")
        
        st.markdown("""
        ### About MindfulChat Pro
        
        *Enhanced Features:*
        - 🧠 Advanced sentiment analysis with emotional intensity detection
        - ❤ Evidence-based therapeutic responses tailored to your emotional state
        - 🔍 Comprehensive crisis detection with specialized support resources
        - 📊 Mood tracking and personalized insights over time
        - 🧘‍♀ Guided evidence-based therapeutic exercises and techniques
        - 💬 Scientifically-validated coping strategies for different emotional states
        - 🔄 Cognitive reframing tools to transform unhelpful thought patterns
        - 🔒 Safe, judgment-free space for emotional expression
        
        *Therapeutic Approaches:*
        - Cognitive Behavioral Therapy (CBT)
        - Mindfulness-Based Stress Reduction (MBSR)
        - Dialectical Behavior Therapy (DBT) skills
        - Acceptance and Commitment Therapy (ACT) principles
        - Solution-Focused Brief Therapy techniques
        - Positive Psychology interventions
        - Self-Compassion practices
        
        *Important:* This AI provides support but is not a replacement for professional mental health care. In crisis situations, please contact emergency services or a mental health professional.
        
        *Research-Based:* All techniques and strategies are based on peer-reviewed psychological research and clinical best practices.
        """)
        
        st.markdown("---")
        
        if st.button("🗑 Clear Chat History", key="clear_chat"):
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
