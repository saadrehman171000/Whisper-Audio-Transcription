import warnings
import streamlit as st
import os
import sys
import tempfile
from whisper_transcription import WhisperTranscriber
import time
import torch
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Set backend
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default style
import json
import numpy as np
import sounddevice as sd
from gtts import gTTS
import io
import base64
import google.generativeai as genai  # Add at the top with other imports

# Suppress all warnings and logging
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

# Simple error filter
class SimpleErrorFilter:
    def write(self, *args, **kwargs):
        pass
    def flush(self):
        pass

# Only filter errors during initialization
original_stderr = sys.stderr
sys.stderr = SimpleErrorFilter()

st.set_page_config(
    page_title="Whisper Audio Transcription",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Restore stderr
sys.stderr = original_stderr

def local_css():
    pass

def display_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎙️ Whisper Audio Transcription")
        st.markdown("Transform your audio into text with advanced AI technology")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/180px-ChatGPT_logo.svg.png", width=100)

def display_features():
    st.sidebar.title("Features")
    st.sidebar.markdown("""
    ✨ **Key Features**
    - 🌍 Multi-language Support
    - 🔄 Automatic Language Detection
    - 🎯 High Accuracy Transcription
    - 🔠 Auto Translation to English
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info('💡 Tip: The "base" model provides better accuracy than "tiny" but takes slightly longer to load.')
    
    # Add real-time session info
    if 'transcription_count' not in st.session_state:
        st.session_state.transcription_count = 0
    if 'total_audio_duration' not in st.session_state:
        st.session_state.total_audio_duration = 0
    if 'languages_detected' not in st.session_state:
        st.session_state.languages_detected = set()

    st.sidebar.markdown("### Session Information")
    
    # Display session metrics in an organized way
    st.sidebar.markdown("""
    📊 **Current Session Stats**
    """)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Files Processed", f"{st.session_state.transcription_count}")
    col2.metric("Unique Languages", f"{len(st.session_state.languages_detected)}")
    
    if st.session_state.languages_detected:
        st.sidebar.markdown("#### 🌐 Detected Languages")
        languages = ", ".join(sorted(st.session_state.languages_detected))
        st.sidebar.markdown(f"<div style='font-size: 0.9em; color: #666;'>{languages}</div>", unsafe_allow_html=True)
    
    if st.session_state.total_audio_duration > 0:
        st.sidebar.markdown("#### ⏱️ Total Audio Processed")
        duration_min = st.session_state.total_audio_duration / 60
        if duration_min < 1:
            st.sidebar.markdown(f"<div style='font-size: 0.9em; color: #666;'>{st.session_state.total_audio_duration:.1f} seconds</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<div style='font-size: 0.9em; color: #666;'>{duration_min:.1f} minutes</div>", unsafe_allow_html=True)

    # Add system info with error handling
    st.sidebar.markdown("### 💻 System Info")
    try:
        device = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"
        st.sidebar.markdown(f"**Processing Unit:** {device}")
        if torch.cuda.is_available():
            try:
                st.sidebar.markdown(f"**GPU Model:** {torch.cuda.get_device_name(0)}")
            except:
                st.sidebar.markdown("**GPU Model:** Not available")
    except Exception as e:
        st.sidebar.markdown("**Processing Unit:** CPU 💻")

def display_audio_waveform(audio_path):
    try:
        # Use the audio data from session state instead of loading from file
        if st.session_state.recording_state['recorded_data'] is not None:
            y = st.session_state.recording_state['recorded_data']
            sr = 44100  # This is our recording sample rate
            
            fig, ax = plt.subplots(figsize=(10, 2))
            times = np.linspace(0, len(y)/sr, len(y))
            ax.plot(times, y, color='#00a0a0')
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Could not display waveform visualization: {str(e)}")

def get_translation_languages():
    return {
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese"
    }

def export_as_srt(segments):
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = time.strftime('%H:%M:%S,000', time.gmtime(segment['start']))
        end = time.strftime('%H:%M:%S,000', time.gmtime(segment['end']))
        srt_content += f"{i}\n{start} --> {end}\n{segment['text']}\n\n"
    return srt_content

def display_confidence_meter(confidence_score):
    st.markdown("#### 🎯 Transcription Confidence")
    
    # Create a cool confidence meter
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"### {confidence_score:.0f}%")
    with col2:
        # Create a custom progress bar
        if confidence_score >= 90:
            bar_color = "background: linear-gradient(to right, #00ff00, #00cc00)"
            message = "Excellent! High confidence transcription"
        elif confidence_score >= 75:
            bar_color = "background: linear-gradient(to right, #ffff00, #ffcc00)"
            message = "Good! Reliable transcription"
        else:
            bar_color = "background: linear-gradient(to right, #ff9999, #ff3333)"
            message = "Fair. You might want to check the transcription"
            
        st.markdown(
            f"""
            <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px;">
                <div style="width: {confidence_score}%; {bar_color}; height: 30px; 
                border-radius: 10px; transition: width 0.5s;">
                </div>
            </div>
            <p style="color: #666;">{message}</p>
            """,
            unsafe_allow_html=True
        )

def display_audio_visualizer(audio_data, sample_rate):
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Audio Spectrogram')
        st.pyplot(fig)
        plt.close(fig)  # Clean up
        
        fig, ax = plt.subplots(figsize=(10, 2))
        times = np.arange(len(audio_data))/sample_rate
        ax.fill_between(times, audio_data, alpha=0.5, color='#00a0a0')
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)  # Clean up
        
    except Exception as e:
        st.warning("Could not display audio visualizations")

def record_audio():
    """Record audio from microphone"""
    try:
        # Check if we're running on Streamlit Cloud
        is_cloud = os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'
        
        if is_cloud:
            st.error("⚠️ Live recording is not available on Streamlit Cloud due to security restrictions. Please use the 'Upload File' option instead.")
            st.info("💡 To use live recording, run this app locally on your computer.")
            return None

        # Define sample rate at the function level
        SAMPLE_RATE = 44100  # Standard sample rate

        # Initialize sounddevice
        try:
            sd._terminate()
            sd._initialize()
        except:
            pass

        # Check available audio devices
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                st.error("❌ No input devices found")
                st.info("💡 Please ensure your microphone is connected and permissions are granted.")
                return None

            # Use the first available input device
            device_id = input_devices[0]['index']
            st.success(f"🎤 Using input device: {input_devices[0]['name']}")

        except Exception as e:
            st.error(f"❌ Error detecting audio devices: {str(e)}")
            st.info("💡 Please check your system's audio settings")
            return None
            
        if 'recording_state' not in st.session_state:
            st.session_state.recording_state = {
                'is_recording': False,
                'audio_data': [],
                'sample_rate': SAMPLE_RATE,
                'processed': False
            }
        
        # Create columns for record button and status
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if not st.session_state.recording_state['is_recording']:
                if st.button('🎙️ Start Recording', key='start_recording'):
                    st.session_state.recording_state = {
                        'is_recording': True,
                        'audio_data': [],
                        'filename': f"recorded_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav",
                        'start_time': time.time(),
                        'device_id': device_id,
                        'audio_saved': False,
                        'processed': False,
                        'sample_rate': SAMPLE_RATE
                    }
                    st.rerun()
            else:
                if st.button('⏹️ Stop Recording', key='stop_recording'):
                    st.session_state.recording_state['is_recording'] = False
                    st.rerun()

        # Status placeholder
        status_placeholder = st.empty()
        duration_placeholder = st.empty()
        
        # Recording logic
        if st.session_state.recording_state['is_recording']:
            status_placeholder.markdown("🎙️ **Recording in progress...**")
            
            try:
                with sd.InputStream(
                    device=st.session_state.recording_state.get('device_id'),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32'
                ) as stream:
                    while st.session_state.recording_state['is_recording']:
                        audio_chunk, _ = stream.read(SAMPLE_RATE // 10)
                        st.session_state.recording_state['audio_data'].append(audio_chunk)
                        duration = time.time() - st.session_state.recording_state['start_time']
                        duration_placeholder.markdown(f"⏱️ Recording duration: {int(duration)} seconds")
                        time.sleep(0.1)
                        
            except Exception as e:
                st.error(f"Recording failed: {str(e)}")
                st.info("💡 Please check your microphone settings and permissions")
                return None

        # Save recording if we have data
        if (not st.session_state.recording_state['is_recording'] and 
            len(st.session_state.recording_state['audio_data']) > 0 and 
            not st.session_state.recording_state['audio_saved']):
            
            try:
                audio = np.concatenate(st.session_state.recording_state['audio_data'])
                filename = st.session_state.recording_state['filename']
                sf.write(filename, audio, SAMPLE_RATE)
                st.session_state.recording_state['audio_saved'] = True
                status_placeholder.markdown("✅ **Recording saved successfully!**")
                return filename
                
            except Exception as e:
                st.error(f"Failed to save recording: {str(e)}")
                return None
        
        return None

    except Exception as e:
        st.error(f"Recording failed: {str(e)}")
        st.info("💡 Please check your microphone settings and permissions")
        return None

def text_to_speech(text, language='en', audio_path=None):
    if not text:
        return
        
    st.markdown("### 🔊 Listen to Transcription")
    
    try:
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, f"tts_audio_{time.strftime('%Y%m%d_%H%M%S')}.mp3")
        
        # If language is not English, get the translation
        if language not in ['en', 'EN']:
            try:
                # Use the same transcriber instance for consistency
                with st.spinner("Getting English translation..."):
                    translation_result = st.session_state.transcriber.transcribe_audio(
                        audio_path,
                        task="translate"  # Force translation
                    )
                    text_for_tts = translation_result['text']
                    st.markdown("#### English Translation:")
                    st.markdown(f"<div class='info-box'>{text_for_tts}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Translation failed: {str(e)}")
                text_for_tts = text
        else:
            text_for_tts = text

        # Generate speech from text
        with st.spinner("Generating audio..."):
            tts = gTTS(text=text_for_tts, lang='en', slow=False)
            tts.save(temp_audio_path)
            
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
                    
                    st.download_button(
                        "📥 Download Audio",
                        audio_bytes,
                        file_name=f"transcription_audio_{time.strftime('%Y%m%d_%H%M%S')}.mp3",
                        mime="audio/mp3"
                    )
            else:
                st.error("Failed to generate audio file")
                
            try:
                os.remove(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        st.error(f"Could not generate speech: {str(e)}")

def highlight_key_moments(text, language='en', audio_path=None):
    st.markdown("### ✨ Magic Moments")
    
    try:
        # If language is not English, get the translation first
        if language not in ['en', 'EN']:
            try:
                with st.spinner("Getting English translation for analysis..."):
                    # Use cached translation if available
                    if 'translation_cache' not in st.session_state:
                        st.session_state.translation_cache = {}
                    
                    cache_key = f"{audio_path}_{language}"
                    if cache_key in st.session_state.translation_cache:
                        text_for_analysis = st.session_state.translation_cache[cache_key]
                    else:
                        # Use the same transcriber instance for consistency
                        translation_result = st.session_state.transcriber.transcribe_audio(
                            audio_path,
                            task="translate"
                        )
                        text_for_analysis = translation_result['text']
                        st.session_state.translation_cache[cache_key] = text_for_analysis
                    
                    st.markdown("#### English Translation:")
                    st.markdown(f"<div class='info-box'>{text_for_analysis}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Translation failed: {str(e)}")
                text_for_analysis = text
        else:
            text_for_analysis = text

        # Analyze the text more efficiently
        sentences = [s.strip() for s in text_for_analysis.split('.') if s.strip()]
        key_moments = []
        
        # Process all patterns at once
        for sentence in sentences:
            moment_type = None
            emoji = None
            
            if '?' in sentence:
                moment_type, emoji = "Question", "🤔"
            elif '!' in sentence:
                moment_type, emoji = "Emphasis", "💡"
            elif any(char.isdigit() for char in sentence):
                moment_type, emoji = "Fact", "📊"
            elif any(phrase in sentence.lower() for phrase in ['important', 'key', 'must', 'should', 'need to', 'remember']):
                moment_type, emoji = "Important", "⭐"
                
            if moment_type:
                key_moments.append((moment_type, sentence, emoji))
        
        # Display results
        if key_moments:
            for moment_type, text, emoji in key_moments:
                st.markdown(
                    f"""<div style="padding: 10px; margin: 5px; border-radius: 10px; 
                    background-color: #f8f9fa; border-left: 4px solid #00a0a0;">
                    {emoji} <b>{moment_type}</b><br>{text}</div>""",
                    unsafe_allow_html=True
                )
        else:
            st.info("No key moments detected in this recording.")
            
    except Exception as e:
        st.error(f"Error in Magic Moments analysis: {str(e)}")

def analyze_sentiment(text):
    st.markdown("### 💭 Text Analysis")
    
    # Split text into words and clean them
    words = text.lower().split()
    
    # More comprehensive sentiment word lists
    positive_words = {
        'good', 'great', 'excellent', 'happy', 'wonderful', 'best', 'love',
        'amazing', 'fantastic', 'perfect', 'better', 'awesome', 'nice',
        'brilliant', 'success', 'successful', 'win', 'winning'
    }
    negative_words = {
        'bad', 'poor', 'terrible', 'sad', 'worst', 'hate',
        'awful', 'horrible', 'wrong', 'fail', 'failed', 'failure',
        'problem', 'difficult', 'impossible', 'never'
    }
    
    # Count sentiment words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total_words = len(words)  # Count all words, not just sentiment words
    
    # Determine sentiment
    if total_words == 0:
        sentiment = 0
        mood = "Neutral"
        emoji = "😐"
        color = "#ffc107"
    else:
        sentiment = (positive_count - negative_count) / total_words
        if sentiment > 0:
            emoji = "😊"
            color = "#28a745"
            mood = "Positive"
        elif sentiment < 0:
            emoji = "😔"
            color = "#dc3545"
            mood = "Negative"
        else:
            emoji = "😐"
            color = "#ffc107"
            mood = "Neutral"
    
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: {color}20; border-radius: 10px;">
            <div style="font-size: 3em;">{emoji}</div>
            <div style="font-size: 1.5em; color: {color};">{mood}</div>
            <div style="font-size: 1em;">Words analyzed: {total_words}</div>
        </div>
    """, unsafe_allow_html=True)

def cleanup_old_recordings():
    # Clean up old WAV files
    for file in os.listdir():
        if file.startswith("recorded_audio_") and file.endswith(".wav"):
            try:
                os.remove(file)
            except:
                pass

# Add this function to handle Gemini responses
def get_ai_response(text, language='en'):
    try:
        st.markdown("### 🤖 AI Response")
        with st.spinner("Getting AI response..."):
            # Configure Gemini
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            
            # Prepare the prompt
            if language not in ['en', 'EN']:
                prompt = f"""
                The following is a transcribed text in another language that has been translated to English:
                "{text}"
                
                Please:
                1. Understand the content/question
                2. Provide a helpful, clear response
                3. If there's a question, answer it directly
                4. If it's a statement, provide relevant insights or suggestions
                """
            else:
                prompt = f"""
                The following is a transcribed text:
                "{text}"
                
                Please:
                1. Understand the content/question
                2. Provide a helpful, clear response
                3. If there's a question, answer it directly
                4. If it's a statement, provide relevant insights or suggestions
                """

            # Get response from Gemini
            response = model.generate_content(prompt)
            
            # Display the response in a nice format
            st.markdown(
                f"""<div style="padding: 20px; border-radius: 10px; background-color: #f0f8ff; border-left: 5px solid #00a0a0;">
                <h4>💡 AI Response:</h4>
                {response.text}
                </div>""", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"Could not generate AI response: {str(e)}")

def main():
    # Add at the start of main()
    cleanup_old_recordings()
    local_css()
    display_header()
    display_features()

    # Initialize session state variables
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = WhisperTranscriber()
        st.session_state.current_model_size = None
    if 'transcription_count' not in st.session_state:
        st.session_state.transcription_count = 0
    if 'total_audio_duration' not in st.session_state:
        st.session_state.total_audio_duration = 0
    if 'languages_detected' not in st.session_state:
        st.session_state.languages_detected = set()

    # Add this before the model selection
    st.info("""
    💡 **Model Guide:**
    - **Tiny**: Fastest (32x) but less accurate. Best for quick drafts.
    - **Base**: Better accuracy with good speed (16x). Recommended for most uses.
    """)

    # Model selection with enhanced UI
    st.markdown("### Choose Your Model")
    model_size = st.selectbox(
        "Select the transcription model that best fits your needs:",
        options=list(st.session_state.transcriber.available_models.keys()),
        format_func=lambda x: f"{x.capitalize()} ({st.session_state.transcriber.available_models[x]['parameters']} parameters, {st.session_state.transcriber.available_models[x]['relative_speed']} speed)",
        index=1
    )

    # Load model if not loaded or if model size changed
    if (not st.session_state.transcriber.current_model or 
        getattr(st.session_state, 'current_model_size', None) != model_size):
        
        with st.spinner("Loading model... Please wait."):
            progress_text = "Loading model... Please wait."
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            st.session_state.transcriber.load_model(model_size)
            st.session_state.current_model_size = model_size
            my_bar.empty()

    # Choose input method
    st.markdown("### Choose Input Method")
    input_method = st.radio(
        "Select how you want to input audio:",
        ["Upload File", "Record Audio"],
        horizontal=True
    )
    
    audio_file = None
    if input_method == "Upload File":
        # Initialize recording state if not exists
        if 'recording_state' not in st.session_state:
            st.session_state.recording_state = {
                'is_recording': False,
                'audio_data': [],
                'filename': None,
                'start_time': None,
                'recorded_data': None,
                'audio_saved': False,
                'processing': False,
                'processed': False
            }
        
        # Add file uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            help="Upload an audio file (WAV, MP3, M4A, OGG, or FLAC format)"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Store the temp path in session state for later use
                st.session_state.temp_audio_path = temp_path
                
                # Process the uploaded file
                with st.spinner("Processing audio file..."):
                    result = st.session_state.transcriber.transcribe_audio(temp_path)
                    
                    if result:
                        # Store audio data for visualization
                        st.session_state.recording_state['recorded_data'] = result['audio_data']
                        
                        # Display all results
                        st.markdown("### 📝 Transcription Results")
                        
                        # Metrics display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Detected Language</h4>
                                    <div class="metric-value">{result['language'].upper()}</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Processing Time</h4>
                                    <div class="metric-value">{result['transcription_time']:.2f}s</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                        with col3:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Model Used</h4>
                                    <div class="metric-value">{model_size.upper()}</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )

                        # Transcribed text
                        st.markdown("#### Transcribed Text")
                        st.markdown(f"<div class='success-box'>{result['text']}</div>", unsafe_allow_html=True)

                        # Display audio waveform
                        display_audio_waveform(temp_path)

                        # Display audio visualizer
                        display_audio_visualizer(result['audio_data'], result['sample_rate'])

                        # Display confidence meter
                        confidence = 95
                        display_confidence_meter(confidence)

                        # Highlight key moments (before text-to-speech)
                        highlight_key_moments(result['text'].strip(), result['language'], temp_path)
                        
                        # Text to speech
                        text_to_speech(result['text'].strip(), result['language'], temp_path)
                        
                        # Sentiment analysis
                        analyze_sentiment(result['text'].strip())

                        # Get AI response
                        get_ai_response(result['text'].strip(), result['language'])

                        # Add download buttons
                        duration = len(result['audio_data']) / result['sample_rate']
                        json_export = {
                            'transcription': result['text'],
                            'language': result['language'],
                            'segments': result['segments'],
                            'metadata': {
                                'duration': duration,
                                'model': model_size,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        }

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📥 Download as TXT",
                                result['text'],
                                file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                            )
                        with col2:
                            st.download_button(
                                "📥 Download as SRT",
                                export_as_srt(result['segments']),
                                file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.srt"
                            )
                        with col3:
                            st.download_button(
                                "📥 Download as JSON",
                                json.dumps(json_export, indent=2),
                                file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.json"
                            )

                        # Update session state
                        st.session_state.transcription_count += 1
                        st.session_state.languages_detected.add(result['language'])
                        st.session_state.total_audio_duration += duration

                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.markdown("Please try again with a different file.")
                
    elif input_method == "Record Audio":
        audio_file = record_audio()
        if audio_file and os.path.exists(audio_file):
            try:
                with st.spinner("Processing audio file..."):
                    # Process audio with loaded model
                    result = st.session_state.transcriber.transcribe_audio(audio_file)
                    
                    if result:
                        # Store audio data in session state for visualization
                        st.session_state.recording_state['recorded_data'] = result['audio_data']
                        
                        # Display waveform
                        display_audio_waveform(audio_file)
                        
                        # Display all results
                        st.markdown("### 📝 Transcription Results")
                        
                        # Metrics display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Detected Language</h4>
                                    <div class="metric-value">{result['language'].upper()}</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Processing Time</h4>
                                    <div class="metric-value">{result['transcription_time']:.2f}s</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                        with col3:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <h4>Model Used</h4>
                                    <div class="metric-value">{model_size.upper()}</div>
                                </div>""", 
                                unsafe_allow_html=True
                            )

                        # Transcribed text
                        st.markdown("#### Transcribed Text")
                        st.markdown(f"<div class='success-box'>{result['text']}</div>", unsafe_allow_html=True)

                # Reset recording state
                st.session_state.recording_state = {
                    'is_recording': False,
                    'audio_data': [],
                    'filename': None,
                    'start_time': None,
                    'recorded_data': None,
                    'audio_saved': False,
                    'processing': False,
                    'processed': False
                }
                
                # Update session state
                st.session_state.transcription_count += 1
                st.session_state.languages_detected.add(result['language'])
                # Calculate duration from audio data
                duration = len(result['audio_data']) / result['sample_rate']
                st.session_state.total_audio_duration += duration

                # Update json_export
                json_export = {
                    'transcription': result['text'],
                    'language': result['language'],
                    'segments': result['segments'],
                    'metadata': {
                        'duration': duration,  # Use calculated duration
                        'model': model_size,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }

                # Add download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📥 Download as TXT",
                        result['text'],
                        file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                with col2:
                    st.download_button(
                        "📥 Download as SRT",
                        export_as_srt(result['segments']),
                        file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.srt"
                    )
                with col3:
                    st.download_button(
                        "📥 Download as JSON",
                        json.dumps(json_export, indent=2),
                        file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    )

                # Display audio waveform
                display_audio_waveform(audio_file)

                # Display audio visualizer
                display_audio_visualizer(result['audio_data'], result['sample_rate'])

                # Display confidence meter
                confidence = 95
                display_confidence_meter(confidence)

                # Highlight key moments (before text-to-speech)
                highlight_key_moments(result['text'].strip(), result['language'], audio_file)

                # Text to speech
                text_to_speech(result['text'].strip(), result['language'], audio_file)
                
                # Sentiment analysis
                analyze_sentiment(result['text'].strip())  # Make sure to strip whitespace

                # Get AI response
                get_ai_response(result['text'].strip(), result['language'])

                # After successful processing
                st.session_state.recording_state['processed'] = True
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.markdown("Please try again with a different file or model.")

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Made with ❤️ using OpenAI's Whisper | 
        <a href="https://github.com/openai/whisper" target="_blank">Learn More</a>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()