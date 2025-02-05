import streamlit as st
import os
import tempfile
from whisper_transcription import WhisperTranscriber
import time
import torch
import soundfile as sf

def set_page_config():
    st.set_page_config(
        page_title="Whisper Audio Transcription",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def local_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem 3rem;
        }
        .stProgress > div > div > div > div {
            background-color: #00a0a0;
        }
        .stButton>button {
            background-color: #00a0a0;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #008080;
        }
        .status-box {
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #d1f5d3;
            border: 1px solid #7eca9c;
        }
        .info-box {
            background-color: #e1f5fe;
            border: 1px solid #4fc3f7;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00a0a0;
        }
        </style>
    """, unsafe_allow_html=True)

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

def main():
    set_page_config()
    local_css()
    display_header()
    display_features()

    # Initialize the transcriber
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = WhisperTranscriber()

    # Model selection with enhanced UI
    st.markdown("### Choose Your Model")
    model_size = st.selectbox(
        "Select the transcription model that best fits your needs:",
        options=list(st.session_state.transcriber.available_models.keys()),
        format_func=lambda x: f"{x.capitalize()} ({st.session_state.transcriber.available_models[x]['parameters']} parameters, {st.session_state.transcriber.available_models[x]['relative_speed']} speed)",
        index=1
    )

    # File upload section with instructions
    st.markdown("### Upload Your Audio")
    st.markdown("Supported formats: MP3, WAV, M4A")
    
    uploaded_file = st.file_uploader(
        label="Choose an audio file",
        type=['mp3', 'wav', 'm4a'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "Upload time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create an expandable section for file details
            with st.expander("📁 File Details", expanded=True):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
                
                # Add processing tips
                st.info("""
                💡 **Processing Tips:**
                - Larger files may take longer to process
                - For best results, ensure clear audio quality
                - Supported languages: 90+ languages
                """)

            with st.spinner("Processing audio file..."):
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name

                # Load model with progress tracking
                if (not st.session_state.transcriber.current_model or 
                    getattr(st.session_state, 'current_model_size', None) != model_size):
                    
                    progress_text = "Loading model... Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1, text=progress_text)
                    
                    st.session_state.transcriber.load_model(model_size)
                    st.session_state.current_model_size = model_size
                    my_bar.empty()

                # Transcribe with progress indicator
                with st.spinner("✨ Magic happening... Transcribing your audio"):
                    result = st.session_state.transcriber.transcribe_audio(audio_path)

                # Display results in a clean format
                st.markdown("---")
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

                # Translation for non-English audio
                if result['language'] != 'en':
                    with st.spinner("🌐 Translating to English..."):
                        translation = st.session_state.transcriber.current_model.transcribe(
                            audio_path,
                            task="translate",
                            fp16=False,
                            verbose=None
                        )
                        st.markdown("#### English Translation")
                        st.markdown(f"<div class='info-box'>{translation['text']}</div>", unsafe_allow_html=True)

            # Clean up
            os.unlink(audio_path)

            # Update session state
            st.session_state.transcription_count += 1
            st.session_state.languages_detected.add(result['language'])
            
            # Calculate audio duration from the audio file
            try:
                audio_data, sample_rate = sf.read(audio_path)
                audio_duration = len(audio_data) / sample_rate
                st.session_state.total_audio_duration += audio_duration
            except Exception as e:
                st.warning("Could not calculate audio duration")

            # Add download buttons
            col1, col2 = st.columns(2)
            with col1:
                # Create text file with transcription
                transcript_text = f"""Transcription Results
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
File: {uploaded_file.name}
Model: {model_size}
Language: {result['language']}
Duration: {audio_duration:.2f} seconds
---
{result['text']}
"""
                st.download_button(
                    label="📥 Download Transcription",
                    data=transcript_text,
                    file_name=f"transcription_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            if result['language'] != 'en':
                with col2:
                    # Create text file with translation
                    translation_text = f"""English Translation
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
Original Language: {result['language']}
---
{translation['text']}
"""
                    st.download_button(
                        label="📥 Download Translation",
                        data=translation_text,
                        file_name=f"translation_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

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