import streamlit as st
import os
from pathlib import Path
from whisper_transcription import WhisperTranscriber, record_audio
import tempfile

def main():
    st.title("🎙️ Whisper Audio Transcription")
    st.write("Transcribe audio using OpenAI's Whisper model")

    # Initialize the transcriber
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = WhisperTranscriber()

    # Model selection
    model_size = st.selectbox(
        "Select Whisper Model",
        options=list(st.session_state.transcriber.available_models.keys()),
        format_func=lambda x: f"{x.capitalize()} ({st.session_state.transcriber.available_models[x]['parameters']} parameters, {st.session_state.transcriber.available_models[x]['relative_speed']} speed)",
        index=1  # Default to "base" model
    )

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
        
        if uploaded_file:
            with st.spinner("Processing audio file..."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name

            process_audio(audio_path, model_size)
            # Clean up temporary file
            os.unlink(audio_path)

    with tab2:
        st.header("Record Audio")
        duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
        
        if st.button("Start Recording"):
            with st.spinner(f"Recording for {duration} seconds..."):
                # Create audio directory if it doesn't exist
                audio_directory = "audio_samples"
                Path(audio_directory).mkdir(parents=True, exist_ok=True)
                audio_path = os.path.join(audio_directory, "recording.wav")
                
                # Record audio
                record_audio(audio_path, duration=duration)
                st.success("Recording completed!")
                
                process_audio(audio_path, model_size)

def process_audio(audio_path, model_size):
    """Process the audio file and display results"""
    try:
        # Load model if not already loaded or if model size changed
        if (not st.session_state.transcriber.current_model or 
            getattr(st.session_state, 'current_model_size', None) != model_size):
            with st.spinner(f"Loading {model_size} model..."):
                st.session_state.transcriber.load_model(model_size)
                st.session_state.current_model_size = model_size

        # Transcribe audio
        with st.spinner("Transcribing audio..."):
            result = st.session_state.transcriber.transcribe_audio(audio_path)

        # Display results
        st.subheader("Transcription Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Detected Language:**")
            st.write(result['language'])
        with col2:
            st.write("**Processing Time:**")
            st.write(f"{result['transcription_time']:.2f} seconds")

        st.write("**Transcribed Text:**")
        st.write(result['text'])

        # Handle translation for non-English audio
        if result['language'] != 'en':
            with st.spinner("Translating to English..."):
                translation = st.session_state.transcriber.current_model.transcribe(
                    audio_path,
                    task="translate",
                    fp16=False,
                    verbose=None
                )
                st.write("**English Translation:**")
                st.write(translation["text"])

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main() 