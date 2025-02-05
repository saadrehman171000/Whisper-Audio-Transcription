import streamlit as st
import os
import tempfile
from whisper_transcription import WhisperTranscriber

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

    # File uploader
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
    
    if uploaded_file:
        try:
            with st.spinner("Processing audio file..."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name

                # Load model if needed
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

            # Clean up temporary file
            os.unlink(audio_path)

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main() 