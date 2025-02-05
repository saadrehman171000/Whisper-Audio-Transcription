# Import required libraries
import scipy.signal  # For audio signal processing (resampling)
import whisper      # OpenAI's Whisper model for speech recognition
import time        # For timing operations and delays
import os          # For file and path operations
from pathlib import Path  # For cross-platform path handling
import torch       # Deep learning framework
import numpy as np  # For numerical operations
import soundfile as sf  # For handling audio files

class WhisperTranscriber:
    """A class to handle speech transcription using OpenAI's Whisper model."""
    
    def __init__(self):
        # Dictionary of available models with their parameters and relative speeds
        self.available_models = {
            "tiny": {"parameters": "39M", "relative_speed": "32x"},
            "base": {"parameters": "74M", "relative_speed": "16x"}
        }
        self.current_model = None  # Will hold the loaded model
        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_size="base"):
        """Load a Whisper model of specified size."""
        if model_size not in self.available_models:
            raise ValueError(f"Model size must be one of {list(self.available_models.keys())}")
        
        print(f"\nLoading {model_size} model...")
        print(f"Model details: {self.available_models[model_size]}")
        print(f"Using device: {self.device}")
        
        start_time = time.time()
        self.current_model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - start_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        return self.current_model

    def transcribe_audio(self, audio_path, language=None):
        """Transcribe audio file and return results with metrics."""
        if not self.current_model:
            raise ValueError("No model loaded. Please load a model first.")
        
        audio_path = os.path.abspath(os.path.normpath(audio_path))
        print(f"\nChecking if the file exists at {audio_path}")
        
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load and preprocess audio
            print(f"Loading audio file...")
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                print("Converting stereo to mono...")
                audio_data = audio_data.mean(axis=1)
            
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz")
                new_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = scipy.signal.resample(audio_data, new_length)
                sample_rate = 16000
            
            # Ensure minimum duration (at least 0.1 seconds)
            min_samples = int(0.1 * sample_rate)
            if len(audio_data) < min_samples:
                print("Audio too short, padding with silence...")
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)))
            
            # Ensure maximum duration (trim if too long)
            max_samples = int(30 * 60 * sample_rate)  # 30 minutes max
            if len(audio_data) > max_samples:
                print("Audio too long, trimming...")
                audio_data = audio_data[:max_samples]
            
            print(f"Successfully loaded audio file")
            print(f"Audio duration: {len(audio_data)/sample_rate:.2f} seconds")
            print(f"Audio shape: {audio_data.shape}")
            print(f"Audio dtype: {audio_data.dtype}")
            print(f"Audio range: [{audio_data.min():.2f}, {audio_data.max():.2f}]")
            
            start_time = time.time()
            
            # Detect language if not specified
            if not language:
                print("Detecting language...")
                try:
                    result = self.current_model.transcribe(
                        audio_data,
                        task="detect_language",
                        fp16=False
                    )
                    language = result["language"]
                    print(f"Detected language: {language}")
                except Exception as e:
                    print(f"Language detection failed: {e}")
                    language = None  # Fall back to auto-detection in transcription
            
            # Perform transcription
            try:
                result = self.current_model.transcribe(
                    audio_data,
                    language=language,
                    task="transcribe",
                    fp16=False,
                    verbose=None
                )
                
                transcription_time = time.time() - start_time
                
                return {
                    "text": result["text"],
                    "language": result["language"],
                    "segments": result["segments"],
                    "transcription_time": transcription_time,
                    "audio_data": audio_data,  # Include processed audio data
                    "sample_rate": sample_rate
                }
            except Exception as e:
                print(f"Transcription failed: {str(e)}")
                print(f"Audio shape: {audio_data.shape}")
                print(f"Sample rate: {sample_rate}")
                raise ValueError(f"Transcription failed: Please ensure the audio file is valid and try again. Error: {str(e)}")
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            print(f"Full error details: {repr(e)}")
            raise
