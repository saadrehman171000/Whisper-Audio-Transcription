# Import required libraries
import scipy.signal  # For audio signal processing (resampling)
import whisper      # OpenAI's Whisper model for speech recognition
import time        # For timing operations and delays
import os          # For file and path operations
from pathlib import Path  # For cross-platform path handling
import torch       # Deep learning framework
import numpy as np  # For numerical operations
import soundfile as sf  # For handling audio files
import librosa       # For audio processing

class WhisperTranscriber:
    """A class to handle speech transcription using OpenAI's Whisper model."""
    
    def __init__(self):
        # Dictionary of available models with their parameters and relative speeds
        self.available_models = {
            "tiny": {"parameters": "39M", "relative_speed": "32x"},
            "base": {"parameters": "74M", "relative_speed": "16x"},
            "small": {"parameters": "244M", "relative_speed": "8x"},
            "medium": {"parameters": "769M", "relative_speed": "4x"}
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
        # Load model with multilingual support
        self.current_model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - start_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        return self.current_model

    def transcribe_audio(self, audio_path, task="transcribe"):
        try:
            # Load and preprocess audio
            audio_data, sample_rate = self._load_audio(audio_path)
            start_time = time.time()
            
            # First detect language
            print("Detecting language...")
            detect_result = self.current_model.transcribe(
                audio_data,
                task="detect_language",
                fp16=False
            )
            detected_language = detect_result["language"]
            print(f"Detected language: {detected_language}")
            
            # Perform transcription or translation
            try:
                # Important: Use the same settings for both transcription and translation
                result = self.current_model.transcribe(
                    audio_data,
                    language=detected_language,
                    task=task,  # This will be either "transcribe" or "translate"
                    fp16=False,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    best_of=5,
                    beam_size=5,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    initial_prompt=None  # Remove any previous prompts
                )
                
                transcription_time = time.time() - start_time
                
                # Add word timing info
                words_with_time = []
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word in segment["words"]:
                                words_with_time.append({
                                    "word": word.get("text", "").strip(),
                                    "start": word.get("start", 0),
                                    "end": word.get("end", 0),
                                    "confidence": word.get("confidence", 0)
                                })
                
                # Return results
                return {
                    "text": result.get("text", "").strip(),
                    "language": detected_language,
                    "segments": result.get("segments", []),
                    "transcription_time": transcription_time,
                    "audio_data": audio_data,
                    "sample_rate": sample_rate,
                    "words_with_time": words_with_time,
                    "confidence": result.get("confidence", 0)
                }
                
            except Exception as e:
                print(f"Transcription failed: {str(e)}")
                raise ValueError(f"Transcription failed: {str(e)}")
            
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            raise

    def _load_audio(self, audio_path):
        try:
            # Use soundfile instead of librosa for loading
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to float32 and mono if stereo
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16000 Hz if needed
            if sample_rate != 16000:
                # Calculate number of samples for target length
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = scipy.signal.resample(audio_data, target_length).astype(np.float32)
                sample_rate = 16000
            
            # Normalize audio (ensure float32)
            max_value = np.max(np.abs(audio_data))
            if max_value > 0:
                audio_data = (audio_data / max_value).astype(np.float32)
            
            return audio_data, sample_rate
        except Exception as e:
            print(f"Audio loading error details: {str(e)}")
            print(f"Audio data type: {type(audio_data) if 'audio_data' in locals() else 'Not loaded'}")
            if 'audio_data' in locals():
                print(f"Audio data shape: {audio_data.shape}")
                print(f"Audio data dtype: {audio_data.dtype}")
            raise ValueError(f"Failed to load audio file: {str(e)}")
