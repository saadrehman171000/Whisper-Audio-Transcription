# Import required libraries
import scipy.signal  # For audio signal processing (resampling)
import whisper      # OpenAI's Whisper model for speech recognition
import time        # For timing operations and delays
import os          # For file and path operations
from pathlib import Path  # For cross-platform path handling
import torch       # Deep learning framework
import sounddevice as sd  # For recording audio from microphone
import scipy.io.wavfile as wav  # For saving WAV files
import numpy as np  # For numerical operations
import soundfile as sf  # For handling audio files

class WhisperTranscriber:
    """A class to handle speech transcription using OpenAI's Whisper model."""
    
    def __init__(self):
        # Dictionary of available models with their parameters and relative speeds
        self.available_models = {
            "tiny": {"parameters": "39M", "relative_speed": "32x"},
            "base": {"parameters": "74M", "relative_speed": "16x"},
            "small": {"parameters": "244M", "relative_speed": "6x"},
            "medium": {"parameters": "769M", "relative_speed": "2x"},
            "large": {"parameters": "1550M", "relative_speed": "1x"}
        }
        self.current_model = None  # Will hold the loaded model
        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_size="base"):
        """Load a Whisper model of specified size.
        
        Args:
            model_size (str): Size of the model to load (tiny, base, small, medium, large)
        Returns:
            The loaded model
        """
        # Validate model size
        if model_size not in self.available_models:
            raise ValueError(f"Model size must be one of {list(self.available_models.keys())}")
        
        print(f"\nLoading {model_size} model...")
        print(f"Model details: {self.available_models[model_size]}")
        print(f"Using device: {self.device}")
        
        # Time the model loading process
        start_time = time.time()
        self.current_model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - start_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        return self.current_model

    def transcribe_audio(self, audio_path, language=None):
        """Transcribe audio file and return results with metrics.
        
        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code for transcription
        Returns:
            dict: Contains transcription text, language, segments, and timing
        """
        # Check if model is loaded
        if not self.current_model:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Convert to absolute path and normalize
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
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz")
                new_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = scipy.signal.resample(audio_data, new_length)
            
            # Normalize audio to float32 between -1 and 1
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Print audio information
            print(f"Successfully loaded audio file")
            print(f"Audio duration: {len(audio_data)/16000:.2f} seconds")
            print(f"Audio shape: {audio_data.shape}")
            print(f"Audio dtype: {audio_data.dtype}")
            print(f"Audio range: [{audio_data.min():.2f}, {audio_data.max():.2f}]")
            
            print(f"\nTranscribing audio data...")
            start_time = time.time()
            
            # Detect language if not specified
            if not language:
                print("Detecting language...")
                result = self.current_model.transcribe(
                    audio_data,
                    task="detect_language",
                    fp16=False  # Avoid FP16 for better compatibility
                )
                language = result["language"]
                print(f"Detected language: {language}")
            
            # Perform transcription
            result = self.current_model.transcribe(
                audio_data,
                language=language,
                task="transcribe",
                fp16=False,
                verbose=None,  # Disable default progress bar
                word_timestamps=True  # Enable word-level timestamps
            )
            
            transcription_time = time.time() - start_time
            
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"],
                "transcription_time": transcription_time
            }
        except Exception as e:
            # Detailed error reporting
            print(f"Error processing audio: {str(e)}")
            print(f"Full error details: {repr(e)}")
            print(f"File exists: {os.path.exists(audio_path)}")
            print(f"File is readable: {os.access(audio_path, os.R_OK)}")
            print(f"File size: {os.path.getsize(audio_path)} bytes")
            raise

def record_audio(filename, duration=5, sample_rate=16000):
    """Record audio from microphone and save to file.
    
    Args:
        filename (str): Path to save the recorded audio
        duration (float): Recording duration in seconds
        sample_rate (int): Sampling rate in Hz
    """
    # Countdown before recording
    print(f"\nRecording will start in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("\nRecording now! Speak into your microphone...")
    
    # Record audio using sounddevice
    recording = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1)
    
    # Show recording progress
    for i in range(int(duration)):
        print(f"Recording: {i+1}/{int(duration)} seconds")
        time.sleep(1)
    sd.wait()
    
    print("\nRecording finished!")
    
    # Process and save the recording
    recording = np.squeeze(recording)  # Remove extra dimensions
    recording = recording / np.max(np.abs(recording))  # Normalize
    wav.write(filename, sample_rate, (recording * 32767).astype(np.int16))
    
    print(f"Audio saved to: {filename}")

def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV format using soundfile.
    
    Args:
        mp3_path (str): Path to the MP3 file
    Returns:
        str: Path to the converted WAV file
    """
    print(f"Converting {mp3_path} to WAV format...")
    
    # Read MP3 and save as WAV
    audio_data, samplerate = sf.read(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    
    sf.write(wav_path, audio_data, samplerate)
    
    print(f"Conversion complete: {wav_path}")
    return wav_path

def main():
    """Main function to run the transcription program."""
    transcriber = WhisperTranscriber()
    
    while True:
        # Display menu
        print("\nOptions:")
        print("1. Record and transcribe")
        print("2. Load audio file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            # Handle recording option
            audio_directory = "audio_samples"
            Path(audio_directory).mkdir(parents=True, exist_ok=True)
            audio_path = os.path.join(audio_directory, "recording.wav")
            
            # Get recording duration from user
            try:
                duration = float(input("\nEnter recording duration in seconds (default 5): ") or 5)
            except ValueError:
                duration = 5
            
            record_audio(audio_path, duration=duration)
            
        elif choice == "2":
            # Handle file loading option
            audio_directory = "audio_samples"
            test_audio = "test_audio.mp3"
            audio_path = os.path.abspath(os.path.join(audio_directory, test_audio))
            
            if not os.path.isfile(audio_path):
                print(f"\nError: Audio file not found at {audio_path}")
                continue
            
        elif choice == "3":
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
            continue
        
        # Process the audio
        try:
            # Load medium model for better multilingual support
            transcriber.load_model("medium")
            
            # Load and preprocess audio
            audio_data, _ = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)
            
            # Perform transcription
            result = transcriber.transcribe_audio(audio_path)
            
            # Display results
            print(f"\nResults:")
            print("-" * 50)
            print(f"Detected Language: {result['language']}")
            print("\nTranscribed Text:")
            print(result["text"])
            
            # Handle translation for non-English audio
            if result['language'] != 'en':
                translation = transcriber.current_model.transcribe(
                    audio_data,
                    task="translate",
                    fp16=False,
                    verbose=None
                )
                print("\nEnglish Translation:")
                print(translation["text"])
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()
