# 🎙️ Whisper Audio Transcription App

A powerful audio transcription application that uses OpenAI's Whisper model to convert speech to text, with additional features like sentiment analysis, key moment detection, and AI-powered responses.

## ✨ Features

- **Audio Input Options**
  - Upload audio files (WAV, MP3, M4A, OGG, FLAC)
  - Record audio directly in the browser
  
- **Transcription Capabilities**
  - Multiple language support
  - High accuracy transcription
  - Real-time processing
  - Confidence scoring

- **Advanced Analysis**
  - Sentiment analysis
  - Key moment detection
  - AI-powered responses using Google's Gemini
  - Audio visualization

- **Export Options**
  - Download transcriptions as TXT
  - Export with timestamps (SRT format)
  - Full metadata export (JSON)
  - Text-to-speech conversion

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/saadrehman171000/Whisper-Audio-Transcription.git
    cd Whisper-Audio-Transcription
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment:
   - Create a `.streamlit/secrets.toml` file
   - Add your Gemini API key:
    ```toml
    GEMINI_API_KEY = "your-api-key-here"
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Speech Recognition**: OpenAI Whisper
- **AI Integration**: Google Gemini
- **Audio Processing**: 
  - librosa
  - soundfile
  - sounddevice
- **Text-to-Speech**: gTTS
- **Data Analysis**: numpy, scipy

## 📦 Model Options

- **Tiny**: Fastest (32x) but less accurate
- **Base**: Better accuracy with good speed (16x)
- **Small**: More accurate but slower (8x)
- **Medium**: Most accurate but slowest (4x)

## 🔧 Configuration

The app supports various configuration options:
- Model selection for different accuracy/speed tradeoffs
- Multiple language support for transcription
- Customizable audio recording settings
- Adjustable visualization parameters

## 📝 Usage

1. **Select Model**: Choose between different Whisper models based on your needs
2. **Input Method**: Select either file upload or audio recording
3. **Process**: Wait for the transcription and analysis
4. **Results**: View transcription, analysis, and AI insights
5. **Export**: Download results in your preferred format

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Google for the Gemini API
- Streamlit for the web framework
- All other open-source contributors

## 📞 Support

For support, please open an issue in the GitHub repository or contact saadrehman17100@gmail.com

---
Made with ❤️ using OpenAI's Whisper
