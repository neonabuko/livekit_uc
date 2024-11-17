# LiveKit Use Case - AI Assistant with Voice

A real-time AI assistant implementation using LiveKit for audio streaming and ChatGPT for conversational AI. This project combines text-to-speech and natural language processing to create an interactive voice-based AI assistant.

## Features

- Real-time streaming conversation with ChatGPT
- Text-to-Speech synthesis using gTTS (Google Text-to-Speech)
- Audio streaming through LiveKit
- Asynchronous architecture for efficient real-time communication

## Components

- `chatgpt_client.py`: Handles streaming communication with ChatGPT API
- `my_llm.py`: LLM (Language Learning Model) implementation for chat functionality
- `my_tts.py`: Text-to-Speech service implementation using gTTS
- `main.py`: Main application entry point and LiveKit integration

## Prerequisites

- Python 3.8+
- LiveKit server setup
- ChatGPT API access

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
CHATGPT_API_KEY=your_api_key_here
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
```

## Usage

1. Start the LiveKit server
2. Run the application:
```bash
python main.py
```

## Project Structure

```
.
├── main.py              # Main application entry point
├── chatgpt_client.py    # ChatGPT streaming client
├── my_llm.py           # LLM implementation
├── my_tts.py           # Text-to-Speech service
└── .gitignore          # Git ignore file
```

## Features in Detail

- **Streaming Chat**: Implements server-sent events (SSE) for real-time chat responses
- **Voice Synthesis**: Converts text responses to speech using Google's TTS service
- **LiveKit Integration**: Handles real-time audio streaming and WebRTC communication
- **Error Handling**: Comprehensive error handling for network and service issues

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
