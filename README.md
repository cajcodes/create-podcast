# AI Podcast Generator

A comprehensive tool that converts PDF documents into natural-sounding podcast conversations using AI text processing and text-to-speech technology.

## Features

- PDF text extraction and preprocessing
- Conversion of academic/technical content into natural dialogue
- Multiple TTS options (OpenAI, ElevenLabs, or Bark)
- Dynamic audio processing with crossfading and ambient effects
- Progress tracking and error handling

## Prerequisites

- Python 3.8+
- PyTorch
- An API key for your chosen TTS service:
  - OpenAI API key for OpenAI TTS
  - ElevenLabs API key for ElevenLabs TTS
  - (Bark runs locally)

## Installation

1. Clone the repository:
bash
git clone <repository-url>
cd ai-podcast-generator

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_key_here
ELEVEN_API_KEY=your_elevenlabs_key_here
```

## Usage

1. Place your PDF file in the `resources` directory.

2. Run the preprocessing step:
```bash
python 1_pre_processing_logic.py
```

3. Convert to dialogue:
```bash
python 3_re_writer.py
```

4. Generate audio using your preferred TTS service:

For OpenAI TTS:
```bash
python 4_tts_openai.py
```

For ElevenLabs:
```bash
python 4_tts_eleven.py
```

For Bark (local):
```bash
python 4_tts_workflow.py
```

The final podcast will be saved as `_podcast.mp3` in the project root directory.

## Configuration

- Adjust chunk sizes in `1_pre_processing_logic.py`
- Modify speaker personalities in `3_re_writer.py`
- Customize audio processing parameters in the TTS scripts

## Project Structure

- `1_pre_processing_logic.py`: PDF text extraction and cleaning
- `3_re_writer.py`: Converts text to natural dialogue
- `4_tts_openai.py`: OpenAI TTS implementation
- `4_tts_eleven.py`: ElevenLabs TTS implementation
- `4_tts_workflow.py`: Local Bark TTS implementation
- `resources/`: Directory for input files and temporary data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT and TTS services
- ElevenLabs for voice synthesis
- Suno for the Bark text-to-speech model
- PyPDF2 for PDF processing
- PyDub for audio manipulation

## Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Ensure PDF is not password protected
   - Check if PDF contains extractable text

2. **API Key Errors**
   - Verify API keys are correctly set in `.env`
   - Check API quota limits

3. **Memory Issues**
   - Reduce chunk size in preprocessing
   - Clear cache between generations

### Getting Help

For issues and support:
1. Check the issues section in the repository
2. Create a new issue with detailed information
3. Include error messages and system information
