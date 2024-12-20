import requests
import json
import time
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

SYSTEM_PROMPT = """You are a world-class podcast writer. Your task is to convert educational text into natural-sounding podcast dialogue between two speakers:

Speaker 1 is the lead teacher who explains concepts and shares anecdotes.
Speaker 2 is a curious learner who asks insightful follow-up questions.

Make the dialogue sound natural by:
- Including filler words (um, hmm)
- Adding brief interruptions
- Using conversational language
- Including natural back-and-forth
- Adding relevant examples and analogies

Format the output as:
Speaker 1: [dialogue]
Speaker 2: [dialogue]

Keep the technical accuracy but make it engaging and conversational."""

def read_file_to_string(file_path):
    """Attempts to read file content with multiple encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file with any of the encodings: {encodings}")

def generate_transcript(input_text):
    """Generates podcast transcript using LM Studio's local API."""
    
    # LM Studio API endpoint (default)
    url = "http://localhost:1234/v1/chat/completions"
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Convert this text into a natural podcast dialogue: {input_text}"}
    ]
    
    # API request parameters
    data = {
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 4000,  # Adjust based on your model and needs
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for bad status codes
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        print("Make sure LM Studio is running and the API is enabled!")
        return None

def main():
    # Create resources directory if it doesn't exist
    Path("./resources").mkdir(exist_ok=True)
    
    # Read input file (adjust path as needed)
    input_path = "clean_podcast.txt"  # Update this path
    try:
        input_text = read_file_to_string(input_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Generate transcript
    print("Generating transcript...")
    transcript = generate_transcript(input_text)
    
    if transcript:
        # Save output
        output_path = "./resources/data.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(transcript, f)
        print(f"\nTranscript saved to {output_path}")
        
        # Print preview
        print("\nPreview of generated transcript:")
        print(transcript[:500] + "...")
    else:
        print("Failed to generate transcript")

if __name__ == "__main__":
    main()

