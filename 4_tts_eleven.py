import torch
import gc
import numpy as np
from transformers import BarkModel, AutoProcessor
from pydub import AudioSegment
import io
from tqdm import tqdm
import warnings
import pickle
import time
import traceback
from elevenlabs import api, Voice, generate as eleven_generate
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from random import uniform
import librosa

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_device():
    """Configure the device for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")
    return device

def load_models(device):
    """Load API key for ElevenLabs"""
    print("Setting up ElevenLabs...")
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('ELEVEN_API_KEY')  # Updated env variable name
    if not api_key:
        raise ValueError("ELEVEN_API_KEY environment variable not set")
    api.api_key = api_key  # Updated way to set API key
    return None, None

def clear_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
def numpy_to_audio_segment(numpy_array, sample_rate):
    """Convert numpy array to AudioSegment"""
    # Ensure the array is in the correct format
    audio_array = (numpy_array * 32767).astype(np.int16)
    
    # Convert to bytes
    byte_io = io.BytesIO()
    byte_io.write(audio_array.tobytes())
    byte_io.seek(0)
    
    return AudioSegment.from_raw(
        byte_io, 
        sample_width=2, 
        frame_rate=sample_rate, 
        channels=1
    )

def change_audio_properties(audio_segment, rate=1.0, pitch=0.0):
    """Adjust speech rate and pitch"""
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_segment.frame_rate

    if rate != 1.0:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=int(sample_rate * rate))
        sample_rate = int(sample_rate * rate)

    if pitch != 0.0:
        samples = librosa.effects.pitch_shift(samples, sr=sample_rate, n_steps=pitch)

    samples = np.clip(samples, -32768, 32767)
    samples = samples.astype(np.int16)

    return AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    ).normalize()

def generate_speaker_audio(text, voice_id, *args):
    """Generic function to generate audio using ElevenLabs"""
    try:
        print(f"Processing text: {text[:50]}...")
        
        audio_array = eleven_generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # Convert bytes to audio segment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_array), format="mp3")
        
        # Add natural variation
        audio_segment = change_audio_properties(
            audio_segment,
            rate=uniform(0.95, 1.05),
            pitch=uniform(-0.5, 0.5)
        )
        
        audio_array = np.array(audio_segment.get_array_of_samples())
        
        print("Audio generation successful!")
        return audio_array
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

# Replace existing speaker functions with calls to generic function
def generate_speaker1_audio(text, *args):
    return generate_speaker_audio(text, "JBFqnCBsd6RMkjVDRZzb")

def generate_speaker2_audio(text, *args):
    return generate_speaker_audio(text, "vdrGair91sJJuWWlib6J")

def main():
    # Setup
    device = setup_device()
    
    # Load podcast data
    try:
        with open('resources/rewritten_data.pkl', 'rb') as f:
            podcast_segments = pickle.load(f)
            print(f"\nLoaded {len(podcast_segments)} segments from rewritten_data.pkl")
            # Debug: Print first segment
            if podcast_segments:
                print("First segment preview:")
                print(podcast_segments[0])
    except FileNotFoundError:
        print("Error: rewritten_data.pkl not found")
        return
    except Exception as e:
        print(f"Error loading podcast data: {e}")
        return

    if not podcast_segments:
        print("Error: No segments found in rewritten_data.pkl")
        return

    # Initialize final audio
    final_podcast = AudioSegment.empty()
    
    # Process each segment with crossfade
    print("\nGenerating podcast segments...")
    successful_segments = 0
    failed_segments = 0
    
    for i, segment in enumerate(tqdm(podcast_segments)):
        speaker = segment['speaker']
        text = segment['text']
        
        print(f"\nProcessing segment {i+1}/{len(podcast_segments)}")
        print(f"Speaker: {speaker}")
        print(f"Text: {text[:50]}...")  # Print first 50 chars of text
        
        # Add a small pause between segments
        if i > 0:
            final_podcast += AudioSegment.silent(duration=500)
        
        # Generate audio based on speaker
        audio_array = None
        if speaker == "Speaker 1":
            audio_array = generate_speaker1_audio(text)
        else:
            audio_array = generate_speaker2_audio(text)
            
        if audio_array is not None and len(audio_array) > 0:
            try:
                audio_segment = numpy_to_audio_segment(audio_array, 24000)
                
                # Add crossfade between segments
                if i > 0:
                    final_podcast = final_podcast.append(audio_segment, crossfade=300)
                    # Add natural pause
                    pause_duration = uniform(200, 800)
                    final_podcast += AudioSegment.silent(duration=pause_duration)
                else:
                    final_podcast += audio_segment
                
                successful_segments += 1
                print(f"Successfully generated audio for segment {i+1}")
            except Exception as e:
                print(f"Error converting segment {i+1} to audio: {e}")
                failed_segments += 1
        else:
            print(f"Failed to generate audio for segment {i+1}")
            failed_segments += 1
        
        # Clear memory after each segment
        clear_memory()
        time.sleep(1)
    
    # Normalize final audio
    if successful_segments > 0:
        print("\nNormalizing and exporting podcast...")
        final_podcast = final_podcast.normalize()
        final_podcast.export("final_podcast.mp3", format="mp3", bitrate="192k")
        print("Podcast export complete!")
    else:
        print("\nError: No audio segments were successfully generated. Cannot create podcast file.")

if __name__ == "__main__":
    main()
