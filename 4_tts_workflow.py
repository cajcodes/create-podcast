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
    """Load TTS models with proper configuration for M3"""
    print("Loading models...")
    
    # Load Bark
    bark_model = BarkModel.from_pretrained("suno/bark").to(device)
    bark_processor = AutoProcessor.from_pretrained("suno/bark")
    
    # Remove Parler and use Bark for both speakers
    return bark_model, bark_processor

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

def generate_speaker1_audio(text, bark_model, bark_processor, device):
    """Generate audio using Bark (Speaker 1)"""
    try:
        print(f"Processing text: {text[:50]}...")
        
        # Step 1: Process inputs
        print("Processing inputs...")
        inputs = bark_processor(
            text=[text],
            voice_preset="v2/en_speaker_9",
            return_tensors="pt",
        )
        
        # Step 2: Move to device and ensure correct dtypes
        input_ids = inputs["input_ids"].to(dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        # Step 3: Generate audio
        print("Generating audio...")
        with torch.no_grad():
            audio_array = bark_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=256,
                temperature=1.0,
                use_cache=True,
                pad_token_id=0  # Explicitly set pad_token_id
            )
        
        # Step 4: Post-process
        print("Post-processing audio...")
        audio_array = audio_array.cpu().numpy().squeeze()
        
        print("Audio generation successful!")
        return audio_array
    except Exception as e:
        print(f"Error generating Speaker 1 audio: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def generate_speaker2_audio(text, bark_model, bark_processor, device):
    """Generate audio using Bark (Speaker 2)"""
    try:
        print(f"Processing text: {text[:50]}...")
        
        # Step 1: Process inputs
        print("Processing inputs...")
        inputs = bark_processor(
            text=[text],
            voice_preset="v2/en_speaker_6",
            return_tensors="pt",
        )
        
        # Step 2: Move to device and ensure correct dtypes
        input_ids = inputs["input_ids"].to(dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        # Step 3: Generate audio
        print("Generating audio...")
        with torch.no_grad():
            audio_array = bark_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=256,
                temperature=1.0,
                use_cache=True,
                pad_token_id=0  # Explicitly set pad_token_id
            )
        
        # Step 4: Post-process
        print("Post-processing audio...")
        audio_array = audio_array.cpu().numpy().squeeze()
        
        print("Audio generation successful!")
        return audio_array
    except Exception as e:
        print(f"Error generating Speaker 2 audio: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    # Setup
    device = setup_device()
    bark_model, bark_processor = load_models(device)
    
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
    
    # Process each segment
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
            audio_array = generate_speaker1_audio(text, bark_model, bark_processor, device)
        else:
            audio_array = generate_speaker2_audio(text, bark_model, bark_processor, device)
            
        if audio_array is not None and len(audio_array) > 0:
            try:
                audio_segment = numpy_to_audio_segment(audio_array, 24000)
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
    
    # Check if we have any successful segments before exporting
    print(f"\nGeneration complete. Successful segments: {successful_segments}, Failed segments: {failed_segments}")
    
    if successful_segments > 0:
        print("\nExporting podcast...")
        final_podcast.export("_podcast.mp3", format="mp3", bitrate="192k")
        print("Podcast export complete!")
    else:
        print("\nError: No audio segments were successfully generated. Cannot create podcast file.")

if __name__ == "__main__":
    main()
