import pickle
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
from pydub.effects import normalize, compress_dynamic_range
import numpy as np

def generate_audio(text, voice, client):
    """Generate audio using OpenAI's TTS API"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        return response
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

def process_audio_segment(audio_segment, speaker):
    """Apply audio processing chain to a segment"""
    # Normalize first
    audio_segment = normalize(audio_segment)
    
    # Apply speaker-specific EQ using low_pass_filter and high_pass_filter
    if speaker == "Speaker 1":  # Main host voice
        audio_segment = (audio_segment
            .low_pass_filter(4000)  # Reduce sibilance
            .high_pass_filter(100)  # Keep warmth
        )
    else:  # Guest voice
        audio_segment = (audio_segment
            .low_pass_filter(5000)  # Subtle high cut
            .high_pass_filter(150)  # Different warmth
        )
    
    # Apply compression
    audio_segment = compress_dynamic_range(
        audio_segment,
        threshold=-20,
        ratio=2.5,
        attack=5,
        release=50
    )
    
    return audio_segment

def calculate_crossfade_duration(curr_segment, next_segment, base_duration=300):
    """Calculate dynamic crossfade duration based on content"""
    # Longer crossfade for same speaker, shorter for different speakers
    if curr_segment['speaker'] == next_segment['speaker']:
        return min(500, base_duration * 1.5)
    return base_duration

def add_room_ambience(audio_segment, volume=-50):
    """Enhanced room ambience with dynamic adjustment"""
    duration_ms = len(audio_segment)
    
    # Create room tone with slight variations
    room_noise = AudioSegment.silent(duration=duration_ms).overlay(
        AudioSegment.from_file("./resources/room.wav", format="wav"),
        loop=True
    )
    
    # Add subtle volume variations to make it more natural
    chunks = len(room_noise) // 1000  # 1-second chunks
    for i in range(chunks):
        chunk_volume = volume + np.random.uniform(-3, 3)
        start_ms = i * 1000
        end_ms = start_ms + 1000
        room_noise_chunk = room_noise[start_ms:end_ms].apply_gain(chunk_volume)
        if i == 0:
            dynamic_room_noise = room_noise_chunk
        else:
            dynamic_room_noise = dynamic_room_noise.append(room_noise_chunk, crossfade=50)
    
    return audio_segment.overlay(dynamic_room_noise)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client with API key
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load podcast data
    try:
        with open('./resources/rewritten_data.pkl', 'rb') as f:
            podcast_segments = pickle.load(f)
            print(f"\nLoaded {len(podcast_segments)} segments from rewritten_data.pkl")
    except FileNotFoundError:
        print("Error: rewritten_data.pkl not found")
        return
    except Exception as e:
        print(f"Error loading podcast data: {e}")
        return

    if not podcast_segments:
        print("Error: No segments found in rewritten_data.pkl")
        return

    # Initialize final audio with fade-in room tone
    room_tone_intro = AudioSegment.silent(duration=2000)
    room_tone_intro = add_room_ambience(room_tone_intro, volume=-45)
    final_podcast = room_tone_intro.fade_in(1000)
    
    # Keep track of previous segment for smart transitions
    prev_segment = None
    
    # Process each segment
    print("\nGenerating podcast segments...")
    successful_segments = 0
    failed_segments = 0
    
    for i, segment in enumerate(tqdm(podcast_segments)):
        speaker = segment['speaker']
        text = segment['text']
        
        print(f"\nProcessing segment {i+1}/{len(podcast_segments)}")
        
        # Generate audio based on speaker
        voice = "alloy" if speaker == "Speaker 1" else "echo"
        response = generate_audio(text, voice, client)
            
        if response:
            try:
                temp_path = Path(f"temp_segment_{i}.mp3")
                response.stream_to_file(temp_path)
                audio_segment = AudioSegment.from_mp3(temp_path)
                
                # Apply processing chain
                audio_segment = process_audio_segment(audio_segment, speaker)
                
                # Add natural room ambience
                audio_segment = add_room_ambience(audio_segment)
                
                # Dynamic crossfade based on content
                if prev_segment:
                    crossfade_duration = calculate_crossfade_duration(prev_segment, segment)
                    final_podcast = final_podcast.append(audio_segment, crossfade=crossfade_duration)
                else:
                    final_podcast = final_podcast.append(audio_segment, crossfade=500)
                
                prev_segment = segment
                temp_path.unlink()
                successful_segments += 1
                
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")
                failed_segments += 1
        
        # Dynamic rate limiting based on segment length
        sleep_time = max(0.5, len(text) / 1000)  # Longer sleep for longer segments
        time.sleep(sleep_time)
    
    # Add fade out to the end
    final_podcast = final_podcast.fade_out(2000)
    
    # Final master processing
    if successful_segments > 0:
        print("\nApplying final mastering...")
        final_podcast = compress_dynamic_range(
            final_podcast,
            threshold=-15,
            ratio=1.5,
            attack=10,
            release=100
        )
        final_podcast = normalize(final_podcast)
        
        print("\nExporting podcast...")
        final_podcast.export(
            "_podcast.mp3",
            format="mp3",
            bitrate="192k",
            tags={
                'title': 'AI Generated Podcast',
                'artist': 'OpenAI TTS',
                'album': 'AI Podcasts',
                'date': time.strftime('%Y-%m-%d')
            }
        )
        print("Podcast export complete!")
    else:
        print("\nError: No audio segments were successfully generated.")

if __name__ == "__main__":
    main()
