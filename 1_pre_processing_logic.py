import PyPDF2
import requests
import json
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Dict
import time

# API Configuration
API_URL = "http://localhost:1234/v1/chat/completions"  # Default LM Studio API endpoint
CHUNK_SIZE = 1000  # words per chunk

def validate_pdf(file_path: str) -> bool:
    """Validate if file exists and is PDF."""
    path = Path(file_path)
    return path.exists() and path.suffix.lower() == '.pdf'

def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> str:
    """Extract text from PDF with progress tracking."""
    if not validate_pdf(file_path):
        raise ValueError("Invalid PDF file")
    
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Extracting text...", total=None)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text + "\n"
                
                if len(text) > max_chars:
                    print(f"Warning: Truncating text to {max_chars} characters")
                    return text[:max_chars]
    
    return text

def create_word_bounded_chunks(text: str, target_chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of approximately target_chunk_size words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += 1
        
        if current_size >= target_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_chunk(chunk: str) -> str:
    """Process a single chunk of text through the local LLM."""
    system_prompt = """You are a helpful assistant that cleans up text extracted from PDFs. Your task is to:
    1. Remove LaTeX math notation
    2. Fix formatting issues and remove unnecessary whitespace
    3. Remove references, citations, and footnotes
    4. Make the text more suitable for podcast transcription
    5. Preserve the main content and meaning
    Only output the cleaned text without any explanations or additional commentary."""
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Clean this text: {chunk}"}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: API returned status code {response.status_code}")
            return chunk
            
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return chunk

def process_pdf(pdf_path: str, output_path: str = None):
    """Process entire PDF and save cleaned text."""
    if output_path is None:
        output_path = f"clean_{Path(pdf_path).stem}.txt"
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    # Create chunks
    print("Creating chunks...")
    chunks = create_word_bounded_chunks(text)
    
    # Process chunks
    print(f"Processing {len(chunks)} chunks through LLM...")
    with Progress() as progress:
        task = progress.add_task("Processing chunks...", total=len(chunks))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                cleaned_chunk = process_chunk(chunk)
                f.write(cleaned_chunk + "\n\n")
                progress.update(task, advance=1)
                time.sleep(0.1)  # Prevent overwhelming the API
    
    print(f"Processing complete! Cleaned text saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    pdf_path = "./resources/podcast.pdf"
    process_pdf(pdf_path)
